#!/usr/bin/env python3
"""推理态固化模型 - BitNet b1.58 的 cim::matmul custom op 2bit 三值权重前向。

把训练态 BitLinear (STE 伪量化 + F.linear) 替换为推理态 BitLinearInference:
  norm -> per-token int8 量化 -> cim::matmul (int8 × 2bit 打包 -> int32) -> rescale

权重以 2bit 补码打包 uint8[N, K//4] 常量固化 (4 code/byte, -1->0b11, 0->0b00, +1->0b01)。
矩阵乘用注册的 cim::matmul custom op (见 cim_op.py), torch.export 保留为 op 节点不内联,
使 CPU/CIM 在 IR 里天然分离 (CPU 量化/rescale, CIM=cim.matmul op)。

对应 cim_mlp.md 的 CIM 定点通路:
  int8 激活 × 2bit 节点 -> int32 累加 -> CPU rescale (FP32 留 CPU 侧, 不写回共享缓存)。
"""
import torch
import torch.nn as nn

from bitnet.model import BitNet, BitLinear, RotaryMHA
from torch.nn.functional import scaled_dot_product_attention
from cim_compiler.export import cim_op  # noqa: F401  (注册 cim::matmul custom op)


class BitLinearInference(nn.Module):
    """推理态 BitLinear: 2bit 打包三值权重 + cim::matmul 定点前向。

    持有原 BitLinear 的 nn.RMSNorm (norm 子模块)、2bit 打包权重 buffer (w_packed)、
    每张量 scale_w 标量。forward 对应 CIM 定点通路:
      norm -> per-token int8 量化 -> cim::matmul (int8 × 2bit 打包 -> int32) -> rescale (FP32)
    """
    def __init__(self, norm: nn.Module, w_packed: torch.Tensor, scale_w: float):
        super().__init__()
        self.norm = norm
        self.register_buffer("w_packed", w_packed.to(torch.uint8).cpu())
        self.scale_w = float(scale_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        K = x.shape[-1]
        x_norm = self.norm(x).reshape(-1, K)                                   # [M, K]
        scale_x = 127.0 / x_norm.abs().max(dim=-1, keepdim=True).values.clamp_min(1e-5)  # [M, 1]
        x_int8 = (x_norm * scale_x).round().clamp(-128, 127).to(torch.int8)
        # cim::matmul custom op: int8 × 2bit 打包三值权重 -> int32 (CIM Macro)
        # torch.export 保留为 op 节点不内联, CPU/CIM 在 IR 天然分离 (见 cim_op.py)
        acc = torch.ops.cim.matmul(x_int8, self.w_packed)                       # [M, N] int32
        out = acc.to(torch.float32) / (scale_x * self.scale_w)                 # rescale, FP32 留 CPU 侧
        return out.reshape(*lead, -1)


class RotaryMHAInference(nn.Module):
    """推理态 RotaryMHA + per-layer KV cache (CPU 侧 FP32)。

    复用原 RotaryMHA 的 q/k/v/o_proj (已被 _replace 换成 BitLinearInference)、
    inv_freq 与配置; 加 k_cache/v_cache (动态 python 状态, 非 state_dict)。

    架构 A (S3): K 需 rope (位置相关)、V 需 rescale、attention 用 SDPA, 三者全在
    CPU (CIM 只做 matmul), 故 KV cache 只能在 CPU 侧 -- CIM 持久化 rope/rescale 前
    的 int32 PSUM 语义上不可用 (attention 无法直接消费)。O(n²)->O(n) 由每步只算新
    token 的 K/V proj (CIM matmul M 维 T->1) 自然达成, CIM/lowering 零改动。

    forward:
      - use_cache=False: 全序列 is_causal SDPA (兼容原无状态行为, 数值等价, 供 export)
      - use_cache=True:
          prefill: x=[B,T,d] (T=prompt_len), 算全 K/V, rope(pos 0..T-1), 存 cache, is_causal SDPA
          decode:  x=[B,1,d], 算新 K/V, rope(pos=t), append cache,
                   SDPA(q[1], k_cache[:t+1]) is_causal=False (q 只 1 位置看全历史)

    位置语义: 绝对位置 pos=cache_len..cache_len+T-1。seq_len <= block_size (无滑动
    crop) 时与原模型相对位置 (窗口起点 0) 数值等价; > block_size 的滑动窗口 KV cache
    (cache 滑动 + 位置重算) 列后续。
    """

    def __init__(self, orig: RotaryMHA):
        super().__init__()
        self.n_head = orig.n_head
        self.n_kv = orig.n_kv
        self.head_dim = orig.head_dim
        self.register_buffer("inv_freq", orig.inv_freq)
        self.register_buffer("causal_mask", orig.causal_mask)   # 旧 RotaryMHA buffer (进 .pt2 placeholder)
        # q/k/v/o_proj 已是 BitLinearInference (_replace 先替换 RotaryMHA 内部 BitLinear)
        self.q_proj = orig.q_proj
        self.k_proj = orig.k_proj
        self.v_proj = orig.v_proj
        self.o_proj = orig.o_proj
        self._rep = self.n_head // self.n_kv                  # GQA repeat 因子
        # KV cache (None = 未初始化), shape [B, n_kv, T_past, head_dim]
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None
        self.cache_len = 0

    @staticmethod
    def _rope(t, sin, cos):
        t1, t2 = t[..., ::2], t[..., 1::2]
        return torch.cat([t1 * cos - t2 * sin, t1 * sin + t2 * cos], dim=-1)

    def reset_cache(self):
        """清空 KV cache (新一轮生成前调用)。"""
        self.k_cache = None
        self.v_cache = None
        self.cache_len = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """无 cache 全序列 attention, 与旧 RotaryMHA 数值等价 (export 友好, 供 _LogitsOnly)。

        repeat(dim=2) 在 permute 前 (同旧 RotaryMHA), rope 传 (cos, sin) 复刻原反序调用。
        不含 use_cache 分支/动态 cat, torch.export 不生成额外 shape guard。
        """
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv, self.head_dim)
        k = k.repeat_interleave(self._rep, dim=2)                # GQA 广播 (permute 前, 同旧)
        v = v.repeat_interleave(self._rep, dim=2)
        seq = torch.arange(T, device=x.device, dtype=torch.float32)
        freqs = torch.einsum("t,d->td", seq, self.inv_freq)
        cos, sin = freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]
        # _rope(t, sin, cos) 传 (cos, sin) 复刻旧 RotaryMHA.rope(q, cos, sin) 反序调用
        q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        out = scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """增量 forward (KV cache, 不走 export): prefill 全序列 / decode 单 token。

        每步 rope 用绝对位置 pos=cache_len..cache_len+T-1 (seq_len<=block_size 时
        == 原模型相对位置), K/V append 到 cache, attention 用全 cache。
        decode (T=1) is_causal=False (q 只 1 位置看全历史); prefill (T>1) is_causal=True。
        """
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv, self.head_dim)
        pos = torch.arange(self.cache_len, self.cache_len + T, device=x.device, dtype=torch.float32)
        freqs = torch.einsum("t,d->td", pos, self.inv_freq)
        cos, sin = freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]
        q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if self.k_cache is None:
            self.k_cache, self.v_cache = k, v
        else:
            self.k_cache = torch.cat([self.k_cache, k], dim=2)
            self.v_cache = torch.cat([self.v_cache, v], dim=2)
        self.cache_len += T
        k_full = self.k_cache.repeat_interleave(self._rep, dim=1)   # GQA 广播到 n_head
        v_full = self.v_cache.repeat_interleave(self._rep, dim=1)
        out = scaled_dot_product_attention(q, k_full, v_full, attn_mask=None,
                                           is_causal=(T > 1), dropout_p=0.0)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)

    def step_stateless(self, h: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor,
                       cos: torch.Tensor, sin: torch.Tensor):
        """无状态增量 step (供 export): 单 token + cache 显式 IO。

        h [B,1,d], k_in/v_in [B,T,n_kv,hd] (permute 前, T=0 首步空 cache),
        cos/sin [1,1,1,hd/2] (该 token 绝对位置的 rope cos/sin, 外部按 cache_len 算好传入)。
        -> (out [B,1,d], new_k [B,T+1,n_kv,hd], new_v [B,T+1,n_kv,hd])

        cache 存 permute 前 [B,T,n_kv,hd], repeat(dim=2) 同旧 forward (export 友好,
        避免 repeat dim=1 + 动态 T 的 min shape guard)。CIM 侧 cim.matmul M=1 静态。
        """
        B = h.shape[0]
        q = self.q_proj(h).view(B, 1, self.n_head, self.head_dim)
        k = self.k_proj(h).view(B, 1, self.n_kv, self.head_dim)
        v = self.v_proj(h).view(B, 1, self.n_kv, self.head_dim)
        q = self._rope(q, cos, sin)
        k = self._rope(k, cos, sin)
        new_k = torch.cat([k_in, k], dim=1)                    # [B, T+1, n_kv, hd] (permute 前)
        new_v = torch.cat([v_in, v], dim=1)
        q = q.permute(0, 2, 1, 3)                              # [B, n_head, 1, hd]
        k_full = new_k.repeat_interleave(self._rep, dim=2).permute(0, 2, 1, 3)  # [B, n_head, T+1, hd]
        v_full = new_v.repeat_interleave(self._rep, dim=2).permute(0, 2, 1, 3)
        out = scaled_dot_product_attention(q, k_full, v_full, attn_mask=None,
                                           is_causal=False, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(B, 1, -1)
        return self.o_proj(out), new_k, new_v


class _LogitsOnly(nn.Module):
    """包装 BitNet 只返回 logits (torch-mlir fx.export_and_import 不支持 None 输出)。"""
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # 直接走 BitNet 的 logits 路径 (避免 BitNet.forward 返回 (logits, None), jit.script 不支持)
        x = self.m.embed_tokens(idx)
        for layer in self.m.layers:
            x = layer(x)
        x = self.m.ln_f(x)
        return self.m.lm_head(x)


class _KVCacheModel(nn.Module):
    """无状态 KV cache 增量 forward (供 export): 单 token decode + cache 显式 IO。

    forward(idx[B,1], k_caches[L,B,T,n_kv,hd], v_caches[...], cos[1,1,1,hd/2], sin)
        -> (logits[B,1,vocab], new_k_caches[L,B,T+1,n_kv,hd], new_v_caches[...])
    每层 step_stateless (cache 显式 IO, 存 permute 前), 全程 CIM matmul M=1 (decode 单 token)。
    cos/sin 外部按 cache_len 算好传入 (避免动态 arange); cache T 维动态增长。
    """
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, idx, k_caches, v_caches, cos, sin):
        x = self.m.embed_tokens(idx)                       # [B, 1, d]
        ks = torch.unbind(k_caches, dim=0)                 # L x [B,n_kv,T,hd]
        vs = torch.unbind(v_caches, dim=0)
        new_ks, new_vs = [], []
        for li, layer in enumerate(self.m.layers):
            h = layer.input_ln(x)
            out, nk, nv = layer.attn.step_stateless(h, ks[li], vs[li], cos, sin)
            x = x + out
            h = layer.post_ln(x)
            x = x + layer.mlp(h)
            new_ks.append(nk)
            new_vs.append(nv)
        x = self.m.ln_f(x)
        logits = self.m.lm_head(x)                         # [B, 1, vocab]
        return logits, torch.stack(new_ks, dim=0), torch.stack(new_vs, dim=0)


@torch.no_grad()
def build_inference_model(
    ternary_path: str,
    vocab_size: int,
    d_model: int = 512,
    block_size: int = 256,
    n_layer: int = 6,
    n_head: int = 8,
    n_kv_head: int = 4,
    ffn_dim: int = 1664,
) -> BitNet:
    """从 2bit 打包权重文件构建推理态 BitNet。

    1. 构建 BitNet 骨架 (与训练配置一致)
    2. 加载 base tensors (embed_tokens, norms, SubLayerNorm gamma, 因果 mask, inv_freq)
    3. 递归替换每个 BitLinear -> BitLinearInference (装填 w_packed + scale_w, 保留原 norm)
    """
    model = BitNet(
        vocab_size=vocab_size, d_model=d_model, block_size=block_size,
        n_layer=n_layer, n_head=n_head, n_kv_head=n_kv_head, ffn_dim=ffn_dim,
    )
    data = torch.load(ternary_path, map_location="cpu", weights_only=True)
    base_sd = {k: v for k, v in data.items() if not isinstance(v, dict)}
    model.load_state_dict(base_sd, strict=False)
    model = model.to(torch.float32)

    def _replace(mod: nn.Module, prefix: str = "") -> None:
        for name, child in list(mod.named_children()):
            full = f"{prefix}{name}"
            if isinstance(child, BitLinear):
                key = full + ".weight"
                if key not in data or not isinstance(data[key], dict):
                    raise KeyError(f"ternary checkpoint missing BitLinear entry: {key}")
                e = data[key]
                setattr(mod, name, BitLinearInference(child.norm, e["packed"], e["scale"]))
            elif isinstance(child, RotaryMHA):
                _replace(child, full + ".")               # 先替换其内部 q/k/v/o_proj (BitLinear)
                setattr(mod, name, RotaryMHAInference(child))
            else:
                _replace(child, full + ".")

    _replace(model)
    model.eval()
    return model


@torch.no_grad()
def forward_kv(model: BitNet, idx: torch.Tensor) -> torch.Tensor:
    """带 KV cache 的前向 (不走 export, 直接遍历层)。

    手动展开 Block.forward (input_ln -> attn(use_cache=True) -> residual ->
    post_ln -> mlp -> residual), 把 use_cache 透传到 RotaryMHAInference:
      - prefill (idx=[B,T0], T0>1): 各层建 cache, is_causal SDPA
      - decode  (idx=[B,1]):       各层 append cache, is_causal=False SDPA
    返回 logits [B, T, vocab]。与 _LogitsOnly (use_cache=False 全序列) 数值等价。
    """
    x = model.embed_tokens(idx)
    for layer in model.layers:
        h = layer.input_ln(x)
        x = x + layer.attn.step(h)
        h = layer.post_ln(x)
        x = x + layer.mlp(h)
    x = model.ln_f(x)
    return model.lm_head(x)


@torch.no_grad()
def generate_kv(model: BitNet, idx0: torch.Tensor, n: int, block_size: int = 256) -> list:
    """KV cache 增量 greedy 生成 n token, 返回 token 列表 (含 prompt)。

    prefill prompt 一次 (建 cache) + decode 逐 token (每步 idx=[B,1], CIM matmul
    M=1)。假设 prompt_len + n <= block_size (无滑动 crop): 绝对位置 == 原模型相对
    位置 (窗口起点 0), 与 model.generate 全序列重算 (O(n²)) 数值等价, 但 K/V proj
    CIM matmul 每步 M: T->1, 总计算量 O(n²)->O(n)。
    """
    assert idx0.shape[1] + n <= block_size, (
        f"KV cache 当前仅支持 seq_len<=block_size (无 crop): "
        f"{idx0.shape[1]}+{n} > {block_size}")
    for layer in model.layers:                       # 清空各层 cache (新一轮生成)
        if hasattr(layer.attn, "reset_cache"):
            layer.attn.reset_cache()

    tokens = idx0[0].tolist()
    logits = forward_kv(model, idx0)                 # prefill: prompt -> 第 1 个新 token
    nxt = int(logits[0, -1].argmax())
    tokens.append(nxt)
    for _ in range(n - 1):                           # decode: 单 token -> 下一 token (M=1)
        cur = torch.tensor([[nxt]], dtype=torch.long, device=idx0.device)
        logits = forward_kv(model, cur)
        nxt = int(logits[0, -1].argmax())
        tokens.append(nxt)
    return tokens
