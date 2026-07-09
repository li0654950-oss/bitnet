#!/usr/bin/env python3
"""推理态固化模型 — BitNet b1.58 的纯原生 ATen 2bit 三值权重前向。

把训练态 BitLinear (STE 伪量化 + F.linear) 替换为推理态 BitLinearInference:
  norm → per-token int8 量化 → 2bit 打包权重解包 (原生 ATen) → float matmul (int32) → rescale

权重以 2bit 补码打包 uint8[N, K//4] 常量固化 (4 code/byte, -1→0b11, 0→0b00, +1→0b01)。
整条前向是纯原生 ATen 算子, 可被 torch.export 捕获, 零 custom op 注册。

对应 cim_mlp.md 的 CIM 定点通路:
  int8 激活 × 2bit 节点 → int32 累加 → CPU rescale (FP32 留 CPU 侧, 不写回共享缓存)。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# HERE = cim_compiler/export/ ; 上两层 = repo root ; repo root/bitnet = bitnet 包
BITNET = os.path.join(os.path.dirname(os.path.dirname(HERE)), "bitnet")
if BITNET not in sys.path:
    sys.path.insert(0, BITNET)

import torch
import torch.nn as nn

from model import BitNet, BitLinear  # noqa: E402


def unpack_2bit_aten(packed: torch.Tensor) -> torch.Tensor:
    """uint8[..., K//4] (2bit 补码) -> int8[..., K] {-1,0,1}。

    纯原生 ATen 算子链 (可被 torch.export 捕获):
      bit_and(0x3) + rshift(2/4/6) → 4 路取 code → stack/reshape → where(code>=2, code-4, code)
    与 export_ternary.unpack_2bit 数值完全一致 (已验证 max|diff|=0)。
    """
    p = packed.to(torch.int32)
    # 用 div/mod 替代 bitwise (torch-mlir 20240127 对 int bitwise_and/rshift 降级不全; p 非负, div/mod 等价)
    c0 = p % 4
    c1 = (p // 4) % 4
    c2 = (p // 16) % 4
    c3 = (p // 64) % 4
    code = torch.stack([c0, c1, c2, c3], dim=-1).reshape(*packed.shape[:-1], -1)
    # 2bit 补码: 0->0, 1->+1, 3->-1 (2 未用)
    return torch.where(code >= 2, code - 4, code).to(torch.int8)


class BitLinearInference(nn.Module):
    """推理态 BitLinear: 2bit 打包三值权重 + 原生 ATen 定点前向。

    持有原 BitLinear 的 nn.RMSNorm (norm 子模块)、2bit 打包权重 buffer (w_packed)、
    每张量 scale_w 标量。forward 对应 CIM 定点通路:
      norm → per-token int8 量化 → 2bit 解包 → float matmul (int32) → rescale (FP32)
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
        w_int8 = unpack_2bit_aten(self.w_packed)                                 # [N, K] {-1,0,1}
        # float matmul 替代 _int_mm (torch-mlir 对 _int_mm 降级不全;
        # int8×{-1,0,1} 在 float32 精确 |acc|<=65536<2^24, 数值等价 _int_mm int32)
        acc = (x_int8.to(torch.float32) @ w_int8.to(torch.float32).t()).to(torch.int32)
        out = acc.to(torch.float32) / (scale_x * self.scale_w)                 # rescale, FP32 留 CPU 侧
        return out.reshape(*lead, -1)


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
            else:
                _replace(child, full + ".")

    _replace(model)
    model.eval()
    return model
