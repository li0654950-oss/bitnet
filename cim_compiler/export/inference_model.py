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
import cim_op  # noqa: E402, F401  (注册 cim::matmul custom op)


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
