"""CIM BitLinear 适配层: monkey-patch 复用 model.py 全部结构, 仅替换 BitLinear.forward。

接口同 model.BitLinear (norm + set_inference), 但 forward 走 CIM 定点通路
(numpy int32: int8 量化 → 64×64 tile Macro(2bit 节点) → int32 累加 → fp32 rescale)。

非线性 (RMSNorm/SubLayerNorm/RotaryMHA/SDPA/RoPE/ReLU²/embed/generate) 全部复用
model.py 原实现, 仅在 BitLinear 边界做 numpy↔torch 转换。

对应 cim_mlp.md 的理想 CIM 硬件精度模型 (int8/int32/fp32, 无 bf16):
- 权重: 2bit 补码打包 (节点态, 宏内解包), 无需外部 unpack
- FP32 中间结果留 CPU 侧主存, 不写回共享缓存 (§4.7)
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)                            # 工程根: import cim_model
sys.path.insert(0, os.path.join(ROOT, "bitnet"))    # bitnet/: import model

from cim_model.bitlinear_cim import bitlinear_cim


def cim_bitlinear_forward(self, x):
    """CIM 定点 forward (替换 model.BitLinear.forward)。

    数据流: x(bf16/fp32) → RMSNorm → fp32 numpy → CIM 定点 (int8→int32→fp32) → 原 dtype 回 torch

    权重以 2bit 补码打包 (self._ternary_packed, 节点态) 直传 CIM, 宏内解包。
    """
    x_norm = self.norm(x)  # RMSNorm (model.py 内置)

    if self._ternary_packed is None:
        # 未加载三值权重: 走原 STE 伪量化路径 (训练/参考用)
        from torch.nn import functional as F
        from model import activation_quant, weight_quant
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = self.weight + (weight_quant(self.weight) - self.weight).detach()
        return F.linear(x_quant, w_quant)

    # CIM 定点通路: torch → numpy fp32 → CIM (int8 量化 / 2bit 节点 / int32 累加 / fp32 rescale) → torch
    x_np = x_norm.detach().to(torch.float32).cpu().numpy()
    w_packed = self._ternary_packed.cpu().numpy()  # uint8 [N, K//4] 2bit 补码 (节点态)
    y_np = bitlinear_cim(x_np, w_packed, self._scale_w)  # fp32 [..., N] (留 CPU 侧)
    return torch.from_numpy(y_np).to(x.dtype).to(x.device)


def patch_bitlinear():
    """把 model.BitLinear.forward 替换为 CIM 定点版。返回原 forward 以便恢复。"""
    import model
    orig = model.BitLinear.forward
    model.BitLinear.forward = cim_bitlinear_forward
    return orig


def unpatch_bitlinear(orig_forward):
    """恢复原 BitLinear.forward。"""
    import model
    model.BitLinear.forward = orig_forward


def load_ternary_into_model(model, ternary_path, device):
    """从 ternary.pt 加载三值权重到 model 的 BitLinear (set_inference 模式)。

    权重以 2bit 补码打包存于 _ternary_packed (节点态), CIM forward 直传宏内解包。
    返回加载的 BitLinear 数量。
    """
    from model import BitLinear
    data = torch.load(ternary_path, map_location=device, weights_only=True)
    base_sd = {k: v for k, v in data.items() if not isinstance(v, dict)}
    model.load_state_dict(base_sd, strict=False)  # embed_tokens, norms, buffers
    n = 0
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            key = name + ".weight"
            if key in data and isinstance(data[key], dict):
                module.set_inference(data[key]["packed"].to(device), data[key]["scale"])
                n += 1
    return n
