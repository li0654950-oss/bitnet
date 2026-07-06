"""CIM BitLinear: 端到端定点数据通路 (2bit 补码权重节点)。

对应 cim_mlp.md BitLinear 端到端通路 (无偏置):
    1. CPU 量化:  x_fp32 → x_int8 + scale_x (fp32, per-token, 基于原始 x; §4.8)
    2. K 维切 64 tile, 每 tile Macro: int8 × 2bit 补码三值 → int32 部分和 y
    3. 累加区 int32 RMW: acc = Σ y  (完整 int32 结果向量; §2.1)
    4. CPU rescale: out = acc / (scale_x · scale_w) → fp32  (留 CPU 侧, 不写回共享缓存; §4.7)

权重以 2bit 补码打包传入 (节点态, 宏内解包, 无偏移编解码), 无偏置。
scale_x 用 fp32 (CPU 侧理想路径)。
"""
import numpy as np

from .macro import macro_matmul, MACRO_SIZE, PACKED_COLS
from .accumulator import Accumulator


def quantize_activation(x_fp32):
    """per-token int8 量化 (CPU 侧, fp32 scale_x)。

    对应 cim_mlp.md §1.1 CPU 激活量化:
        scale_x = 127 / max|x|,  x_int8 = round(x · scale_x).clamp(-128, 127)

    Args: x_fp32: fp32 [..., K]
    Returns: (x_int8 int8 [..., K], scale_x fp32 [..., 1])
    """
    scale_x = 127.0 / np.maximum(np.abs(x_fp32).max(axis=-1, keepdims=True), 1e-5)
    x_int8 = np.clip(np.round(x_fp32 * scale_x), -128, 127).astype(np.int8)
    return x_int8, scale_x.astype(np.float32)


def bitlinear_cim(x_fp32, w_packed, scale_w):
    """CIM BitLinear: fp32 x → int8 → 64×64 tile Macro(2bit 节点) → int32 累加 → rescale fp32。

    严格定点数据通路 (Macro/累加 int32), rescale 出 fp32 (留 CPU 侧, 不写回共享缓存)。无偏置。

    Args:
        x_fp32:   fp32 [..., K]            (K % 4 == 0)
        w_packed: uint8 [N, K//4]          2bit 补码打包三值权重 (节点态, 宏直接加载)
        scale_w:  float (per-tensor 权重 scale, CPU 侧持有)
    Returns: fp32 [..., N]
    """
    lead = x_fp32.shape[:-1]
    K = x_fp32.shape[-1]
    N = w_packed.shape[0]
    assert K % 4 == 0, f"K={K} must be divisible by 4"
    assert w_packed.shape[1] == K // 4, f"w_packed={w_packed.shape} vs K//4={K//4}"
    x2d = x_fp32.reshape(-1, K).astype(np.float32)  # [M, K]
    M = x2d.shape[0]

    # zero-pad N, K 到 64 倍数 (cim_mlp.md §4.8); pad 值 0 → 2bit 补码 0b00 = 三值 0
    K_pad = ((K + MACRO_SIZE - 1) // MACRO_SIZE) * MACRO_SIZE
    N_pad = ((N + MACRO_SIZE - 1) // MACRO_SIZE) * MACRO_SIZE
    x_pad = np.zeros((M, K_pad), dtype=np.float32)
    x_pad[:, :K] = x2d
    w_pad = np.zeros((N_pad, K_pad // 4), dtype=np.uint8)  # 2bit 补码打包
    w_pad[:N, :K // 4] = w_packed

    # CPU 量化 (scale_x 基于原始 x; §4.8 — pad 维为 0 不影响 absmax, scale_x 不变)
    x_int8, scale_x = quantize_activation(x_pad)  # [M, K_pad] int8, [M, 1] scale_x

    # 64×64 tile 切分 + Macro(2bit 节点) + int32 RMW 累加 + CPU rescale
    n_blocks = N_pad // MACRO_SIZE
    k_blocks = K_pad // MACRO_SIZE
    out = np.zeros((M, N_pad), dtype=np.float32)
    for nb in range(n_blocks):
        acc = Accumulator((M, MACRO_SIZE))  # 该 n 块累加区 (M token 批处理)
        n_lo, n_hi = nb * MACRO_SIZE, (nb + 1) * MACRO_SIZE
        for kb in range(k_blocks):
            x_slice = x_int8[:, kb * MACRO_SIZE:(kb + 1) * MACRO_SIZE]            # [M, 64]
            w_tile = w_pad[n_lo:n_hi, kb * PACKED_COLS:(kb + 1) * PACKED_COLS]     # [64,16] 2bit
            psum = macro_matmul(x_slice, w_tile)  # int32 [M, 64] 部分和
            acc.accumulate(psum)                  # int32 RMW 累加
        # CPU rescale: acc / (scale_x · scale_w) → fp32 (留 CPU 侧, 不写回共享缓存)
        out[:, n_lo:n_hi] = acc.read().astype(np.float32) / (scale_x * scale_w)

    return out[:, :N].reshape(*lead, N)  # 截断 padding
