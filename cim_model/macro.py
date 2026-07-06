"""CIM Macro: 64×64 矩阵向量乘单元 (2bit 补码三值权重节点)。

对应 cim_mlp.md §1.2 Macro 计算 + §3.2 节点态:
    节点存储 2bit 补码 {-1,0,+1}: -1→0b11, 0→0b00, +1→0b01 (4 code/byte uint8 打包)
    输入 int8 向量 x[64] (K 维一个 64 切片) × 节点 2bit 三值权重
    → 输出 int32 部分和向量 y[64]
        y_j = sum_i W[j,i] * x[i]   (int32, W∈{-1,0,1})

"传输格式即节点态" (§3.2): 宏直接接收 2bit 补码打包权重, 内部按补码语义取三值
{-1,0,1} (0b11→-1, 0b00→0, 0b01→+1), 无 {0,1,2} 偏移编解码。三值×int8 精确整数,
int32 累加无舍入 — 等价理想 CIM 纯整数路径。
"""
import numpy as np

MACRO_SIZE = 64        # CIM Macro 物理维度 64×64
PACKED_COLS = MACRO_SIZE // 4  # 64 个 2bit code → 16 byte/行


def _unpack_2bit_tile(w_packed):
    """节点 2bit 补码 → 三值 int8 (宏内读出, 非偏移编解码)。

    uint8 [R, 16] (4 code/byte, 64 code/行) → int8 [R, 64] ⊆ {-1,0,1}。
    2bit 补码: 0b11→-1, 0b00→0, 0b01→+1 (0b10 未用)。
    """
    p = w_packed.astype(np.int32)
    c0 = p & 0x3
    c1 = (p >> 2) & 0x3
    c2 = (p >> 4) & 0x3
    c3 = (p >> 6) & 0x3
    code = np.stack([c0, c1, c2, c3], axis=-1).reshape(w_packed.shape[0], -1)  # [R, 64]
    return np.where(code >= 2, code - 4, code).astype(np.int8)  # {-1,0,1}


def macro_matmul(x_int8, w_packed):
    """单 Macro: 64×64 矩阵向量乘, int8 × 2bit 补码三值节点 → int32 部分和。

    节点直接存 2bit 补码 (w_packed), 宏内按补码语义取三值 {-1,0,1} 计算 (无偏移编解码)。

    Args:
        x_int8:   int8 [B, 64] — K 维一个 64 切片, B token 批处理
        w_packed: uint8 [64, 16] — 2bit 补码打包三值权重 tile (out, in//4)
                  节点态: -1→0b11, 0→0b00, +1→0b01, 4 code/byte
    Returns:
        int32 [B, 64] — 部分和向量 y (单 tile, **非最终结果**, 需 K 维多 tile 累加)

    数学: Y = X @ W.T,  y[b,j] = sum_i x[b,i] * W[j,i]   (W∈{-1,0,1})
    """
    assert x_int8.shape[-1] == MACRO_SIZE, \
        f"x last dim must be {MACRO_SIZE}, got {x_int8.shape}"
    assert w_packed.shape == (MACRO_SIZE, PACKED_COLS), \
        f"w_packed must be ({MACRO_SIZE},{PACKED_COLS}), got {w_packed.shape}"
    W = _unpack_2bit_tile(w_packed)  # int8 [64,64] {-1,0,1} (节点 2bit 补码读出)
    # int8 → int32 整数 matmul (三值×int8 精确整数, int32 累加无舍入)
    y = x_int8.astype(np.int32) @ W.astype(np.int32).T
    return y  # int32 [B, 64]
