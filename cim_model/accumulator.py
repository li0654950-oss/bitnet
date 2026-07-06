"""累加区: int32 RMW 累加多 tile 部分和 → acc (完整 int32 结果向量)。

对应 cim_mlp.md §2.1 部分和累加区 (int32 ALU):
    写入端口前置 int32 ALU, 触发原位 RMW: MEM[Addr] += Incoming_Psum (int32)
    K 维多 tile 累加为 acc, 无 bias 预写 (BitLinear 无偏置)。

acc 是 K 维全部 tile 累加后的完整 int32 结果向量 (rescale 前)。
"""
import numpy as np


class Accumulator:
    """累加区: int32 RMW 累加 K 维多 tile 部分和 → acc。

    shape 可为 (64,) (单 token 一个 n 块) 或 (M, 64) (M token 批处理同一 n 块)。
    """

    def __init__(self, shape, dtype=np.int32):
        self.shape = tuple(shape)
        self.acc = np.zeros(self.shape, dtype=dtype)

    def accumulate(self, psum):
        """RMW: acc += psum (int32)。对应 MEM[Addr] += Incoming_Psum。"""
        assert psum.shape == self.shape, f"psum {psum.shape} != acc {self.shape}"
        self.acc = self.acc + psum  # int32 原位累加

    def read(self):
        """读 acc (CPU rescale 前的完整 int32 结果向量)。"""
        return self.acc.copy()

    def reset(self):
        self.acc[:] = 0
