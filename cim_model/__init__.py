"""CIM 定点计算模型 — 验证 cim_mlp.md 描述的定点数据通路。

非指令驱动: 聚焦 Macro(64×64 int8 × 2bit 补码三值节点 → int32) → 累加区(int32 RMW) →
CPU rescale(→fp32, 留 CPU 侧) 的定点计算通路, 不模拟 ISA / 门铃 / 中断 / 共享缓存 PAGE。

权重以 2bit 补码打包 (节点态, 宏内解包, 无偏移编解码); 严格定点 (numpy int32),
不借 fp32 matmul; FP32 中间结果全程 CPU 侧私有, 不写回共享缓存。对应 cim_mlp.md 理想 CIM。
"""
from .macro import macro_matmul, MACRO_SIZE
from .accumulator import Accumulator
from .bitlinear_cim import bitlinear_cim, quantize_activation

__all__ = [
    "macro_matmul",
    "MACRO_SIZE",
    "Accumulator",
    "bitlinear_cim",
    "quantize_activation",
]
