"""export — BitNet b1.58 模型 → 原生三值权重 FX graph 导出工程。

把 BitNet b1.58 模型导出为「原生三值权重」FX graph, 作为 CIM 硬件 + CPU 异构计算的
编译器前端输入。

  - BitLinearInference: 2bit 打包三值权重 + 原生 ATen 定点前向
      norm → per-token int8 量化 → 2bit 解包 → matmul (int32) → rescale (FP32 留 CPU 侧)
  - torch.export → .pt2: 部署 IR, 动态 seq len [1..block_size], 2bit 权重作常量内嵌, 零 custom op
  - 旁路 .bin: CIM 权重预加载 blob (自描述二进制, Preload 阶段用)

对应 cim_mlp.md 的 CIM 定点通路: int8 激活 × 2bit 节点 → int32 累加 → CPU rescale。
"""
from .inference_model import (
    BitLinearInference,
    build_inference_model,
    unpack_2bit_aten,
)
from .weight_blob import write_weight_blob, read_weight_blob, WeightEntry

__all__ = [
    "BitLinearInference",
    "build_inference_model",
    "unpack_2bit_aten",
    "write_weight_blob",
    "read_weight_blob",
    "WeightEntry",
]
