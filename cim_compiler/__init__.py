"""cim_compiler — BitNet → CIM/CPU 异构推理平台的编译器前端。

子工程:
  - export: BitNet b1.58 模型 → 原生三值权重 FX graph (.pt2) + 权重 blob (.bin)
"""
from .export import (
    BitLinearInference,
    build_inference_model,
    unpack_2bit_aten,
    write_weight_blob,
    read_weight_blob,
    WeightEntry,
)

__all__ = [
    "BitLinearInference",
    "build_inference_model",
    "unpack_2bit_aten",
    "write_weight_blob",
    "read_weight_blob",
    "WeightEntry",
]
