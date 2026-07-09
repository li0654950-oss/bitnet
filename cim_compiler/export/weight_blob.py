#!/usr/bin/env python3
"""CIM 权重预加载 blob — 旁路导出每层 BitLinear 的 2bit 打包权重。

与 FX graph (.pt2) 分离:
  - .pt2 给 compiler 做算子调度 (2bit 权重也作常量内嵌于图)
  - .bin 给 CIM 做权重预加载 (Preload 阶段, 见 cim_mlp.md §4.6 两阶段共享缓存)

格式自描述 (小端):
  magic(4)=b"CIMW" | version(4) | num_entries(4)
  每 entry:
    name_len(4) | name(utf-8) | N(4) | K(4) | scale_w(8, f64) | packed_bytes(4) | packed(uint8)
  其中 packed 为 uint8[N, K//4] 的原始字节 (2bit 补码, 4 code/byte)。
"""
import os
import sys
import struct
from dataclasses import dataclass

import torch

# 确保兄弟模块 (inference_model) 可被绝对导入 (直接运行 / 包导入两种模式)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

MAGIC = b"CIMW"
VERSION = 1


@dataclass
class WeightEntry:
    name: str
    N: int
    K: int
    scale_w: float
    packed: bytes  # uint8[N, K//4] 原始字节


def write_weight_blob(model: torch.nn.Module, path: str) -> int:
    """遍历模型中的 BitLinearInference, 写出自描述二进制 blob。返回 entry 数。"""
    from inference_model import BitLinearInference  # 延迟导入避免循环

    entries: list[WeightEntry] = []
    for name, mod in model.named_modules():
        if isinstance(mod, BitLinearInference):
            w = mod.w_packed.cpu().to(torch.uint8).contiguous()
            N, K4 = w.shape
            entries.append(WeightEntry(name, N, K4 * 4, mod.scale_w, w.numpy().tobytes()))

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", VERSION, len(entries)))
        for e in entries:
            name_b = e.name.encode("utf-8")
            f.write(struct.pack("<I", len(name_b)))
            f.write(name_b)
            f.write(struct.pack("<II", e.N, e.K))
            f.write(struct.pack("<d", e.scale_w))
            f.write(struct.pack("<I", len(e.packed)))
            f.write(e.packed)
    return len(entries)


def read_weight_blob(path: str) -> list:
    """读回 blob, 返回 WeightEntry 列表 (验证 / CIM 后端参考)。"""
    entries: list[WeightEntry] = []
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"bad magic {magic!r}, expected {MAGIC!r}")
        version, n = struct.unpack("<II", f.read(8))
        if version != VERSION:
            raise ValueError(f"unsupported blob version {version}, expected {VERSION}")
        for _ in range(n):
            (nl,) = struct.unpack("<I", f.read(4))
            name = f.read(nl).decode("utf-8")
            N, K = struct.unpack("<II", f.read(8))
            (scale_w,) = struct.unpack("<d", f.read(8))
            (bl,) = struct.unpack("<I", f.read(4))
            packed = f.read(bl)
            entries.append(WeightEntry(name, N, K, scale_w, packed))
    return entries
