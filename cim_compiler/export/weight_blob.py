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
import struct
from dataclasses import dataclass
from typing import BinaryIO

import torch

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
    from cim_compiler.export.inference_model import BitLinearInference  # 延迟导入避免循环

    entries: list[WeightEntry] = []
    for name, mod in model.named_modules():
        if isinstance(mod, BitLinearInference):
            w = mod.w_packed.cpu().to(torch.uint8).contiguous()
            if w.ndim != 2:
                raise ValueError(f"{name}: w_packed ndim={w.ndim}, expected 2")
            N, K4 = w.shape
            if N == 0 or K4 == 0:
                raise ValueError(f"{name}: empty w_packed shape {tuple(w.shape)}")
            K = K4 * 4
            packed = w.numpy().tobytes()
            if len(packed) != N * K // 4:           # 对称校验: 确保写出的能被 read 校验过
                raise ValueError(f"{name}: packed {len(packed)} != N*K/4 ({N*K//4})")
            entries.append(WeightEntry(name, N, K, mod.scale_w, packed))

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


def _read_exact(f: BinaryIO, n: int) -> bytes:
    """读恰好 n 字节, 不足抛 ValueError (友好, 非 struct.error / 静默截断)。"""
    b = f.read(n)
    if len(b) != n:
        raise ValueError(f"unexpected EOF: read {len(b)} bytes, expected {n}")
    return b


# 损坏 blob 防御上限 (防巨值触发巨量内存分配 / 超长循环)
_MAX_ENTRIES = 100_000
_MAX_NAME_LEN = 1024
_MAX_PACKED = 1 << 28  # 256 MiB


def read_weight_blob(path: str) -> list[WeightEntry]:
    """读回 blob, 返回 WeightEntry 列表 (验证 / CIM 后端参考)。

    带边界校验: 读取不足 / 长度超上限 / N,K 异常 / packed 长度与 N,K 不一致
    均抛 ValueError, 损坏 blob 早报错 (而非 struct.error 或静默截断/巨量分配)。
    """
    entries: list[WeightEntry] = []
    with open(path, "rb") as f:
        magic = _read_exact(f, 4)
        if magic != MAGIC:
            raise ValueError(f"bad magic {magic!r}, expected {MAGIC!r}")
        version, n = struct.unpack("<II", _read_exact(f, 8))
        if version != VERSION:
            raise ValueError(f"unsupported blob version {version}, expected {VERSION}")
        if n > _MAX_ENTRIES:
            raise ValueError(f"bad num_entries {n} > {_MAX_ENTRIES}")
        for _ in range(n):
            (nl,) = struct.unpack("<I", _read_exact(f, 4))
            if nl == 0 or nl > _MAX_NAME_LEN:
                raise ValueError(f"bad name length {nl} (expected 1..{_MAX_NAME_LEN})")
            name = _read_exact(f, nl).decode("utf-8")
            N, K = struct.unpack("<II", _read_exact(f, 8))
            if N == 0 or K == 0 or K % 4 != 0:
                raise ValueError(f"{name}: bad shape N={N} K={K} (K must be positive multiple of 4)")
            (scale_w,) = struct.unpack("<d", _read_exact(f, 8))
            (bl,) = struct.unpack("<I", _read_exact(f, 4))
            if bl > _MAX_PACKED:
                raise ValueError(f"{name}: packed bytes {bl} > {_MAX_PACKED}")
            packed = _read_exact(f, bl)
            if bl != N * K // 4:
                raise ValueError(f"{name}: packed bytes {bl} != N*K/4 ({N*K//4})")
            entries.append(WeightEntry(name, N, K, scale_w, packed))
    return entries
