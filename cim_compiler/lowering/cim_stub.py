#!/usr/bin/env python3
"""L5: @cim_launch CPU 仿真 stub (ctypes, 匹配 LLVM memref calling convention)。

@cim_launch_<idx> 的 LLVM calling convention (L4 产出, finalize-memref-to-llvm 展开):
  (X memref 2D, W memref 2D) -> result memref 2D
  每个 2D memref 展开 7 参数: (allocated_ptr, aligned_ptr, offset, size0, size1, stride0, stride1)

  X: si8 [M, K]            per-token int8 激活
  W: ui8 [N, K//4]         2bit 补码打包三值权重
  -> result: si32 [M, N]   累加输出

stub 语义 (同 cim_op.cim_matmul): unpack W (ternary) + float matmul -> int32。
float32 精确 (|acc|<=65536<2^24), 数值等价 _int_mm int32。

L6 JIT 用 ExecutionEngine.register_runtime 注册本 stub 到 37 个 @cim_launch_<idx>
(逻辑相同, sizes 从 memref descriptor 读, 故一个通用 stub 注册多次)。
"""
import ctypes
import numpy as np


class Memref2D(ctypes.Structure):
    """LLVM 2D memref descriptor: 7 连续字段 (等价 struct<ptr,ptr,i64,array<2xi64>,array<2xi64>>)。
    ctypes 不支持返回含 array 的 Structure, 故 sizes/strides 展开成单字段 (layout 相同)。"""
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("size0", ctypes.c_int64), ("size1", ctypes.c_int64),
        ("stride0", ctypes.c_int64), ("stride1", ctypes.c_int64),
    ]


# 保持 result memref buffer 不被 GC (去 buffer-dealloc, leak OK; 测试场景)
_KEEP = []


def unpack_2bit_np(packed: np.ndarray) -> np.ndarray:
    """uint8[..., K//4] (2bit 补码) -> int8[..., K] {-1,0,1}。同 cim_op.unpack_2bit_aten。"""
    p = packed.astype(np.int32)
    c0 = p % 4
    c1 = (p // 4) % 4
    c2 = (p // 16) % 4
    c3 = (p // 64) % 4
    code = np.stack([c0, c1, c2, c3], axis=-1).reshape(*packed.shape[:-1], -1)
    return np.where(code >= 2, code - 4, code).astype(np.int8)


def cim_launch(alloc_a, align_a, off_a, sa0, sa1, st0, st1,
               alloc_b, align_b, off_b, sb0, sb1, stb0, stb1):
    """@cim_launch stub: X si8[M,K] x W ui8[N,K//4] -> result si32[M,N]。

    args (LLVM memref 展开, 每 2D memref 7 参数):
      X: alloc_a, align_a, off_a, sa0=M, sa1=K, st0, st1
      W: alloc_b, align_b, off_b, sb0=N, sb1=K//4, stb0, stb1
    """
    M, K = int(sa0), int(sa1)
    N, K4 = int(sb0), int(sb1)
    # X: si8 [M, K] (contiguous, stride 行主)
    x = np.ctypeslib.as_array(
        (ctypes.c_int8 * (M * K)).from_address(int(align_a))).reshape(M, K)
    # W: ui8 [N, K4] (MLIR i8, 按 uint8 解释 packed ternary)
    w = np.ctypeslib.as_array(
        (ctypes.c_int8 * (N * K4)).from_address(int(align_b))).view(np.uint8).reshape(N, K4)
    # unpack W -> [N, K] {-1,0,1}, float matmul -> int32 [M, N]
    w_int8 = unpack_2bit_np(w)
    acc = (x.astype(np.float32) @ w_int8.astype(np.float32).T).astype(np.int32)
    # alloc result memref (si32 [M, N])
    buf = (ctypes.c_int32 * (M * N))()
    np.ctypeslib.as_array(buf).reshape(M, N)[:] = acc
    _KEEP.append(buf)
    ptr = ctypes.cast(buf, ctypes.c_void_p).value
    return Memref2D(ptr, ptr, 0, M, N, N, 1)


# ctypes CFUNCTYPE: 14 参数 (2x 2D memref) -> Memref2D
STUB_T = ctypes.CFUNCTYPE(
    Memref2D,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
)


def make_stub():
    """创建一个 stub callback (CFUNCTYPE, 引用计数保活)。每个 @cim_launch_<idx> 一个。"""
    return STUB_T(cim_launch)


def register_all(ee, names):
    """把 stub 注册到 ExecutionEngine 的所有 @cim_launch_<idx> symbol。"""
    cbs = []
    for nm in names:
        cb = make_stub()
        ee.register_runtime(nm, cb)
        cbs.append(cb)  # 保活 (CFUNCTYPE 要 outlive EE)
    return cbs
