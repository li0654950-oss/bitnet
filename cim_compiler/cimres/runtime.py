#!/usr/bin/env python3
"""C5: 运行时动态 M 驱动。

单 token 指令流 (C3 forward.bin, M=1) + 运行时循环吸收动态 seq_len:
  for m in range(M):                 # 动态 seq_len 在这里, 不在指令流
    写 x_int8[m] -> A_PAGE           # PAGE 串行复用 (覆盖写)
    发射单 token 指令流 (该 BitLinear MATMUL 序列) + 门铃 + 等 IRQ
    读 PSUM_PAGE acc[m] -> rescale
  A_PAGE/PSUM_PAGE 串行复用 (M 个 token 复用同一组 PAGE, 首个 k_blk ACCUM=0 清旧值)

验证: 对每 BitLinear, M 个 token 循环执行 (每 token 不同 x_int8), acc[m] 对齐参考
      (x_int8[m] @ W.T)。证明动态 M 由运行时循环支持, 指令流静态 M=1 (不改导出, 方案 C)。

用法:
  nanogpt-gpu python cim_compiler/cimres/runtime.py
  nanogpt-gpu python cim_compiler/cimres/runtime.py --M 10
"""
import os
import sys
import math
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CIM_COMPILER = os.path.dirname(HERE)
REPO = os.path.dirname(CIM_COMPILER)
EXPORT_DIR = os.path.join(CIM_COMPILER, "export")
for _p in (REPO, EXPORT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from torch_mlir import ir
from cim_compiler.cimres.dialect import register_cimres, attr_i32, attr_bool, attr_str
from cim_compiler.cimres.place import A_PAGE_BASE, PSUM_PAGE_BASE
from cim_compiler.cimres.hw_simulator import unpack_2bit_np, _norm, TILE, SHARED_SIZE
from cim_compiler.export.weight_blob import read_weight_blob


def run_dynamic_m(placed_path, weights_path, M=5, seed=0):
    weights = read_weight_blob(weights_path)
    wmap = {_norm(w.name): w for w in weights}
    ctx = ir.Context()
    ctx.load_all_available_dialects()
    register_cimres(ctx)
    with ctx:
        mod = ir.Module.parse(open(placed_path).read(), ctx)

    shared = np.zeros(SHARED_SIZE, dtype=np.uint8)   # 1MB 共享缓存 (M token 串行复用)
    shared32 = shared.view(np.int32)
    macro = {}                                        # dest_id -> [64,64] int8 (预载驻留, M 复用)
    rng = np.random.default_rng(seed)
    max_diff = 0
    n_func = 0

    for op in list(mod.body):
        if op.operation.name != "func.func":
            continue
        n_func += 1
        blk = op.regions[0].blocks[0]
        mats = [i for i in list(blk.operations) if i.operation.name == "cimres.macro_matmul"]
        name = attr_str(mats[0], "bitlinear_name")
        we = wmap[name]
        N, K = we.N, we.K
        n_tiles = math.ceil(N / TILE)
        k_tiles = math.ceil(K / TILE)
        packed = np.frombuffer(we.packed, dtype=np.uint8).reshape(N, K // 4)
        w_ternary = unpack_2bit_np(packed)
        Np, Kp = n_tiles * TILE, k_tiles * TILE
        W = np.zeros((Np, Kp), dtype=np.int32)
        W[:N, :K] = w_ternary.astype(np.int32)

        # 预载 Macro (一次, M token 复用, 权重驻留)
        for m_op in mats:
            d = attr_i32(m_op, "dest_id")
            nb = attr_i32(m_op, "n_blk")
            kb = attr_i32(m_op, "k_blk")
            macro[d] = W[nb * TILE:(nb + 1) * TILE, kb * TILE:(kb + 1) * TILE].astype(np.int8)

        # ---- 动态 M 循环: 单 token 指令流重复 M 次, PAGE 串行复用 ----
        xpad = np.zeros(Kp, dtype=np.int8)
        last_diff = 0
        for m in range(M):
            x_int8 = rng.integers(-128, 127, size=K, dtype=np.int8)   # 每 token 不同激活
            xpad[:K] = x_int8
            # 写 A_PAGE (CPU 量化后写入, 覆盖)
            for kb in range(k_tiles):
                a_page = A_PAGE_BASE + kb
                shared[a_page * 256:a_page * 256 + 64] = xpad[kb * TILE:(kb + 1) * TILE].astype(np.uint8)
            # 发射单 token 指令流 (该 BitLinear MATMUL 序列, 首个 k_blk ACCUM=0 清旧 acc)
            for m_op in mats:
                d = attr_i32(m_op, "dest_id")
                a = attr_i32(m_op, "a_page")
                p = attr_i32(m_op, "psum_page")
                accum = attr_bool(m_op, "accum")
                x = shared[a * 256:a * 256 + 64].astype(np.int8).astype(np.int32)
                y = macro[d].astype(np.int32) @ x
                poff = p * 64
                if not accum:
                    shared32[poff:poff + 64] = y
                else:
                    shared32[poff:poff + 64] = shared32[poff:poff + 64] + y
            # 读 PSUM_PAGE acc[m]
            acc_sim = np.zeros(N, dtype=np.int32)
            for nb in range(n_tiles):
                p = PSUM_PAGE_BASE + nb
                vec = shared32[p * 64:p * 64 + 64]
                s = nb * TILE
                e = min(s + TILE, N)
                acc_sim[s:e] = vec[:e - s]
            acc_ref = (x_int8.astype(np.int32) @ w_ternary.astype(np.int32).T).astype(np.int32)
            diff = int(np.max(np.abs(acc_sim.astype(np.int64) - acc_ref.astype(np.int64))))
            max_diff = max(max_diff, diff)
            last_diff = diff
        if n_func <= 3 or last_diff != 0:
            print(f"  [{n_func:2d}] {name} N={N} K={K}: M={M} token 循环, "
                  f"末 token diff={last_diff} {'OK' if last_diff == 0 else 'FAIL'}", file=sys.stderr)
    print(f"[C5] {n_func} BitLinear × M={M} token 动态循环, max_diff={max_diff} "
          f"{'PASS ✓' if max_diff == 0 else 'FAIL ✗'}", file=sys.stderr)
    return max_diff


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--placed", default="cim_compiler/cimres/checkpoints/bitnet_ternary_cimres_placed.mlir")
    p.add_argument("--weights", default="checkpoints/bitnet_ternary_weights.bin")
    p.add_argument("--M", type=int, default=5, help="动态 token 数 (模拟 seq_len)")
    args = p.parse_args()
    sys.exit(0 if run_dynamic_m(args.placed, args.weights, args.M) == 0 else 1)
