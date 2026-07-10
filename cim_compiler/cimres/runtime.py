#!/usr/bin/env python3
"""C5: 运行时动态 M 驱动 (经 MMIO 驱动纯硬件, 模拟 cim_stub.c 的 M 循环)。

单 token 指令流 (forward.bin, M=1) + 运行时循环吸收动态 seq_len:
  for m in range(M):                 # 动态 seq_len 在这里, 不在指令流
    driver_launch: 写 A_PAGE + 门铃 + poll IRQ + 读 PSUM_PAGE acc[m]
  A_PAGE/PSUM_PAGE 串行复用 (首 k_blk ACCUM=0 清旧 acc, 固化在指令里)

验证: 对每 BitLinear, M 个 token 循环 (driver_launch 经 MMIO 驱动纯硬件),
      acc[m] 对齐参考 (x_int8[m] @ W.T)。证明动态 M 由运行时循环支持,
      指令流静态 M=1 (方案 C, 不改导出)。

用法:
  nanogpt-gpu python cim_compiler/cimres/runtime.py
  nanogpt-gpu python cim_compiler/cimres/runtime.py --M 10
"""
import os
import sys
import json
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CIM_COMPILER = os.path.dirname(HERE)
REPO = os.path.dirname(CIM_COMPILER)
EXPORT_DIR = os.path.join(CIM_COMPILER, "export")
for _p in (REPO, EXPORT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cim_compiler.cimres.hw_simulator import (
    HwCimSimulator, driver_preload, driver_launch, unpack_2bit_np, _norm,
)
from cim_compiler.export.weight_blob import read_weight_blob


def run_dynamic_m(weights_path, M=5, seed=0):
    partition = os.path.join(REPO, "checkpoints/bitnet_ternary_partition.json")
    preload = os.path.join(REPO, "cim_compiler/cimres/checkpoints/preload.bin")
    forward = os.path.join(REPO, "cim_compiler/cimres/checkpoints/forward.bin")

    sim = HwCimSimulator()
    driver_preload(sim, preload)                       # Preload: MMIO 驱动 (一次性)
    print(f"[C5] preload (MMIO driver): {len(sim.macros.macro)} Macro 预载", file=sys.stderr)

    part = json.load(open(partition))
    idx2name = {blk["idx"]: blk["bitlinear_name"] for blk in part["cim_blocks"]}
    weights = read_weight_blob(weights_path)
    wmap = {_norm(w.name): w for w in weights}
    w_ternary = {n: unpack_2bit_np(np.frombuffer(wmap[n].packed, dtype=np.uint8).reshape(wmap[n].N, wmap[n].K // 4))
                 for n in idx2name.values()}

    rng = np.random.default_rng(seed)
    max_diff = 0
    n_func = 0
    for idx in sorted(idx2name):
        n_func += 1
        name = idx2name[idx]
        we = wmap[name]
        N, K = we.N, we.K
        x = rng.integers(-128, 127, size=(M, K), dtype=np.int8)   # 每 token 不同激活
        acc_sim = np.zeros((M, N), dtype=np.int32)
        for m in range(M):                              # 动态 M 循环 (单 token 指令流重复)
            acc_sim[m], _ = driver_launch(sim, forward, idx, x[m], N, K)
        acc_ref = (x.astype(np.int32) @ w_ternary[name].astype(np.int32).T).astype(np.int32)
        diff = int(np.max(np.abs(acc_sim.astype(np.int64) - acc_ref.astype(np.int64))))
        max_diff = max(max_diff, diff)
        if n_func <= 3 or diff != 0:
            print(f"  [{n_func:2d}] {name} N={N} K={K}: M={M} 循环, diff={diff} "
                  f"{'OK' if diff == 0 else 'FAIL'}", file=sys.stderr)
    print(f"[C5] {n_func} BitLinear × M={M} 动态循环, max_diff={max_diff} "
          f"{'PASS ✓' if max_diff == 0 else 'FAIL ✗'}", file=sys.stderr)
    return max_diff


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="checkpoints/bitnet_ternary_weights.bin")
    p.add_argument("--M", type=int, default=5, help="动态 token 数 (模拟 seq_len)")
    args = p.parse_args()
    sys.exit(0 if run_dynamic_m(args.weights, args.M) == 0 else 1)
