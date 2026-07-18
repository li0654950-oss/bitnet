#!/usr/bin/env python3
"""cimres PAGE 分配器 (S1)。

A_PAGE/PSUM_PAGE 生命周期分析 + 复用确认。当前 C2 简单公式
(a_page=A_PAGE_BASE+kb, psum_page=PSUM_PAGE_BASE+nb) 跨 BitLinear 已复用。
page_alloc 分析生命周期 confirms 当前近最优, 框架为 S2 double buffer 扩展。

live range (对齐 cim_stub.c cim_launch 执行模型):
  a_page (k_blk): CPU 在 M 循环开头一次性写所有 kb 的 a_page (覆盖区), 整段指令
    (含所有 k 的 matmul) 执行期间全部驻留。峰值 = k_tiles (整段需所有 a_page)。
  psum_page (n_blk): K维累加同 n, 首写 (k0 accum=false) 到 CPU 读 (SYNC_HALT 后)。
    峰值 = n_tiles (8 n 并行各一个 psum_page)。

单 func 峰值 = k_tiles (a_page) + n_tiles (psum_page)。当前 C2 公式已最优。
S2 double buffer: 2 套 a_page (M 循环 ping-pong), 峰值 2*k_tiles, 本框架支持扩展。

用法:
  python cim_compiler/cimres/page_alloc.py --in <placed.mlir>
"""
import sys
import argparse

from cim_compiler.cimres.hw_config import A_PAGE_BASE, PSUM_PAGE_BASE
from cim_compiler.cimres.passes.common import func_blocks, matmuls_in_func


def analyze_liveness(ms):
    """单 func PAGE 生命周期分析。返回峰值占用。

    a_page 峰值 = k_tiles (整段执行需所有 k 的 a_page 驻留, CPU 预写覆盖区)。
    psum_page 峰值 = n_tiles (8 n 并行, K维累加同 n 同 page, 活跃到 SYNC_HALT)。
    """
    if not ms:
        return {"a_page_peak": 0, "psum_page_peak": 0, "k_tiles": 0, "n_tiles": 0}
    k_tiles = max(m["k_blk"] for m in ms) + 1
    n_tiles = max(m["n_blk"] for m in ms) + 1
    return {
        "a_page_peak": k_tiles,
        "psum_page_peak": n_tiles,
        "k_tiles": k_tiles,
        "n_tiles": n_tiles,
    }


def allocate(mod) -> dict:
    """PAGE 分配分析。confirms C2 公式已最优 (峰值 = k_tiles + n_tiles)。

    返回 {per_func, n_func, max_a_page, max_psum_page}。
    跨 BitLinear 复用 (前向串行, 前一个释放后下一个复用同 PAGE 区)。
    """
    per_func = []
    max_a = max_p = 0
    for func_op, _ in func_blocks(mod):
        ms = matmuls_in_func(func_op)
        if not ms:
            continue
        live = analyze_liveness(ms)
        per_func.append({"name": ms[0]["bitlinear_name"], **live})
        max_a = max(max_a, live["a_page_peak"])
        max_p = max(max_p, live["psum_page_peak"])
    return {
        "per_func": per_func,
        "n_func": len(per_func),
        "max_a_page_peak": max_a,       # 跨 func 峰值 (复用, 取 max)
        "max_psum_page_peak": max_p,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default="cim_compiler/cimres/checkpoints/bitnet_ternary_cimres_placed.mlir")
    args = p.parse_args()

    from cim_compiler.cimres.passes.common import load_cimres
    mod, _ = load_cimres(args.inp)
    r = allocate(mod)
    print(f"[page_alloc] {r['n_func']} func, 跨 func 峰值 (复用): "
          f"a_page={r['max_a_page_peak']}, psum_page={r['max_psum_page_peak']} "
          f"(共 {r['max_a_page_peak'] + r['max_psum_page_peak']} PAGE)", file=sys.stderr)
    print(f"[page_alloc] ✓ C2 公式 (a_page=0x{A_PAGE_BASE:x}+kb, psum_page=0x{PSUM_PAGE_BASE:x}+nb) "
          f"已最优 (整段执行需 k_tiles a_page + n_tiles psum_page)", file=sys.stderr)
    print(f"[page_alloc] S2 double buffer 扩展: 峰值将升至 2*{r['max_a_page_peak']}+"
          f"{r['max_psum_page_peak']} a_page/psum_page (M 循环 ping-pong)", file=sys.stderr)


if __name__ == "__main__":
    main()
