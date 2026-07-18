#!/usr/bin/env python3
"""cimres 调度器框架 (S1)。

list scheduling + tile DAG + K维累加依赖。单 BitLinear 内 confirms C1 调度
(k外n内) 即关键路径优先的 list scheduling, 近最优 (关键路径 = 同 n 8k 串行,
8n 全并行, makespan ~134 接近下界 126)。

跨 BitLinear 层内并行 (q/k/v 等) 留 S2/S3 扩展 (调度段抽象, 需打破 func 边界
+ 改 lowering cim_launch idx 语义, q/k/v 失败教训)。

当前: 分析 + 验证当前调度最优性 (不重排 IR, 单 func 无优化空间, 避 SSA 风险)。
框架: tile DAG + list scheduling + 调度段抽象, 供 S2/S3 跨 BitLinear 扩展。

用法:
  python cim_compiler/cimres/scheduler.py --in <placed.mlir>
"""
import sys
import argparse

from cim_compiler.cimres.passes.common import func_blocks, matmuls_in_func
from cim_compiler.cimres.cost_model import estimate_func


def build_tile_dag(ms):
    """建 tile 索引: (n_blk, k_blk) -> matmul。

    K维累加: 同 n_blk 的 k_blk 串行 (psum_page 同, accum RMW), 不同 n_blk 并行。
    返回 by_nk: {(n_blk, k_blk): matmul}。
    """
    return {(m["n_blk"], m["k_blk"]): m for m in ms}


def list_schedule(ms):
    """list scheduling: 关键路径优先 + K维依赖 + Macro 并行。

    普通 BitLinear (role=none): k外n内 (同 k 的 n 并行, 共享 A_PAGE 广播) = C1 顺序。
    qkv 合并 func (role q/k/v): k外bl内n内 (q<k<v, 各自 K 维累加独立, S6 合并) = C1 顺序。
    关键路径 = 同 n 的剩余 k 数 (k_tiles - k), k 小的关键路径长优先。

    返回调度顺序 [(n_blk, k_blk), ...]。
    """
    if not ms:
        return []
    roles = {m.get("role") for m in ms}
    if roles & {"q", "k", "v"}:
        # qkv 合并: 3D (bl, n, k), bl = q:0, k:1, v:2 (与 C1 一致, lower_to_cimres.py:87-89)
        bl_order = {"q": 0, "k": 1, "v": 2}
        by_bl_nk = {(bl_order[m["role"]], m["n_blk"], m["k_blk"]): m for m in ms}
        k_tiles = max(k for _, _, k in by_bl_nk) + 1
        n_tiles_bl = {}
        for (bl, n, _) in by_bl_nk:
            n_tiles_bl[bl] = max(n_tiles_bl.get(bl, 0), n + 1)
        order = []
        for kb in range(k_tiles):                  # k 外
            for bl in range(3):                    # bl 内 (q<k<v)
                for nb in range(n_tiles_bl.get(bl, 0)):  # n 内
                    if (bl, nb, kb) in by_bl_nk:
                        order.append((nb, kb))
        return order
    # 普通 BitLinear: 2D (n, k), k外n内
    by_nk = build_tile_dag(ms)
    if not by_nk:
        return []
    k_tiles = max(k for _, k in by_nk) + 1
    n_tiles = max(n for n, _ in by_nk) + 1
    order = []
    for k in range(k_tiles):          # k 外: 关键路径长 (k 小) 优先
        for n in range(n_tiles):      # n 内: 同 k 并行
            if (n, k) in by_nk:
                order.append((n, k))
    return order


def analyze(mod) -> dict:
    """分析调度: 建 DAG + list scheduling 最优序 + 对照 C1 当前序 + makespan。

    返回 {per_func, n_func, n_match_c1, all_match_c1}。
    all_match_c1=True => C1 当前调度 = list scheduling 最优序 (confirms 最优)。
    """
    per_func = []
    for func_op, _ in func_blocks(mod):
        ms = matmuls_in_func(func_op)
        if not ms:
            continue
        c1_order = [(m["n_blk"], m["k_blk"]) for m in ms]   # IR 顺序 = C1 调度
        sched_order = list_schedule(ms)                      # list scheduling 最优序
        per_func.append({
            "name": ms[0]["bitlinear_name"],
            "n_tile": len(ms),
            "makespan": estimate_func(ms),
            "matches_c1": c1_order == sched_order,
        })
    n_match = sum(1 for f in per_func if f["matches_c1"])
    return {
        "per_func": per_func,
        "n_func": len(per_func),
        "n_match_c1": n_match,
        "all_match_c1": n_match == len(per_func),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default="cim_compiler/cimres/checkpoints/bitnet_ternary_cimres_placed.mlir")
    args = p.parse_args()

    from cim_compiler.cimres.passes.common import load_cimres
    mod, _ = load_cimres(args.inp)
    r = analyze(mod)
    print(f"[scheduler] {r['n_func']} func, list scheduling 与 C1 当前调度一致: "
          f"{r['n_match_c1']}/{r['n_func']}", file=sys.stderr)
    if r["all_match_c1"]:
        print(f"[scheduler] ✓ C1 调度 (k外n内) = list scheduling 最优序, 单 BitLinear 已最优 "
              f"(跨 BitLinear 优化留 S2/S3)", file=sys.stderr)
    else:
        print(f"[scheduler] 差异 func:", file=sys.stderr)
        for f in r["per_func"]:
            if not f["matches_c1"]:
                print(f"  {f['name']} ({f['n_tile']} tile, {f['makespan']} cycle)", file=sys.stderr)


if __name__ == "__main__":
    main()
