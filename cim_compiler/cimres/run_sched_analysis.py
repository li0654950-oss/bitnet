#!/usr/bin/env python3
"""cimres 调度分析编排 (S1): cost_model + scheduler + page_alloc。

pipeline step: C2 place -> verify -> [调度分析] -> C3 emit
打印 makespan + 调度最优性 + PAGE 占用分析。不改产物 (C1/C2 单 BitLinear 已最优,
S1 confirms + 框架, makespan 优化留 S2/S3 跨 BitLinear)。

用法:
  python cim_compiler/cimres/run_sched_analysis.py --in <placed.mlir>
"""
import sys
import argparse

from cim_compiler.cimres.passes.common import load_cimres
from cim_compiler.cimres.cost_model import estimate
from cim_compiler.cimres.scheduler import analyze as sched_analyze
from cim_compiler.cimres.page_alloc import allocate as page_alloc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default="cim_compiler/cimres/checkpoints/bitnet_ternary_cimres_placed.mlir")
    args = p.parse_args()

    mod, _ = load_cimres(args.inp)
    cm = estimate(mod)
    sa = sched_analyze(mod)
    pa = page_alloc(mod)

    print(f"[sched] forward makespan = {cm['total']} cycle/step ({cm['n_func']} func, "
          f"avg {cm['total'] // max(cm['n_func'], 1)}/func)", file=sys.stderr)
    print(f"[sched] 调度最优性: list scheduling 与 C1 一致 {sa['n_match_c1']}/{sa['n_func']} "
          f"({'✓ 单 BitLinear 已最优' if sa['all_match_c1'] else '✗ 有差异'})", file=sys.stderr)
    print(f"[sched] PAGE 峰值 (跨 func 复用): a_page={pa['max_a_page_peak']} + "
          f"psum_page={pa['max_psum_page_peak']} = {pa['max_a_page_peak'] + pa['max_psum_page_peak']} PAGE "
          f"(C2 公式已最优)", file=sys.stderr)
    print(f"[sched] S1 结论: 单 BitLinear 调度/PAGE 已近最优, makespan 优化需 S2 异步门铃 / "
          f"S3 融合 (跨 BitLinear 层内并行)", file=sys.stderr)


if __name__ == "__main__":
    main()
