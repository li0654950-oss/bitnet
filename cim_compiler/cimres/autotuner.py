#!/usr/bin/env python3
"""S4: autotuner - 枚举布局/调度策略, cost_model 评估选 min makespan。

搜索空间:
  - layout_strategy: linear / hotspot / bitlinear_cluster (macro_layout)
  - T_ROUT_PER_HOP: 路由延迟系数 (布局感知开关, 默认 1)
  - 调度策略暂硬编码 k外n内 (scheduler 未参数化, 后续 S4+ 扩展 n外k内等)

评估: cost_model.estimate (轻量 busy_until 时序, 不跑数值仿真)。
选 min makespan, 设置最优 layout 到 hw_config.LAYOUT_MAP。

用法:
  python cim_compiler/cimres/autotuner.py --rout 1
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
CIM_COMPILER = os.path.dirname(HERE)
REPO = os.path.dirname(CIM_COMPILER)
EXPORT_DIR = os.path.join(CIM_COMPILER, "export")
for _p in (REPO, EXPORT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cim_compiler.cimres import macro_layout, cost_model, hw_config


def search(mod, rout=1):
    """枚举 layout 策略, cost_model 评估。返回 [(strategy, makespan, stats)] 升序。"""
    results = []
    for strategy in macro_layout.STRATEGIES:
        layout_map = macro_layout.apply(mod, strategy)   # 设 hw_config.LAYOUT_MAP
        hw_config.T_ROUT_PER_HOP = rout                   # 启用布局感知 (hw_config 统一, sim+cost_model 都生效)
        r = cost_model.estimate(mod)
        st = macro_layout.layout_stats(layout_map)
        results.append({"strategy": strategy, "makespan": r["total"], "stats": st})
    results.sort(key=lambda x: x["makespan"])
    return results


def tune(mod, rout=1, apply_best=True, out_json=None):
    """搜索 + 选最优 + (可选) 设置最优 layout + 序列化 layout_config.json。返回 (best, all_results)。"""
    results = search(mod, rout)
    best = results[0]
    if apply_best:
        layout_map = macro_layout.apply(mod, best["strategy"])   # 设置最优到 hw_config.LAYOUT_MAP
        if out_json:                                              # 序列化供 cim_sim_server 跨进程加载
            import json
            cfg = {"strategy": best["strategy"], "t_rout_per_hop": rout,
                   "layout_map": {int(k): list(v) for k, v in layout_map.items()}}
            with open(out_json, "w") as f:
                json.dump(cfg, f)
    return best, results


def main():
    from cim_compiler.cimres.passes.common import load_cimres
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default="cim_compiler/cimres/checkpoints/bitnet_ternary_cimres_placed.mlir")
    p.add_argument("--rout", type=int, default=1, help="T_ROUT_PER_HOP (0=关闭布局感知)")
    args = p.parse_args()

    mod, _ = load_cimres(args.inp)
    out_json = os.path.join(HERE, "checkpoints", "layout_config.json")
    best, results = tune(mod, args.rout, out_json=out_json)
    baseline = next((r for r in results if r["strategy"] == "linear"), results[0])
    speedup = 1 - best["makespan"] / max(baseline["makespan"], 1)

    print(f"[autotuner] rout={args.rout}, 搜索 {len(results)} layout 策略:", file=sys.stderr)
    for r in results:
        mark = " <- 最优" if r["strategy"] == best["strategy"] else ""
        print(f"  {r['strategy']:20s} makespan={r['makespan']:6d} cycle  "
              f"avg_hops={r['stats']['avg_hops']:.1f}{mark}", file=sys.stderr)
    print(f"[autotuner] 最优: {best['strategy']} makespan={best['makespan']} "
          f"(vs linear {baseline['makespan']}, Mesh 模型下减损 {speedup:.1%})", file=sys.stderr)
    print(f"[autotuner] 注: T_ROUT={args.rout} 是 Mesh NoC 研究假设 (cim_mlp.md §2.2 spec 为广播总线, "
          f"T_ROUT=0 基线 makespan=6268 不受 layout 影响); 此为 layout 优化研究, 非实际硬件提升",
          file=sys.stderr)
    return best


if __name__ == "__main__":
    main()
