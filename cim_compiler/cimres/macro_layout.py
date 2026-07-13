#!/usr/bin/env python3
"""S4: macro_layout - dest_id -> 2D (x,y) 物理坐标映射策略。

CIM 4096 Macro 排 64x64 2D Mesh, 总线入口在原点 (0,0), 路由延迟 =
Manhattan(dest, 原点) * T_ROUT_PER_HOP (hw_simulator/cost_model 已接入)。
macro_layout 优化 dest_id->(x,y) 映射, 把高频复用 Macro 放近原点减路由 cycle。

策略:
  linear: 线性扫描 (dest_id%64, dest_id//64), 等价默认无重映射 (向后兼容)
  hotspot: 按 dest_id 复用频次降序分配近原点坐标 (频次高 -> 小 x+y, 减路由)
  bitlinear_cluster: 同 BitLinear 的 tile 物理相邻聚簇 (减跨 BitLinear 路由)

输出: hw_config.LAYOUT_MAP = {dest_id: (x,y)}, 供 hw_simulator/cost_model 算路由延迟。
默认 T_ROUT_PER_HOP=0 时布局不影响 cycle (兼容); autotuner/macro_layout 设 >0 启用。

用法:
  python cim_compiler/cimres/macro_layout.py --strategy hotspot --rout 1
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

from cim_compiler.cimres.hw_config import MESH_DIM
from cim_compiler.cimres.passes.common import func_blocks, matmuls_in_func


def _near_origin_order(n):
    """生成 n 个 (x,y), 按 x+y 升序 (近原点优先), 同 x+y 按 x 升序。"""
    pts = [(x, y) for x in range(MESH_DIM) for y in range(MESH_DIM)]
    pts.sort(key=lambda p: (p[0] + p[1], p[0]))
    return pts[:n]


def collect_dests(mod):
    """从 placed IR 收集所有 dest_id + 复用频次 + bitlinear 归属。

    返回 (dests, freq, bl): dests=[(dest_id,name),...] 按出现序,
    freq={dest_id:matmul次数}, bl={dest_id:bitlinear_name}。
    """
    dests, freq, bl = [], {}, {}
    for func_op, _ in func_blocks(mod):
        for m in matmuls_in_func(func_op):
            d = m["dest_id"]; name = m["bitlinear_name"]
            freq[d] = freq.get(d, 0) + 1
            bl[d] = name
            dests.append((d, name))
    return dests, freq, bl


def layout_linear(mod):
    """线性扫描: dest_id -> (dest_id%64, dest_id//64)。等价默认无重映射。"""
    dests, _, _ = collect_dests(mod)
    return {d: (d % MESH_DIM, d // MESH_DIM) for d, _ in dests}


def layout_hotspot(mod):
    """热点优先: 按 dest_id 复用频次降序分配近原点坐标 (频次高 -> 小 x+y)。"""
    dests, freq, _ = collect_dests(mod)
    unique = sorted({d for d, _ in dests}, key=lambda d: -freq[d])  # 频次降序
    pts = _near_origin_order(len(unique))
    return {d: pts[i] for i, d in enumerate(unique)}


def layout_bitlinear_cluster(mod):
    """BitLinear 聚簇: 同 BitLinear 的 tile 物理相邻 (连续 (x,y) 块)。"""
    dests, _, _ = collect_dests(mod)
    groups = {}
    for d, name in dests:
        groups.setdefault(name, []).append(d)
    layout, idx = {}, 0
    for name in groups:                       # 按 BitLinear 分组, 组内按 dest_id
        for d in sorted(set(groups[name])):
            layout[d] = (idx % MESH_DIM, idx // MESH_DIM)
            idx += 1
    return layout


STRATEGIES = {
    "linear": layout_linear,
    "hotspot": layout_hotspot,
    "bitlinear_cluster": layout_bitlinear_cluster,
}


def apply(mod, strategy="linear"):
    """计算 layout_map 并设置到 hw_config.LAYOUT_MAP。返回 layout_map。"""
    from cim_compiler.cimres import hw_config
    layout_map = STRATEGIES[strategy](mod)
    hw_config.LAYOUT_MAP = layout_map
    return layout_map


def layout_stats(layout_map):
    """统计布局路由特征: 平均/最大 hops。"""
    from cim_compiler.cimres.hw_simulator import dest_origin_hops
    hops = [dest_origin_hops(d) for d in layout_map]
    return {"n_macro": len(layout_map), "avg_hops": sum(hops) / max(len(hops), 1),
            "max_hops": max(hops) if hops else 0}


def main():
    from cim_compiler.cimres.passes.common import load_cimres
    from cim_compiler.cimres import cost_model, hw_config
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default="cim_compiler/cimres/checkpoints/bitnet_ternary_cimres_placed.mlir")
    p.add_argument("--strategy", default="linear", choices=list(STRATEGIES))
    p.add_argument("--rout", type=int, default=1, help="T_ROUT_PER_HOP (0=关闭布局感知, 默认1)")
    args = p.parse_args()

    mod, _ = load_cimres(args.inp)
    layout_map = apply(mod, args.strategy)
    hw_config.T_ROUT_PER_HOP = args.rout            # 启用布局感知 (hw_config 统一, sim+cost_model 都生效)
    r = cost_model.estimate(mod)
    st = layout_stats(layout_map)
    print(f"[macro_layout] strategy={args.strategy} rout={args.rout}", file=sys.stderr)
    print(f"  Macro={st['n_macro']} avg_hops={st['avg_hops']:.1f} max_hops={st['max_hops']} "
          f"makespan={r['total']} cycle", file=sys.stderr)
    return r["total"]


if __name__ == "__main__":
    main()
