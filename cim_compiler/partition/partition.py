#!/usr/bin/env python3
"""FX graph CPU/CIM 划分 — 产出逻辑子图 (节点标注 + 边界张量 + CIM 块)。

不修改 export 图: 遍历标注每节点 backend, 输出 CPU↔CIM 边界张量, 分组为 CPU/CIM 逻辑子图。
为 compiler 后端提供调度依据 (CIM 节点 → Macro, CPU 节点 → CPU, 边界张量 = 共享缓存读写点)。

产物 partition.json:
  summary: {total, cpu, cim, cim_blocks}
  node_backend: {node_name: 'CPU'|'CIM'}
  cim_blocks: [{idx, bitlinear_name, int_mm, w_packed, x_int8_in, acc_out, unpack_nodes}]
  boundaries: {cpu_to_cim: [...], cim_to_cpu: [...]}

用法:
  python cim_compiler/partition/partition.py
  python cim_compiler/partition/partition.py --graph checkpoints/bitnet_ternary.pt2 \\
    --out checkpoints/bitnet_ternary_partition.json
"""
import os
import sys
import json
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch
import torch.export
import torch.fx as fx

from classify import mark_cim_nodes, node_backend, is_cim_matmul


@dataclass
class CimBlock:
    idx: int
    bitlinear_name: str       # 对应 BitLinear 路径 (从 w_packed 名解析)
    int_mm: str               # matmul 节点名
    w_packed: str             # w_packed placeholder 节点名
    x_int8_in: str            # CPU→CIM 边界: 激活 int8 输入节点名
    acc_out: str              # CIM→CPU 边界: int32 输出节点名 (matmul 本身)
    unpack_nodes: list        # 解包链节点名列表


@dataclass
class Boundary:
    node: str                 # 边界张量对应的节点名
    direction: str            # 'cpu_to_cim' | 'cim_to_cpu'
    dtype: str                # 'int8' / 'int32' / ...
    shape: list               # 形状描述 (动态维用符号字符串)


@dataclass
class Partition:
    node_backend: dict        # {node_name: 'CPU'|'CIM'}
    cim_blocks: list          # list[CimBlock]
    cpu_nodes: list           # list[str]
    cim_nodes: list           # list[str]
    boundaries: dict          # {'cpu_to_cim': [Boundary], 'cim_to_cpu': [Boundary]}


def _find_w_packed(node: fx.Node) -> Optional[fx.Node]:
    """从权重侧节点反向追溯到 w_packed placeholder / get_attr。

    处理 stack 等算子的 list/tuple args (节点在 list 内)。
    """
    if node.op in ("placeholder", "get_attr"):
        return node
    for a in node.args:
        if isinstance(a, fx.Node):
            r = _find_w_packed(a)
            if r is not None:
                return r
        elif isinstance(a, (list, tuple)):
            for x in a:
                if isinstance(x, fx.Node):
                    r = _find_w_packed(x)
                    if r is not None:
                        return r
    return None


def _parse_bitlinear_name(w_packed_name: str) -> str:
    """从 w_packed placeholder 名解析 BitLinear 路径。

    export placeholder 名形如 b_layers_0_attn_q_proj_w_packed
    → layers.0.attn.q.proj (数字段是 layer index, 下划线还原为点)。
    """
    name = w_packed_name
    for prefix in ("b_", "p_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    for suffix in ("_w_packed", "_weight"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.replace("_", ".")


def _node_meta(node: fx.Node):
    """取节点 meta 中的 val (tensor), 用于 dtype/shape 描述。"""
    return node.meta.get("val")


def _dtype_desc(node: fx.Node) -> str:
    v = _node_meta(node)
    if v is None:
        return "?"
    try:
        return str(v.dtype).replace("torch.", "")
    except Exception:
        return "?"


def _shape_desc(node: fx.Node) -> list:
    v = _node_meta(node)
    if v is None:
        return []
    try:
        return [str(s) for s in v.shape]
    except Exception:
        return []


def partition_graph(prog) -> Partition:
    """对 export 图做 CPU/CIM 划分, 返回 Partition。不修改图。"""
    graph = prog.graph
    cim_set = mark_cim_nodes(graph)

    all_cf = [n for n in graph.nodes if n.op == "call_function"]
    cpu_nodes = [n.name for n in all_cf if n not in cim_set]
    cim_nodes = [n.name for n in all_cf if n in cim_set]

    # node_backend: call_function + placeholder
    node_backend_map = {}
    for n in graph.nodes:
        if n.op in ("call_function", "placeholder"):
            node_backend_map[n.name] = node_backend(n, cim_set)

    # CIM 块: 每个 matmul 一个
    cim_blocks = []
    cpu_to_cim = []
    cim_to_cpu = []

    int_mms = [n for n in graph.nodes if is_cim_matmul(n)]
    for idx, mm in enumerate(int_mms):
        x_int8 = mm.args[0]                # CPU→CIM 边界 (激活)
        w_t = mm.args[1]                    # w_int8.t()
        w_packed = _find_w_packed(w_t)      # 追溯到 placeholder
        bitlinear_name = _parse_bitlinear_name(w_packed.name) if w_packed else "?"

        # 解包链: 权重侧追溯到 w_packed 的节点 (排除 mm 本身)
        unpack = []
        def collect(node, seen):
            if node is mm or node in seen:
                return
            seen.add(node)
            if node.op == "call_function":
                unpack.append(node.name)
            for a in node.args:
                if isinstance(a, fx.Node):
                    collect(a, seen)
                elif isinstance(a, (list, tuple)):
                    for x in a:
                        if isinstance(x, fx.Node):
                            collect(x, seen)
        collect(w_t, set())

        cim_blocks.append(CimBlock(
            idx=idx,
            bitlinear_name=bitlinear_name,
            int_mm=mm.name,
            w_packed=w_packed.name if w_packed else "?",
            x_int8_in=x_int8.name,
            acc_out=mm.name,
            unpack_nodes=unpack,
        ))
        cpu_to_cim.append(Boundary(
            node=x_int8.name, direction="cpu_to_cim",
            dtype=_dtype_desc(x_int8), shape=_shape_desc(x_int8),
        ))
        cim_to_cpu.append(Boundary(
            node=mm.name, direction="cim_to_cpu",
            dtype=_dtype_desc(mm), shape=_shape_desc(mm),
        ))

    return Partition(
        node_backend=node_backend_map,
        cim_blocks=cim_blocks,
        cpu_nodes=cpu_nodes,
        cim_nodes=cim_nodes,
        boundaries={"cpu_to_cim": cpu_to_cim, "cim_to_cpu": cim_to_cpu},
    )


def to_json(part: Partition, path: str) -> dict:
    """序列化 Partition 为 JSON。返回 summary。"""
    data = {
        "summary": {
            "total_call_function": len(part.cpu_nodes) + len(part.cim_nodes),
            "cpu_nodes": len(part.cpu_nodes),
            "cim_nodes": len(part.cim_nodes),
            "cim_blocks": len(part.cim_blocks),
        },
        "node_backend": part.node_backend,
        "cim_blocks": [asdict(b) for b in part.cim_blocks],
        "boundaries": {
            "cpu_to_cim": [asdict(b) for b in part.boundaries["cpu_to_cim"]],
            "cim_to_cpu": [asdict(b) for b in part.boundaries["cim_to_cpu"]],
        },
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return data["summary"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graph", default="checkpoints/bitnet_ternary.pt2")
    p.add_argument("--out", default="checkpoints/bitnet_ternary_partition.json")
    args = p.parse_args()

    prog = torch.export.load(args.graph)
    part = partition_graph(prog)
    summary = to_json(part, args.out)
    print(f"[partition] CPU={summary['cpu_nodes']} CIM={summary['cim_nodes']} "
          f"CIM块={summary['cim_blocks']} (共 {summary['total_call_function']} call_function)", file=sys.stderr)
    print(f"[partition] 边界: cpu_to_cim={len(part.boundaries['cpu_to_cim'])} "
          f"cim_to_cpu={len(part.boundaries['cim_to_cpu'])}", file=sys.stderr)
    print(f"[partition] saved: {args.out}", file=sys.stderr)
    for b in part.cim_blocks[:5]:
        print(f"  [{b.idx:2d}] {b.bitlinear_name}: unpack={len(b.unpack_nodes)} 节点, "
              f"x_in={b.x_int8_in}, acc={b.acc_out}", file=sys.stderr)
    if len(part.cim_blocks) > 5:
        print(f"  ... (共 {len(part.cim_blocks)} 个 CIM 块)", file=sys.stderr)


if __name__ == "__main__":
    main()
