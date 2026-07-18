#!/usr/bin/env python3
"""FX graph CPU/CIM 划分 - 产出逻辑子图 (节点标注 + 边界张量 + CIM 块)。

不修改 export 图: 遍历标注每节点 backend, 输出 CPU↔CIM 边界张量, 分组为 CPU/CIM 逻辑子图。
为 compiler 后端提供调度依据 (CIM 节点 -> Macro, CPU 节点 -> CPU, 边界张量 = 共享缓存读写点)。

custom op 模式 (cim::matmul): CIM 节点 = cim.matmul op 节点本身 (解包/累加封装在 op 内, 不内联),
故 CIM 块 = 1 个 op 节点 (无解包链, 与内联 _int_mm 模式不同)。

产物 partition.json:
  summary: {total, cpu, cim, cim_blocks}
  node_backend: {node_name: 'CPU'|'CIM'}
  cim_blocks: [{idx, bitlinear_name, int_mm, w_packed, x_int8_in, acc_out}]
  boundaries: {cpu_to_cim: [...], cim_to_cpu: [...]}

用法:
  python cim_compiler/partition/partition.py
  python cim_compiler/partition/partition.py --graph checkpoints/bitnet_ternary.pt2 \\
    --out checkpoints/bitnet_ternary_partition.json
"""
import sys
import json
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional

# 注册 cim::matmul custom op (torch.export.load 反序列化 .pt2 需要 op 已注册)
from cim_compiler.export import cim_op  # noqa: F401

import torch
import torch.export
import torch.fx as fx

from cim_compiler.partition.classify import mark_cim_nodes, node_backend, is_cim_matmul


@dataclass
class CimBlock:
    idx: int
    bitlinear_name: str       # 对应 BitLinear 路径 (从 w_packed 名解析); qkv: q.proj (primary)
    int_mm: str               # cim.matmul op 节点名; qkv: q 的
    w_packed: str             # w_packed placeholder 节点名; qkv: q 的
    x_int8_in: str            # CPU->CIM 边界: 激活 int8 输入节点名 (qkv: q/k/v 共享)
    acc_out: str              # CIM->CPU 边界: int32 输出节点名; qkv: q 的
    is_kv_proj: bool = False  # S3: 是否 K/V proj (KV cache 产出者, bitlinear_name 含 k.proj/v.proj)
    # S6: qkv 合并补充字段 (is_qkv=True 时, q 复用上面字段, k/v 用下面; x_int8_in 各自独立 -- q/k/v 各自 RMSNorm+quantize 致 x_int8 不同, A_PAGE 3 组)
    is_qkv: bool = False
    x_int8_in_k: str = ""
    x_int8_in_v: str = ""
    # S6 方案B: w_packed placeholder 在 input_specs 的 arg_number (L3 识别 qkv 用, -1=未知)
    w_packed_arg: int = -1
    w_packed_arg_k: int = -1
    w_packed_arg_v: int = -1
    bitlinear_name_k: str = ""
    bitlinear_name_v: str = ""
    w_packed_k: str = ""
    w_packed_v: str = ""
    int_mm_k: str = ""
    int_mm_v: str = ""
    acc_out_k: str = ""
    acc_out_v: str = ""


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

    custom op 模式: cim.matmul 的 args[1] 直接是 w_packed placeholder, 无需追溯解包链
    (解包在 op 内部)。此函数保留追溯能力以兼容非直接传入的情况。
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


def _parse_bitlinear_name(name: str) -> str:
    """[P1-4] 从 w_packed target/placeholder 名解析 BitLinear 路径 -> layers.0.attn.q.proj。

    兼容三种命名 (鲁棒):
      target 名     : m.layers.0.attn.q_proj.w_packed  (export graph_signature, 最稳定)
      placeholder 名: b_m_layers_0_attn_q_proj_w_packed (新 export, 含 m_ 模块根)
      placeholder 名: b_layers_0_attn_q_proj_w_packed   (旧 export, 无 m_)
    去 b_/p_ + m./m_ 模块根前缀 + .w_packed/_w_packed 后缀, _ -> . 还原层级 (与 emit_instr._norm 一致)。
    """
    for prefix in ("b_", "p_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    if name.startswith("m."):
        name = name[2:]
    elif name.startswith("m_"):
        name = name[2:]
    for suffix in (".w_packed", "_w_packed", ".weight", "_weight"):
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

    # [P1-4] placeholder name -> target 名映射 (target 名 m.layers.0... 更稳定, 不依赖 b_/p_ 前缀)
    ph_to_target = {}
    ph_to_arg = {}   # S6 方案B: placeholder 名 -> graph block arg 编号 (L3 识别 qkv; 用 graph.nodes 顺序, 非 input_specs)
    _arg_idx = 0
    for _n in graph.nodes:           # graph placeholder 顺序 = block arg 编号 (= linalg IR %argN)
        if _n.op == "placeholder":
            ph_to_arg[_n.name] = _arg_idx
            _arg_idx += 1
    for s in prog.graph_signature.input_specs:
        if s.arg is not None and s.target:
            ph_to_target[s.arg.name] = s.target

    # CIM 块: 每个 cim.matmul op 一个 (无解包链, 解包在 op 内)
    cim_blocks = []
    cpu_to_cim = []
    cim_to_cpu = []

    int_mms = [n for n in graph.nodes if is_cim_matmul(n)]
    # 解析所有 int_mm 的元数据 (mm, bitlinear_name, w_packed_name, x_int8_node)
    parsed = []
    for mm in int_mms:
        x_int8 = mm.args[0]
        w_node = mm.args[1]
        w_packed = _find_w_packed(w_node)
        wp_name = w_packed.name if w_packed else "?"
        target = ph_to_target.get(wp_name, wp_name)
        bitlinear_name = _parse_bitlinear_name(target) if w_packed else "?"
        parsed.append((mm, bitlinear_name, w_packed.name if w_packed else "?", x_int8))

    # S6: 按 layer 分组 q/k/v triplet (layers.N.attn.{q,k,v}.proj, 共享同一 x_int8 节点) 合并为 1 cim_block
    import re
    used = [False] * len(parsed)
    idx = 0
    for i, (mm, name, wp, x_int8) in enumerate(parsed):
        if used[i]:
            continue
        m = re.match(r"layers\.(\d+)\.attn\.q\.proj$", name)
        if m:
            layer = m.group(1)
            ki = next((j for j in range(i + 1, len(parsed))
                       if not used[j]
                       and re.match(rf"layers\.{layer}\.attn\.k\.proj$", parsed[j][1])), None)
            vi = next((j for j in range(i + 1, len(parsed))
                       if not used[j]
                       and re.match(rf"layers\.{layer}\.attn\.v\.proj$", parsed[j][1])), None)
            if ki is not None and vi is not None:
                kmm, kname, kwp, _ = parsed[ki]
                vmm, vname, vwp, _ = parsed[vi]
                cim_blocks.append(CimBlock(
                    idx=idx, bitlinear_name=name, int_mm=mm.name, w_packed=wp,
                    x_int8_in=x_int8.name, acc_out=mm.name, is_qkv=True,
                    x_int8_in_k=kmm.args[0].name, x_int8_in_v=vmm.args[0].name,
                    w_packed_arg=ph_to_arg.get(wp, -1),
                    w_packed_arg_k=ph_to_arg.get(kwp, -1),
                    w_packed_arg_v=ph_to_arg.get(vwp, -1),
                    bitlinear_name_k=kname, w_packed_k=kwp, int_mm_k=kmm.name, acc_out_k=kmm.name,
                    bitlinear_name_v=vname, w_packed_v=vwp, int_mm_v=vmm.name, acc_out_v=vmm.name,
                ))
                used[i] = used[ki] = used[vi] = True
                idx += 1
                # q/k/v 各自 x_int8 边界 (3 组 A_PAGE, 不共享)
                for xx in (x_int8, kmm.args[0], vmm.args[0]):
                    cpu_to_cim.append(Boundary(node=xx.name, direction="cpu_to_cim",
                                               dtype=_dtype_desc(xx), shape=_shape_desc(xx)))
                for pmm in (mm, kmm, vmm):
                    cim_to_cpu.append(Boundary(node=pmm.name, direction="cim_to_cpu",
                                               dtype=_dtype_desc(pmm), shape=_shape_desc(pmm)))
                continue
        # 非合并: 原逻辑 (o.proj / mlp / lm_head, 或 triplet 不完整)
        is_kv = name.endswith("k.proj") or name.endswith("v.proj")
        cim_blocks.append(CimBlock(
            idx=idx,
            bitlinear_name=name,
            int_mm=mm.name,
            w_packed=wp,
            x_int8_in=x_int8.name,
            acc_out=mm.name,
            is_kv_proj=is_kv,
            w_packed_arg=ph_to_arg.get(wp, -1),
        ))
        used[i] = True
        idx += 1
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
            "kv_proj": sum(1 for b in part.cim_blocks if b.is_kv_proj),
            "qkv_group": sum(1 for b in part.cim_blocks if b.is_qkv),
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
          f"CIM块={summary['cim_blocks']} (qkv合并={summary['qkv_group']}, 共 {summary['total_call_function']} call_function)", file=sys.stderr)
    print(f"[partition] 边界: cpu_to_cim={len(part.boundaries['cpu_to_cim'])} "
          f"cim_to_cpu={len(part.boundaries['cim_to_cpu'])}", file=sys.stderr)
    print(f"[partition] saved: {args.out}", file=sys.stderr)
    for b in part.cim_blocks[:5]:
        print(f"  [{b.idx:2d}] {b.bitlinear_name}: "
              f"x_in={b.x_int8_in}, acc={b.acc_out}", file=sys.stderr)
    if len(part.cim_blocks) > 5:
        print(f"  ... (共 {len(part.cim_blocks)} 个 CIM 块)", file=sys.stderr)


if __name__ == "__main__":
    main()
