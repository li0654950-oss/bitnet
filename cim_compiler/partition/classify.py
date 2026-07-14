#!/usr/bin/env python3
"""FX graph 节点分类 - 标记 CIM 节点 (cim::matmul custom op)。

CIM 节点 = cim.matmul custom op 节点本身 (解包/累加封装在 op 内部, 不内联)。
对应 cim_mlp.md 的 CIM Macro: 2bit 补码节点 + 宏内解包 + int32 累加。
其余节点 (norm / 激活量化 / rescale / attention / embedding) 归 CPU。
"""
import torch
import torch.fx as fx


def is_cim_matmul(node: fx.Node) -> bool:
    """节点是否为 CIM matmul (cim::matmul custom op)。

    target 对象比较 (非字符串包含, 与 export verify_export 一致):
    node.target == torch.ops.cim.matmul.default。
    需 cim_op 已注册 -- 调用方 (partition.py / verify_partition.py) 均 import cim_op。
    """
    return node.op == "call_function" and node.target == torch.ops.cim.matmul.default


def mark_cim_nodes(graph: fx.Graph) -> set:
    """标记所有 CIM 节点: cim.matmul custom op 节点本身。

    custom op 不内联, 解包/累加封装在 op 内部 (见 cim_op.py), 故只标 op 节点,
    无需追溯权重解包链 (与内联 _int_mm 模式不同)。
    """
    return {n for n in graph.nodes if is_cim_matmul(n)}


def node_backend(node: fx.Node, cim_set: set) -> str:
    """返回节点 backend 标签: 'CIM' 或 'CPU'。"""
    return "CIM" if node in cim_set else "CPU"
