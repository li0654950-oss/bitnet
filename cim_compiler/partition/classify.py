#!/usr/bin/env python3
"""FX graph 节点分类 — 标记 CIM 节点 (matmul + 权重解包链)。

CIM 节点 = matmul + 从其权重输入 (args[1]) 反向追溯到 w_packed placeholder 的所有 producer。
对应 cim_mlp.md 的 CIM Macro: 2bit 补码节点 + 宏内解包 + int32 累加。
其余节点 (norm / 激活量化 / rescale / attention / embedding) 归 CPU。
"""
import torch.fx as fx


def is_cim_matmul(node: fx.Node) -> bool:
    """节点是否为 CIM matmul (BitLinear 矩阵乘)。"""
    return node.op == "call_function" and "matmul" in str(node.target)


def mark_cim_nodes(graph: fx.Graph) -> set:
    """标记所有 CIM 节点: matmul + 权重侧 producer 链 (含 w_packed placeholder)。

    从每个 matmul 的权重输入 (args[1], 即 w_int8.t()) 反向追溯所有 producer。
    只追溯权重侧 (args[1]), 不追溯激活侧 (args[0] = x_int8, 归 CPU)。
    """
    cim = set()

    def mark_back(node: fx.Node):
        if node in cim:
            return
        cim.add(node)
        for a in node.args:
            if isinstance(a, fx.Node):
                mark_back(a)
            elif isinstance(a, (list, tuple)):
                # stack 等算子的 args 是 [list_of_nodes], 需展开
                for x in a:
                    if isinstance(x, fx.Node):
                        mark_back(x)

    for n in graph.nodes:
        if is_cim_matmul(n):
            cim.add(n)
            if len(n.args) >= 2 and isinstance(n.args[1], fx.Node):
                mark_back(n.args[1])
    return cim


def node_backend(node: fx.Node, cim_set: set) -> str:
    """返回节点 backend 标签: 'CIM' 或 'CPU'。"""
    return "CIM" if node in cim_set else "CPU"
