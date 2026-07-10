#!/usr/bin/env python3
"""PoC T1: 验证 torch.export 对 custom op 保留为 op 节点 (不内联)。

注册 dummy cim::matmul custom op, torch.export 导出,
检查图里是 call_function 单节点 (target=torch.ops.cim.matmul.default),
未被内联展开成 _int_mm / aten.matmul / mul+sum 等底层算子。

这是 temp.md 的核心假设: custom op 让 CPU/CIM 在 IR 里天然分离的前提。
temp.md 第1行已验证 (旧环境), 本 PoC 在 torch 2.12 dev 复现确认。

运行: nanogpt-gpu python cim_compiler/ir/poc_export.py
"""
import torch
import torch.export


# 注册 dummy cim::matmul custom op (float matmul 仿真, 纯验证 export 保留)
@torch.library.custom_op("cim::matmul", mutates_args=())
def cim_matmul(x_int8: torch.Tensor, w_packed: torch.Tensor) -> torch.Tensor:
    """CIM Macro (dummy): int8 x packed-ternary -> int32, 这里 float matmul 仿真。"""
    return (x_int8.to(torch.float32) @ w_packed.to(torch.float32).t()).to(torch.int32)


@cim_matmul.register_fake
def _(x_int8: torch.Tensor, w_packed: torch.Tensor) -> torch.Tensor:
    M, _ = x_int8.shape
    N = w_packed.shape[0]
    return torch.empty(M, N, dtype=torch.int32)


class DummyModel(torch.nn.Module):
    """CPU op + custom op + CPU op, 验证 custom op 不被内联。"""
    def forward(self, x_int8: torch.Tensor, w_packed: torch.Tensor) -> torch.Tensor:
        y = x_int8 + 1                            # CPU
        acc = torch.ops.cim.matmul(y, w_packed)   # CIM custom op (应保留为 op 节点)
        return acc + 2                            # CPU


def main():
    m = DummyModel().eval()
    x = torch.randint(-128, 127, (4, 8), dtype=torch.int8)
    w = torch.randint(-1, 2, (6, 8), dtype=torch.int8)

    prog = torch.export.export(m, (x, w))
    g = prog.graph
    print("=== export graph: call_function 节点 ===")
    for n in g.nodes:
        if n.op == "call_function":
            print(f"  {n.name}: {n.target}")

    # 1. cim.matmul op 节点数
    #    custom op target str = "cim.matmul.default" (namespace.opname.overload, 点分隔非 ::)
    cf = [n for n in g.nodes if n.op == "call_function"]
    if cf:
        print(f"custom op target str 样例: {str(cf[0].target)!r} ... {str(cf[1].target)!r}")
    cim_nodes = [n for n in g.nodes
                 if n.op == "call_function" and "cim.matmul" in str(n.target)]
    print(f"\ncim.matmul op 节点数: {len(cim_nodes)}")
    for n in cim_nodes:
        print(f"  {n.name}: target={n.target}")

    assert len(cim_nodes) == 1, f"期望 1 个 cim.matmul op 节点, 实际 {len(cim_nodes)}"

    # 2. 没有被内联成底层 matmul 类算子
    inlined = [n.name for n in g.nodes if n.op == "call_function"
               and "cim.matmul" not in str(n.target)
               and any(k in str(n.target) for k in ("_int_mm", "aten.matmul", "aten.mm", ".mul"))]
    print(f"内联展开的 matmul 类节点: {inlined or '无'}")
    assert not inlined, f"custom op 被内联展开成底层算子: {inlined}"

    # 3. 数值一致 (export 后 forward 等价)
    ref = m(x, w)
    got = prog.module()(x, w)
    assert torch.equal(ref, got), f"数值不符 max|diff|={(ref - got).abs().max()}"
    print(f"\n✓ T1 PASS: custom op 在 export 图保留为 1 个 op 节点 (不内联), 数值一致")


if __name__ == "__main__":
    main()
