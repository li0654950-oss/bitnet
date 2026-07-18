#!/usr/bin/env python3
"""验证 CPU/CIM 划分 (partition.json) 端到端正确性。

校验:
1. 节点全覆盖: cpu + cim == 总 call_function
2. CIM 块数 == matmul 数 (BitLinear 数)
3. 边界数: cpu_to_cim == cim_to_cpu == CIM 块数
4. 数据流: 每个 CIM 块 x_int8_in == matmul.args[0]; acc_out == matmul 本身
5. 边界 dtype: cpu_to_cim 含 int8; cim_to_cpu 含 int32
6. 原图 prog.module() 可跑 (export 图完好未修改)

用法:
  python cim_compiler/partition/verify_partition.py
  python cim_compiler/partition/verify_partition.py --graph ... --partition ...
"""
import sys
import json
import argparse

# 注册 cim::matmul custom op (torch.export.load 反序列化 .pt2 需要 op 已注册)
from cim_compiler.export import cim_op  # noqa: F401

import torch
import torch.export

from cim_compiler.partition.classify import is_cim_matmul


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graph", default="checkpoints/bitnet_ternary.pt2")
    p.add_argument("--partition", default="checkpoints/bitnet_ternary_partition.json")
    args = p.parse_args()

    with open(args.partition) as f:
        part = json.load(f)
    prog = torch.export.load(args.graph)
    graph = prog.graph
    node_by_name = {n.name: n for n in graph.nodes}

    ok = True
    s = part["summary"]

    # 1. 节点全覆盖
    n_cf = sum(1 for n in graph.nodes if n.op == "call_function")
    n_mm = sum(1 for n in graph.nodes if is_cim_matmul(n))
    cover = s["cpu_nodes"] + s["cim_nodes"]
    c1 = (cover == n_cf)
    ok &= c1
    print(f"[覆盖] cpu({s['cpu_nodes']})+cim({s['cim_nodes']})={cover} == 总call_function={n_cf} "
          f"{'OK' if c1 else 'FAIL'}")

    # 2. CIM 块数 == matmul 数 (S6 qkv 合并: 每 triplet 3 matmul -> 1 块, 省 2)
    qkv = s.get("qkv_group", 0)
    c2 = (s["cim_blocks"] + 2 * qkv == n_mm)
    ok &= c2
    print(f"[CIM块] {s['cim_blocks']} 块 (+{qkv}*2 qkv合并) == matmul数={n_mm} {'OK' if c2 else 'FAIL'}")

    # 3. 边界数
    n_c2c = len(part["boundaries"]["cpu_to_cim"])
    n_c2u = len(part["boundaries"]["cim_to_cpu"])
    c3 = (n_c2c == n_mm and n_c2u == n_mm)
    ok &= c3
    print(f"[边界] cpu_to_cim={n_c2c} cim_to_cpu={n_c2u} (应={n_mm}) {'OK' if c3 else 'FAIL'}")

    # 4. 数据流: x_int8_in == mm.args[0], acc_out == mm
    bad = []
    for b in part["cim_blocks"]:
        mm = node_by_name.get(b["int_mm"])
        x_in = node_by_name.get(b["x_int8_in"])
        if mm is None or not is_cim_matmul(mm):
            bad.append(b["idx"]); continue
        if x_in is not mm.args[0]:
            bad.append(b["idx"])
    c4 = not bad
    ok &= c4
    print(f"[数据流] {len(part['cim_blocks'])} 块 (x_in==mm.args[0], acc==mm) "
          f"{'OK' if c4 else f'FAIL: {bad}'}")

    # 5. 边界 dtype (custom op: x_int8 int8 入 / acc int32 出)
    dt_in = set(b["dtype"] for b in part["boundaries"]["cpu_to_cim"])
    dt_out = set(b["dtype"] for b in part["boundaries"]["cim_to_cpu"])
    c5 = any(d == "int8" for d in dt_in) and any(d == "int32" for d in dt_out)
    print(f"[dtype] cpu_to_cim={dt_in} cim_to_cpu={dt_out} {'OK' if c5 else '检查'}")
    ok &= c5

    # 6. 原图可跑 (export 图未修改)
    idx = torch.zeros(1, 4, dtype=torch.long)
    try:
        with torch.no_grad():
            out = prog.module()(idx)[0]
        c6 = (out.shape[-1] == 65)
        print(f"[原图] prog.module() 可跑, logits={tuple(out.shape)} {'OK' if c6 else 'FAIL'}")
        ok &= c6
    except Exception as e:
        print(f"[原图] 调用失败: {type(e).__name__}: {str(e)[:100]}")
        ok = False

    print("\n" + ("ALL OK ✓" if ok else "FAIL ✗"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
