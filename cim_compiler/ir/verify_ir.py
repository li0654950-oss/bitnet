#!/usr/bin/env python3
"""验证导出的 MLIR (.mlir) CPU/CIM 分离正确性 (对照 partition.json)。

校验:
  1. CIM 侧: torch.operator "torch.cim.matmul" 数 == partition.cim_blocks (应 37)
  2. CPU 侧: torch.aten.* op 数 > 0 (CPU op 保留为 torch dialect, 未降级 linalg)
  3. 无 linalg (RAW 模式不降级, linalg 留 lowering 阶段)
  4. 对照 partition.json: CIM 节点数一致

用法:
  python cim_compiler/ir/verify_ir.py
  python cim_compiler/ir/verify_ir.py --mlir checkpoints/bitnet_ternary.mlir \\
    --partition checkpoints/bitnet_ternary_partition.json
"""
import sys
import json
import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mlir", default="checkpoints/bitnet_ternary.mlir")
    p.add_argument("--partition", default="checkpoints/bitnet_ternary_partition.json")
    args = p.parse_args()

    with open(args.mlir) as f:
        ir = f.read()
    with open(args.partition) as f:
        part = json.load(f)

    ok = True
    expect_cim = part["summary"]["cim_blocks"]

    # 1. CIM 侧: torch.operator "torch.cim.matmul" (custom op 保留为 op)
    n_cim = ir.count('torch.operator "torch.cim.matmul"')
    c1 = (n_cim == expect_cim)
    ok &= c1
    print(f"[CIM] torch.operator 'torch.cim.matmul'={n_cim} (应={expect_cim}) "
          f"{'OK' if c1 else 'FAIL'}")

    # 2. CPU 侧: torch.aten.* (RAW 模式保留为 torch dialect, 未降级 linalg)
    n_aten = ir.count("torch.aten.")
    c2 = (n_aten > 0)
    ok &= c2
    print(f"[CPU] torch.aten.*={n_aten} (>0, CPU op 保留 torch dialect) {'OK' if c2 else 'FAIL'}")

    # 3. 无 linalg (RAW 模式不降级; linalg 留 lowering 阶段)
    n_linalg = ir.count("linalg.")
    c3 = (n_linalg == 0)
    ok &= c3
    print(f"[未降级] linalg op={n_linalg} (应=0, RAW; linalg 留 lowering) {'OK' if c3 else 'FAIL'}")

    # 4. 对照 partition node_backend: CIM 节点数一致
    n_cim_nodes = sum(1 for v in part["node_backend"].values() if v == "CIM")
    c4 = (n_cim_nodes == expect_cim)
    ok &= c4
    print(f"[对照] partition CIM 节点={n_cim_nodes} (应={expect_cim}) {'OK' if c4 else 'FAIL'}")

    print("\n" + ("ALL OK ✓" if ok else "FAIL ✗"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
