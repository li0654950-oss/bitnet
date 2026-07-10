#!/usr/bin/env python3
"""L2: 跑完整 LINALG_ON_TENSORS pipeline (placeholder+CPU aten -> linalg)。

加载 L1 产出 (placeholder.mlir, 含 aten.mm placeholder + CPU aten.*), 跑 torch-mlir
LINALG_ON_TENSORS pipeline:
  step1 torchdynamo-export-to-torch-backend-pipeline        (TorchFX -> Torch Backend IR)
  step2 torch-backend-to-linalg-on-tensors-backend-pipeline  (-> linalg)

placeholder (aten.mm) 降级到 linalg.matmul (含 int8 cast 链, L3 据此替换为 func.call @cim_launch);
CPU aten 降级到 linalg.generic; attention 降级到 tm_tensor。

gate: linalg.matmul=37 (placeholder 降级产物), torch.aten=0, torch.operator=0。

用法:
  python cim_compiler/lowering/linalg_lower.py
  python cim_compiler/lowering/linalg_lower.py --in checkpoints/bitnet_ternary_placeholder.mlir \\
    --out checkpoints/bitnet_ternary_linalg.mlir
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d
from torch_mlir.fx import _module_lowering
from torch_mlir.compiler_utils import OutputType


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default="checkpoints/bitnet_ternary_placeholder.mlir")
    p.add_argument("--out", default="checkpoints/bitnet_ternary_linalg.mlir")
    args = p.parse_args()

    src = open(args.inp).read()
    ctx = ir.Context()
    torch_d.register_dialect(ctx)
    mod = ir.Module.parse(src, ctx)

    print(f"[L2] 加载 {args.inp}", file=sys.stderr)
    print(f"[L2] 跑 LINALG_ON_TENSORS pipeline (step1 torchdynamo + step2 linalg)...", file=sys.stderr)

    try:
        _module_lowering(False, False, OutputType.LINALG_ON_TENSORS, mod)
    except Exception as e:
        print(f"[L2] LINALG FAIL: {type(e).__name__}", file=sys.stderr)
        print(f"[L2] {str(e)[:1000]}", file=sys.stderr)
        sys.exit(1)

    ir_out = str(mod)
    with open(args.out, "w") as f:
        f.write(ir_out)

    n_linalg_mm = ir_out.count("linalg.matmul")
    n_linalg_gen = ir_out.count("linalg.generic")
    n_tm = ir_out.count("tm_tensor.")
    n_aten = ir_out.count("torch.aten.")
    n_op = ir_out.count("torch.operator")
    print(f"[L2] LINALG OK ✓", file=sys.stderr)
    print(f"[L2] linalg.matmul={n_linalg_mm} (placeholder 降级), linalg.generic={n_linalg_gen} (CPU), "
          f"tm_tensor={n_tm} (attention), torch.aten={n_aten}, torch.operator={n_op}", file=sys.stderr)
    print(f"[L2] saved: {args.out}", file=sys.stderr)

    ok = True
    if n_op != 0:
        print(f"[L2] FAIL: 残留 torch.operator={n_op}", file=sys.stderr); ok = False
    if n_aten != 0:
        print(f"[L2] FAIL: 残留 torch.aten={n_aten}", file=sys.stderr); ok = False
    if n_linalg_mm != 37:
        print(f"[L2] FAIL: linalg.matmul={n_linalg_mm} (应=37, placeholder 降级产物)", file=sys.stderr); ok = False
    print(f"\n[L2] {'PASS ✓ (placeholder->linalg.matmul, CPU aten->linalg.generic)' if ok else 'FAIL ✗'}",
          file=sys.stderr)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
