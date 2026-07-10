#!/usr/bin/env python3
"""导出 BitNet FX graph (.pt2) 为 MLIR (torch dialect, RAW 模式)。

用 torch-mlir fx.export_and_import (OutputType.RAW) 把 ExportedProgram 导成 MLIR:
  - cim::matmul custom op -> torch.operator "torch.cim.matmul" (保留, 不降级)
  - CPU op (norm/quant/rescale/attention/...) -> torch.aten.* (未降级 linalg)

LINALG_ON_TENSORS 不可用: torchdynamo-export-to-torch-backend-pipeline 把 unknown
torch.operator 标记 illegal (backend_legal_ops 无效), 故用 RAW (见 poc_torch_mlir.py T2)。
CPU 降级 linalg 移到 lowering 阶段 (先 cim.matmul -> func.call 消除 unknown op, 再跑 linalg pipeline)。

用法:
  python cim_compiler/ir/to_mlir.py
  python cim_compiler/ir/to_mlir.py --graph checkpoints/bitnet_ternary.pt2 --out checkpoints/bitnet_ternary.mlir
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
# HERE = cim_compiler/ir/ ; 上两层 = repo root
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
# 注册 cim::matmul custom op (torch.export.load 反序列化 + torch-mlir import 需要)
_EXPORT_DIR = os.path.join(REPO, "cim_compiler", "export")
if _EXPORT_DIR not in sys.path:
    sys.path.insert(0, _EXPORT_DIR)
import cim_op  # noqa: F401

import torch
import torch.export
from torch_mlir.fx import export_and_import
from torch_mlir.compiler_utils import OutputType


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graph", default="checkpoints/bitnet_ternary.pt2")
    p.add_argument("--out", default="checkpoints/bitnet_ternary.mlir")
    args = p.parse_args()

    prog = torch.export.load(args.graph)
    mod = export_and_import(prog, output_type=OutputType.RAW, func_name="main")
    ir = str(mod)

    with open(args.out, "w") as f:
        f.write(ir)

    n_cim = ir.count('torch.operator "torch.cim.matmul"')
    n_aten = ir.count("torch.aten.")
    print(f"[mlir] RAW 模式导出: torch.operator 'torch.cim.matmul'={n_cim}, torch.aten.*={n_aten}",
          file=sys.stderr)
    print(f"[mlir] saved: {args.out} ({len(ir)} chars)", file=sys.stderr)


if __name__ == "__main__":
    main()
