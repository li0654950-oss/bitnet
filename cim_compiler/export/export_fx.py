#!/usr/bin/env python3
"""导出 BitNet 推理模型为 FX graph (.pt2) + 旁路权重 blob (.bin)。

产物:
  - .pt2: torch.export ExportedProgram。2bit 三值权重作常量内嵌, 动态 seq len [1..block_size]。
          cim::matmul custom op 保留为 op 节点 (不内联), CPU/CIM 在 IR 天然分离。
          _LogitsOnly 包装: 只返回 logits (无 None 输出, torch-mlir 降级不支持 None)。
  - .bin: CIM 权重预加载 blob (自描述二进制, 见 weight_blob.py)。

用法:
  python cim_compiler/export/export_fx.py
  python cim_compiler/export/export_fx.py --ternary checkpoints/bitnet_shakespeare_char_ternary.pt \\
    --out_graph checkpoints/bitnet_ternary.pt2 --out_blob checkpoints/bitnet_ternary_weights.bin
"""
import argparse
import sys

import torch
import torch.export
from torch.export import Dim

from cim_compiler.export.inference_model import _LogitsOnly
from cim_compiler.export.weight_blob import write_weight_blob
from cim_compiler.export.export_common import add_model_args, build_model_from_args


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p)
    p.add_argument("--out_graph", default="checkpoints/bitnet_ternary.pt2")
    p.add_argument("--out_blob", default="checkpoints/bitnet_ternary_weights.bin")
    args = p.parse_args()

    model = build_model_from_args(args)
    n_bl = sum(1 for m in model.modules() if m.__class__.__name__ == "BitLinearInference")
    print(f"[build] {n_bl} BitLinear -> BitLinearInference (cim::matmul custom op)", file=sys.stderr)

    # _LogitsOnly 包装: 只返回 logits (BitNet.forward 返回 (logits, None), torch-mlir 降级不支持 None)
    export_model = _LogitsOnly(model)
    idx = torch.zeros(1, 4, dtype=torch.long)
    prog = torch.export.export(
        export_model, (idx,),
        dynamic_shapes={"idx": {1: Dim("T", max=args.block_size)}},
    )
    torch.export.save(prog, args.out_graph)

    n_cf = sum(1 for n in prog.graph.nodes if n.op == "call_function")
    n_cim = sum(1 for n in prog.graph.nodes if n.op == "call_function" and n.target == torch.ops.cim.matmul.default)
    print(f"[export] {n_cf} call_function 节点, {n_cim} cim.matmul op (= {n_bl} BitLinear)", file=sys.stderr)
    print(f"[export] saved: {args.out_graph}", file=sys.stderr)

    # 旁路 weight blob (CIM Preload 用) - 用原 BitNet (model, 非 _LogitsOnly 包装)
    n = write_weight_blob(model, args.out_blob)
    print(f"[blob] {n} BitLinear 权重 -> {args.out_blob}", file=sys.stderr)


if __name__ == "__main__":
    main()
