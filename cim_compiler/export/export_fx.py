#!/usr/bin/env python3
"""导出 BitNet 推理模型为 FX graph (.pt2) + 旁路权重 blob (.bin)。

产物:
  - .pt2: torch.export ExportedProgram。2bit 三值权重作常量内嵌, 动态 seq len [1..block_size]。
          全原生 ATen 算子, 零 custom op。compiler 后端可 pattern-match matmul 调度到 CIM Macro。
  - .bin: CIM 权重预加载 blob (自描述二进制, 见 weight_blob.py)。

用法:
  python cim_compiler/export/export_fx.py
  python cim_compiler/export/export_fx.py --ternary checkpoints/bitnet_shakespeare_char_ternary.pt \\
    --out_graph checkpoints/bitnet_ternary.pt2 --out_blob checkpoints/bitnet_ternary_weights.bin
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
REPO = os.path.dirname(os.path.dirname(HERE))
BITNET = os.path.join(REPO, "bitnet")
if BITNET not in sys.path:
    sys.path.insert(0, BITNET)

import torch
import torch.export
from torch.export import Dim

from inference_model import build_inference_model
from weight_blob import write_weight_blob
from data_char import get_meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ternary", default="checkpoints/bitnet_shakespeare_char_ternary.pt")
    p.add_argument("--out_graph", default="checkpoints/bitnet_ternary.pt2")
    p.add_argument("--out_blob", default="checkpoints/bitnet_ternary_weights.bin")
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_kv_head", type=int, default=4)
    p.add_argument("--ffn_dim", type=int, default=1664)
    args = p.parse_args()

    meta = get_meta()
    model = build_inference_model(
        args.ternary, vocab_size=meta["vocab_size"],
        d_model=args.d_model, block_size=args.block_size,
        n_layer=args.n_layer, n_head=args.n_head,
        n_kv_head=args.n_kv_head, ffn_dim=args.ffn_dim,
    )
    n_bl = sum(1 for m in model.modules() if m.__class__.__name__ == "BitLinearInference")
    print(f"[build] {n_bl} BitLinear -> BitLinearInference (2bit 原生 ATen)", file=sys.stderr)

    # torch.export, 动态 seq len [1..block_size]
    idx = torch.zeros(1, 4, dtype=torch.long)
    prog = torch.export.export(
        model, (idx,),
        dynamic_shapes={"idx": {1: Dim("T", max=args.block_size)}},
    )
    torch.export.save(prog, args.out_graph)

    n_cf = sum(1 for n in prog.graph.nodes if n.op == "call_function")
    n_matmul = sum(1 for n in prog.graph.nodes if n.op == "call_function" and "matmul" in str(n.target))
    print(f"[export] {n_cf} call_function 节点, {n_matmul} matmul (= {n_bl} BitLinear)", file=sys.stderr)
    print(f"[export] saved: {args.out_graph}", file=sys.stderr)

    # 旁路 weight blob (CIM Preload 用) - model 是 BitNet (未包装 _LogitsOnly)
    n = write_weight_blob(model, args.out_blob)
    print(f"[blob] {n} BitLinear 权重 -> {args.out_blob}", file=sys.stderr)


if __name__ == "__main__":
    main()
