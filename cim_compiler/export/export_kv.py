#!/usr/bin/env python3
"""导出增量 KV cache decode forward 为 .pt2 (接入点③: JIT/AOT 实测 KV cache)。

_KVCacheModel.forward(idx[B,1], k_caches[L,B,T,n_kv,hd], v_caches, cos, sin)
  -> (logits[B,1,vocab], new_k_caches, new_v_caches)
单 token decode + 动态 T cache (cat/SDPA/repeat), CIM matmul M=1 (decode 单 token)。
供 lowering -> cim_stub 增量 ABI (cim_main prefill+decode 循环)。

用法: python cim_compiler/export/export_kv.py
"""
import os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BITNET = os.path.join(REPO, "bitnet")
if BITNET not in sys.path: sys.path.insert(0, BITNET)
if HERE not in sys.path: sys.path.insert(0, HERE)

import torch
from torch.export import Dim
from inference_model import build_inference_model, _KVCacheModel
import cim_op  # noqa
from weight_blob import write_weight_blob
from data_char import get_meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ternary", default="checkpoints/bitnet_shakespeare_char_ternary.pt")
    p.add_argument("--out_graph", default="checkpoints/bitnet_ternary_kv.pt2")
    p.add_argument("--out_blob", default="checkpoints/bitnet_ternary_weights.bin")
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_kv_head", type=int, default=4)
    p.add_argument("--ffn_dim", type=int, default=1664)
    args = p.parse_args()

    meta = get_meta()
    model = build_inference_model(args.ternary, vocab_size=meta["vocab_size"],
        d_model=args.d_model, block_size=args.block_size, n_layer=args.n_layer,
        n_head=args.n_head, n_kv_head=args.n_kv_head, ffn_dim=args.ffn_dim)
    kvm = _KVCacheModel(model)
    n_layer = len(model.layers)
    attn0 = model.layers[0].attn
    n_kv, head_dim, inv_freq = attn0.n_kv, attn0.head_dim, attn0.inv_freq

    # sample: idx[1,1], cache T=2 (已 prefill 2 token), 新 token 位置 2
    idx = torch.zeros(1, 1, dtype=torch.long)
    T = 2
    k_caches = torch.zeros(n_layer, 1, T, n_kv, head_dim)
    v_caches = torch.zeros(n_layer, 1, T, n_kv, head_dim)
    pos = torch.tensor([2.0])
    freqs = torch.einsum("t,d->td", pos, inv_freq)
    cos = freqs.cos()[None, :, None, :]                       # [1,1,1,hd/2]
    sin = freqs.sin()[None, :, None, :]

    prog = torch.export.export(kvm, (idx, k_caches, v_caches, cos, sin),
        dynamic_shapes=(None, {2: Dim("T", min=1, max=256)}, {2: Dim("T", min=1, max=256)}, None, None))
    torch.export.save(prog, args.out_graph)
    n_cf = sum(1 for n in prog.graph.nodes if n.op == "call_function")
    n_cim = sum(1 for n in prog.graph.nodes if n.op == "call_function" and "cim.matmul" in str(n.target))
    print(f"[export_kv] {n_cf} call_function, {n_cim} cim.matmul (decode M=1)", file=sys.stderr)
    print(f"[export_kv] saved: {args.out_graph}", file=sys.stderr)
    write_weight_blob(model, args.out_blob)
    print(f"[export_kv] weights -> {args.out_blob}", file=sys.stderr)


if __name__ == "__main__":
    main()
