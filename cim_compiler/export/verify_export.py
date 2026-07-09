#!/usr/bin/env python3
"""验证导出的 FX graph (.pt2) + 权重 blob (.bin) 端到端正确性。

校验:
  1. .pt2 可加载, BOS forward 与变长 seq forward 数值均与 eager 推理模型一致
  2. 图节点: matmul 计数 (= BitLinear 数) / custom op (=0) / detach (=0, 无 STE 残留)
  3. 算子分布
  4. weight blob 可读回, 每层 packed/scale/shape 与模型 buffer 一致

用法:
  python cim_compiler/export/verify_export.py
  python cim_compiler/export/verify_export.py --graph checkpoints/bitnet_ternary.pt2 \\
    --blob checkpoints/bitnet_ternary_weights.bin --ternary checkpoints/bitnet_shakespeare_char_ternary.pt
"""
import os
import sys
import argparse
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
REPO = os.path.dirname(os.path.dirname(HERE))
BITNET = os.path.join(REPO, "bitnet")
if BITNET not in sys.path:
    sys.path.insert(0, BITNET)

import torch
import torch.export

from inference_model import build_inference_model, BitLinearInference
from weight_blob import read_weight_blob
from data_char import get_meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graph", default="checkpoints/bitnet_ternary.pt2")
    p.add_argument("--blob", default="checkpoints/bitnet_ternary_weights.bin")
    p.add_argument("--ternary", default="checkpoints/bitnet_shakespeare_char_ternary.pt")
    p.add_argument("--block_size", type=int, default=256)
    args = p.parse_args()

    ok = True

    # eager 推理模型 (参照)
    model = build_inference_model(
        args.ternary, vocab_size=get_meta()["vocab_size"], block_size=args.block_size,
    )
    prog = torch.export.load(args.graph)

    # 1) 数值一致性: BOS (seq=1) 与变长 (seq=7)
    for T, tag in [(1, "BOS"), (7, "seq=7")]:
        idx = torch.zeros(1, T, dtype=torch.long)
        with torch.no_grad():
            ref = model(idx)[0]
            got = prog.module()(idx)[0]
        d = (got - ref).abs().max().item()
        good = d < 1e-4
        ok &= good
        print(f"[数值] {tag}: export vs eager max|diff|={d:.4e} {'OK' if good else 'FAIL'}")

    # 2) 图节点统计
    n_matmul = sum(1 for n in prog.graph.nodes if n.op == "call_function" and "matmul" in str(n.target))
    n_custom = sum(1 for n in prog.graph.nodes if n.op == "call_function" and "cim::" in str(n.target))
    n_detach = sum(1 for n in prog.graph.nodes if "detach" in str(n.target))
    n_bl = sum(1 for m in model.modules() if isinstance(m, BitLinearInference))
    g_ok = (n_matmul == n_bl and n_custom == 0 and n_detach == 0)
    ok &= g_ok
    print(f"[图] matmul={n_matmul} (应={n_bl}) | custom={n_custom} (应=0) | detach/STE={n_detach} (应=0) "
          f"{'OK' if g_ok else 'FAIL'}")

    ops = Counter(str(n.target).split(".")[-1] for n in prog.graph.nodes if n.op == "call_function")
    print(f"[图] 算子 top8: {ops.most_common(8)}")

    # 3) weight blob 校验
    entries = read_weight_blob(args.blob)
    b_ok = (len(entries) == n_bl)
    blob_map = {e.name: e for e in entries}
    for name, mod in model.named_modules():
        if isinstance(mod, BitLinearInference):
            if name not in blob_map:
                b_ok = False; print(f"[blob] MISSING {name}"); continue
            e = blob_map[name]
            buf = mod.w_packed.cpu().to(torch.uint8).contiguous().numpy().tobytes()
            match = (e.packed == buf
                     and abs(e.scale_w - mod.scale_w) < 1e-12
                     and e.N == mod.w_packed.shape[0]
                     and e.K == mod.w_packed.shape[1] * 4)
            if not match:
                b_ok = False; print(f"[blob] MISMATCH {name}")
    ok &= b_ok
    print(f"[blob] {len(entries)} entries, packed/scale/shape 与模型 buffer 一致 "
          f"{'OK' if b_ok else 'FAIL'}")

    print("\n" + ("ALL OK ✓" if ok else "FAIL ✗"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
