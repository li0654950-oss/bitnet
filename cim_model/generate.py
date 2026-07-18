#!/usr/bin/env python3
"""CIM 定点模型推理生成 (shakespeare_char)。

用 cim_model 定点数据通路 (per-token int8 量化 → 64×64 Macro → int32 累加 → fp32 rescale → 边界降 fp32)
替换 BitLinear.forward, 复用 model.py 全部结构 + nanoGPT 风格采样。

注意: CIM 核心是 numpy int32 (CPU), 故默认 CPU 运行 (即便指定 cuda, BitLinear 内部仍 .cpu() 算)。

Usage:
  python cim_model/generate.py                                  # BOS, 默认
  python cim_model/generate.py --prompt "ROMEO:"                # 条件生成
  python cim_model/generate.py --temperature 0.7 --top_k 40 --max_tokens 500
  python cim_model/generate.py --seed 0                         # 可复现
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bitnet"))

import torch
from torch.nn import functional as F

from model import BitNet
from data_char import CharTokenizer, get_meta
from cim_model.model_cim import patch_bitlinear, unpatch_bitlinear, load_ternary_into_model

CFG = dict(d_model=512, block_size=256, n_layer=6, n_head=8, n_kv_head=4, ffn_dim=1664)
TERNARY = os.path.join(ROOT, "checkpoints", "bitnet_shakespeare_char_ternary.pt")


@torch.no_grad()
def sample(model, idx, max_new_tokens, block_size, temperature, top_k, tok):
    """nanoGPT-style autoregressive sampling (逐 token 流式输出, 有进度可见)."""
    for i in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :].float() / max(temperature, 1e-8)
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(
                logits < v[:, [-1]],
                torch.full_like(logits, float("-inf")),
                logits,
            )
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, idx_next], dim=1)
        print(tok.decode([idx_next.item()]), end="", flush=True)
    return idx


def main():
    p = argparse.ArgumentParser(description="CIM 定点模型推理生成")
    p.add_argument("--prompt", default="", help="起始文本 (空 = BOS)")
    p.add_argument("--max_tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--block_size", type=int, default=256, help="须与训练一致")
    p.add_argument("--device", default="cpu", help="CIM 核心 numpy, 默认 cpu")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    device = args.device
    if args.seed is not None:
        torch.manual_seed(args.seed)

    meta = get_meta()
    tok = CharTokenizer(meta)
    vocab = meta["vocab_size"]

    orig_fwd = patch_bitlinear()
    try:
        model = BitNet(vocab, **CFG).to(device, dtype=torch.float32)
        n_ternary = load_ternary_into_model(model, TERNARY, device)
        model.eval()
        print(f"[CIM 定点通路 | {n_ternary} BitLinear | {device} | "
              f"T={args.temperature} top_k={args.top_k}]", file=sys.stderr)

        if args.prompt:
            idx = torch.tensor([tok.encode(args.prompt)], device=device, dtype=torch.long)
        else:
            idx = torch.zeros((1, 1), device=device, dtype=torch.long)

        print(tok.decode(idx[0].tolist()), end="", flush=True)
        sample(
            model, idx, args.max_tokens, args.block_size,
            args.temperature, args.top_k, tok,
        )
        print()
    finally:
        unpatch_bitlinear(orig_fwd)


if __name__ == "__main__":
    main()
