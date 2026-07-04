#!/usr/bin/env python3
"""
Generate text from a trained BitNet on shakespeare_char.

Loads the val-optimal checkpoint (or any specified) and samples with
nanoGPT-style temperature + top-k. Defaults match nanoGPT shakespeare_char.

Usage:
  python bitnet/generate_shakespeare_char.py                              # from BOS
  python bitnet/generate_shakespeare_char.py --prompt "ROMEO:"             # conditioned
  python bitnet/generate_shakespeare_char.py --temperature 0.7 --top_k 40 --max_tokens 1000
  python bitnet/generate_shakespeare_char.py --seed 0                      # reproducible
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
from torch.nn import functional as F

from model import BitNet
from data_char import CharTokenizer, get_meta


@torch.no_grad()
def sample(model, idx, max_new_tokens, block_size, temperature, top_k):
    """nanoGPT-style autoregressive sampling."""
    for _ in range(max_new_tokens):
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
    return idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/bitnet_shakespeare_char_best.pt")
    p.add_argument("--prompt", default="", help="starting text (empty = BOS)")
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--block_size", type=int, default=256, help="must match training")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--ternary", action="store_true",
                   help="use 2-bit packed ternary weights + Triton kernel")
    p.add_argument("--ternary_path", default="checkpoints/bitnet_shakespeare_char_ternary.pt")
    args = p.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    if args.seed is not None:
        torch.manual_seed(args.seed)

    meta = get_meta()
    tok = CharTokenizer(meta)
    vocab_size = meta["vocab_size"]

    # rebuild model with the training config, then load weights
    model = BitNet(
        vocab_size=vocab_size,
        d_model=512,
        block_size=args.block_size,
        n_layer=6,
        n_head=8,
        n_kv_head=4,
        ffn_dim=1664,
    ).to(device, dtype=torch.bfloat16)

    if args.ternary:
        # 2-bit ternary inference: load packed weights, switch BitLinear to ternary kernel
        from model import BitLinear
        data = torch.load(args.ternary_path, map_location=device, weights_only=True)
        base_sd = {k: v for k, v in data.items() if not isinstance(v, dict)}
        model.load_state_dict(base_sd, strict=False)  # embed_tokens, norms, buffers
        n_ternary = 0
        for name, module in model.named_modules():
            if isinstance(module, BitLinear):
                key = name + ".weight"
                if key in data and isinstance(data[key], dict):
                    module.set_inference(data[key]["packed"].to(device), data[key]["scale"])
                    n_ternary += 1
        ckpt_info = f"ternary 2-bit ({n_ternary} BitLinear kernels) from {args.ternary_path}"
    else:
        sd = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(sd)  # strict=True: errors if block_size mismatches training
        ckpt_info = args.checkpoint
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[loaded {ckpt_info} | {n_params/1e6:.2f}M params | {device} | "
          f"T={args.temperature} top_k={args.top_k}]", file=sys.stderr)

    # initial context: encoded prompt, or BOS (token 0) for char-level
    if args.prompt:
        idx = torch.tensor([tok.encode(args.prompt)], device=device, dtype=torch.long)
    else:
        idx = torch.zeros((1, 1), device=device, dtype=torch.long)

    print(tok.decode(idx[0].tolist()), end="", flush=True)
    out = sample(
        model, idx, args.max_tokens, args.block_size, args.temperature, args.top_k
    )
    print(tok.decode(out[0][idx.size(1):].tolist()))


if __name__ == "__main__":
    main()
