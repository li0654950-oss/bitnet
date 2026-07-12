#!/usr/bin/env python3
"""S3 KV cache 验收: 数值回归 (全序列 greedy == KV cache greedy) + CPU wall-clock。

对比:
  1. generate_full (全序列重算, attn use_cache=False) vs generate_kv (增量 KV cache)
     -- greedy token 级必须完全一致 (n <= block_size, 绝对位置 == 原模型相对位置)
  2. wall-clock: n=32 vs n=128, 验证 KV cache 相对全序列重算的加速比
     (全序列每步重算 T 个 token 的 proj+attention: O(n²·d²)+O(n³·d);
      KV cache 每步 M=1: O(n·d²)+O(n²·d), 省约 n 倍)

用法: conda run -n nanogpt-gpu python cim_compiler/export/verify_kv.py
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BITNET = os.path.join(REPO, "bitnet")
if BITNET not in sys.path:
    sys.path.insert(0, BITNET)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch
from inference_model import build_inference_model, generate_kv


@torch.no_grad()
def generate_full(model, idx0, n, block_size=256):
    """全序列重算 greedy (attn use_cache=False, 与原 model.generate 数值等价)。

    每步把截断后的全 idx 喂入, 重算所有 token 的 K/V proj (CIM matmul M=T) + attention。
    """
    idx = idx0.clone()
    tokens = idx[0].tolist()
    for _ in range(n):
        if idx.shape[1] >= block_size:
            idx = idx[:, -block_size:]
        x = model.embed_tokens(idx)
        for layer in model.layers:
            x = layer(x)                       # Block.forward, attn use_cache=False
        x = model.ln_f(x)
        logits = model.lm_head(x)
        nxt = int(logits[0, -1].argmax())
        tokens.append(nxt)
        idx = torch.cat([idx, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
    return tokens


def main():
    ternary = os.path.join(REPO, "checkpoints/bitnet_shakespeare_char_ternary.pt")
    if len(sys.argv) > 1:
        ternary = sys.argv[1]
    model = build_inference_model(ternary, vocab_size=65)
    print(f"[verify] model built", file=sys.stderr)

    idx0 = torch.tensor([[0]], dtype=torch.long)   # BOS
    block_size = 256

    print(f"{'n':>5} | {'full(s)':>10} | {'kv(s)':>10} | {'speedup':>8} | token一致")
    print("-" * 55)
    all_ok = True
    for n in (32, 128):
        assert 1 + n <= block_size, f"n={n} 超出 block_size (无 crop 区间)"
        t0 = time.perf_counter()
        full = generate_full(model, idx0, n, block_size)
        t_full = time.perf_counter() - t0

        t0 = time.perf_counter()
        kv = generate_kv(model, idx0, n, block_size)
        t_kv = time.perf_counter() - t0

        match = (full == kv)
        all_ok = all_ok and match
        speedup = t_full / t_kv if t_kv > 0 else float("inf")
        print(f"{n:>5} | {t_full:>10.3f} | {t_kv:>10.3f} | {speedup:>7.2f}x | "
              f"{'✓ YES' if match else '✗ NO'}")
        if not match:
            for i, (a, b) in enumerate(zip(full, kv)):
                if a != b:
                    print(f"  首个分歧 @ token[{i}]: full={a} kv={b}")
                    break
            print(f"  full: {full[:40]}{'...' if len(full)>40 else ''}")
            print(f"  kv:   {kv[:40]}{'...' if len(kv)>40 else ''}")

    print("-" * 55)
    print(f"[verify] {'全部通过 ✓' if all_ok else '存在数值分歧 ✗'}", file=sys.stderr)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
