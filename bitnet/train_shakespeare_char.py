#!/usr/bin/env python3
"""
Train a ~15M ternary-weight BitNet on shakespeare_char.

Combines nanoGPT's training loop (random-crop batches, grad-clip, periodic
eval + sampling) with BitNet's paper lr / weight-decay schedule. Weights are
STE-ternarized to {-1, 0, +1} inside BitLinear — no model.py changes.

Model:  d_model=512, n_layer=6, n_head=8, n_kv_head=4, ffn_dim=1664, block=256
        -> 15,041,280 params (15.04M), bfloat16

Checkpoints:
  checkpoints/bitnet_shakespeare_char_best.pt   <- val-optimal (auto-saved)
  checkpoints/bitnet_shakespeare_char_<N>.pt    <- final-step weights

Usage:
  python bitnet/train_shakespeare_char.py --smoke          # 20-step sanity check
  python bitnet/train_shakespeare_char.py                  # full 5000-step run
  python bitnet/train_shakespeare_char.py --patience 5     # early stop (default)
  python bitnet/train_shakespeare_char.py --patience 0      # disable early stop
"""
import os
import sys
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from model import BitNet
from data_char import get_batch, CharTokenizer, get_meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="20-step sanity run")
    p.add_argument("--device", default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=200)
    p.add_argument("--sample_tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--patience", type=int, default=5,
                   help="early stop after N evals w/o val improvement (0=off)")
    p.add_argument("--min_delta", type=float, default=0.0,
                   help="min val decrease to count as an improvement")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device: {device}")

    meta = get_meta()
    vocab_size = meta["vocab_size"]
    assert vocab_size == 65, f"expected shakespeare_char vocab=65, got {vocab_size}"
    print(f"vocab_size: {vocab_size}")

    # ── model: 15.04M ternary BitNet ──────────────────────────────────────
    model = BitNet(
        vocab_size=vocab_size,
        d_model=512,
        block_size=args.block_size,
        n_layer=6,
        n_head=8,
        n_kv_head=4,
        ffn_dim=1664,
    ).to(device, dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,} ({n_params/1e6:.2f}M)  dtype: bfloat16")

    # ── optimizer + BitNet paper schedule (1.5e-3 -> 8e-4 -> 5e-4 -> 0) ────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    max_iters = 20 if args.smoke else args.max_iters
    eval_interval = 10 if args.smoke else args.eval_interval
    eval_iters = 20 if args.smoke else args.eval_iters

    phase1 = max(1, int(0.5 * max_iters))
    ratio1 = 8e-4 / args.lr
    sched1 = LinearLR(optimizer, start_factor=1.0, end_factor=ratio1, total_iters=phase1)
    ratio2_start = (5e-4 / args.lr) / ratio1
    sched2 = LinearLR(
        optimizer, start_factor=ratio2_start, end_factor=0.0,
        total_iters=max(1, max_iters - phase1),
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[sched1, sched2], milestones=[phase1]
    )

    tok = CharTokenizer(meta)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                xb, yb = get_batch(split, args.batch_size, args.block_size, device)
                _, loss = model(xb, yb)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    os.makedirs("checkpoints", exist_ok=True)
    best_path = "checkpoints/bitnet_shakespeare_char_best.pt"
    t0 = time.time()
    best_val = float("inf")
    best_step = 0
    bad_counter = 0
    stopped_early = False

    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss()
            print(
                f"step {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                f"| lr {optimizer.param_groups[0]['lr']:.2e} | {time.time()-t0:.1f}s",
                flush=True,
            )

            # best-checkpoint saving + early stopping
            if losses["val"] < best_val - args.min_delta:
                best_val = losses["val"]
                best_step = it
                bad_counter = 0
                if not args.smoke:
                    torch.save(model.state_dict(), best_path)
                    print(f"  ★ new best val {best_val:.4f} @ step {it} -> saved {best_path}")
            else:
                bad_counter += 1
                if (not args.smoke) and args.patience > 0 and bad_counter >= args.patience:
                    print(
                        f"early stopping @ step {it}: no val improvement for "
                        f"{args.patience} evals (best {best_val:.4f} @ step {best_step})"
                    )
                    stopped_early = True
                    break

            if not args.smoke:
                model.eval()
                with torch.no_grad():
                    print("--- sample ---")
                    model.generate(args.sample_tokens, tok, device)
                    print("--- end sample ---")
                model.train()

        xb, yb = get_batch("train", args.batch_size, args.block_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        # BitNet paper: zero out weight decay halfway through
        if it == phase1:
            for pg in optimizer.param_groups:
                pg["weight_decay"] = 0.0

    if not args.smoke:
        final_path = f"checkpoints/bitnet_shakespeare_char_{max_iters}.pt"
        torch.save(model.state_dict(), final_path)
        tag = " (early stopped)" if stopped_early else ""
        print(f"saved final: {final_path}")
        print(f"best val {best_val:.4f} @ step {best_step} -> {best_path}{tag}")
    else:
        print(f"smoke done. final loss {loss:.4f}, best val {best_val:.4f}")


if __name__ == "__main__":
    main()
