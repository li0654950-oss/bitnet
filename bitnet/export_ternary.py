#!/usr/bin/env python3
"""
Export a trained BitNet checkpoint to 2-bit packed ternary weights.

For every BitLinear weight (those with a sibling `<prefix>.norm.weight`):
  bf16 weight  --weight_quant-->  ternary {-1,0,+1} + per-tensor scale
  ternary  --encode (code=ternary+1 in {0,1,2})-->  2-bit
  2-bit     --pack 4-per-byte-->  uint8 [out, in//4]

Non-BitLinear tensors (embed_tokens, RMSNorm/SubLN gamma, inv_freq,
causal_mask) are stored as-is.

Output: checkpoints/bitnet_shakespeare_char_ternary.pt
  { name: {"packed": uint8[out,in//4], "scale": float, "shape": (out,in)},  # BitLinear
           <or> tensor }                                                  # everything else

Usage:
  python bitnet/export_ternary.py
  python bitnet/export_ternary.py --checkpoint checkpoints/bitnet_shakespeare_char_best.pt
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch


def weight_quant(w: torch.Tensor):
    """BitNet per-tensor abs-mean ternarization (mirror of model.weight_quant)."""
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    ternary = (w * scale).round().clamp_(-1, 1)  # {-1, 0, 1}
    return ternary, scale


def pack_2bit(code: torch.Tensor) -> torch.Tensor:
    """code: [..., K] uint8 in {0,1,2}, K%4==0 -> uint8 [..., K//4].

    Packs 4 consecutive codes into one byte: code[4i] | code[4i+1]<<2 | ...
    """
    assert code.dtype == torch.uint8
    K = code.shape[-1]
    assert K % 4 == 0, f"K={K} must be divisible by 4 for 2-bit packing"
    c0, c1, c2, c3 = code[..., 0::4], code[..., 1::4], code[..., 2::4], code[..., 3::4]
    packed = (c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)).to(torch.uint8)
    return packed


def unpack_2bit(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [..., K//4] -> int8 [..., K] in {-1, 0, 1}."""
    p = packed.to(torch.int32)
    c0 = p & 0x3
    c1 = (p >> 2) & 0x3
    c2 = (p >> 4) & 0x3
    c3 = (p >> 6) & 0x3
    code = torch.stack([c0, c1, c2, c3], dim=-1).reshape(*packed.shape[:-1], -1)
    return (code - 1).to(torch.int8)  # {0,1,2} -> {-1,0,1}


def is_bitlinear_weight(name: str, keys: set) -> bool:
    """A BitLinear weight has a sibling `<prefix>.norm.weight` (BitLinear holds an RMSNorm)."""
    if not name.endswith(".weight"):
        return False
    prefix = name[: -len(".weight")]
    return f"{prefix}.norm.weight" in keys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/bitnet_shakespeare_char_best.pt")
    p.add_argument("--out", default="checkpoints/bitnet_shakespeare_char_ternary.pt")
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args()

    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    keys = set(sd.keys())
    exported = {}
    n_packed = 0
    n_bytes_packed = 0

    for name, t in sd.items():
        if is_bitlinear_weight(name, keys):
            w = t.to(torch.float32)
            ternary, scale = weight_quant(w)            # {-1,0,1} fp32, scale fp32
            code = (ternary + 1).to(torch.uint8)         # {0,1,2}
            packed = pack_2bit(code)                     # uint8 [out, in//4]
            exported[name] = {
                "packed": packed.cpu(),
                "scale": float(scale),
                "shape": tuple(w.shape),
            }
            n_packed += 1
            n_bytes_packed += packed.numel()
        else:
            exported[name] = t.cpu()
    torch.save(exported, args.out)

    bf16_bytes = sum(t.numel() * 2 for n, t in sd.items() if t.dtype == torch.bfloat16)
    print(f"packed {n_packed} BitLinear weights -> {n_bytes_packed:,} bytes "
          f"({n_bytes_packed/1e6:.2f} MB)")
    print(f"original bf16 weights: {bf16_bytes:,} bytes ({bf16_bytes/1e6:.2f} MB)")
    print(f"compression: {bf16_bytes/max(n_bytes_packed,1):.1f}x")
    print(f"saved: {args.out}")

    if not args.no_verify:
        verify(exported, sd)


def verify(exported, sd):
    """Reconstruct weights from packed and confirm they match weight_quant exactly."""
    keys = set(sd.keys())
    max_diff = 0.0
    for name, t in sd.items():
        if not is_bitlinear_weight(name, keys):
            continue
        w = t.to(torch.float32)
        ternary_ref, scale_ref = weight_quant(w)  # reference
        e = exported[name]
        ternary_rec = unpack_2bit(e["packed"]).to(torch.float32)  # {-1,0,1}
        # reconstructed w_quant = ternary / scale
        diff = (ternary_rec - ternary_ref).abs().max().item()
        max_diff = max(max_diff, diff)
        assert abs(e["scale"] - float(scale_ref)) < 1e-8, f"{name}: scale mismatch"
    print(f"verify: max |ternary_reconstructed - ternary_ref| = {max_diff} "
          f"({'OK' if max_diff == 0 else 'MISMATCH'})")
    assert max_diff == 0, "ternary reconstruction does not match weight_quant"


if __name__ == "__main__":
    main()
