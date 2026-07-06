#!/usr/bin/env python3
"""
Export BitNet weights as 2-bit packed ternary {-1, 0, +1} (two's complement encoding).

Encoding (2-bit two's complement, code values are semantically {-1,0,1}):
  -1 -> 0b11 (3),  0 -> 0b00 (0),  +1 -> 0b01 (1)   [0b10 (2) unused]
Packed 4 codes/byte (uint8). Storage identical to {0,1,2} offset encoding,
but the code is the literal two's-complement representation of {-1,0,1}.

For every BitLinear weight: bf16 -> weight_quant -> ternary {-1,0,1} -> 2bit packed + scale_w.

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


def pack_2bit(ternary: torch.Tensor) -> torch.Tensor:
    """ternary: [..., K] int8 {-1,0,1}, K%4==0 -> uint8 [..., K//4].

    2-bit two's complement: -1 -> 0b11 (3), 0 -> 0b00 (0), +1 -> 0b01 (1).
    """
    assert ternary.dtype == torch.int8
    K = ternary.shape[-1]
    assert K % 4 == 0, f"K={K} must be divisible by 4"
    code = (ternary.to(torch.int32) & 0x3).to(torch.uint8)  # -1->3, 0->0, 1->1
    c0, c1, c2, c3 = code[..., 0::4], code[..., 1::4], code[..., 2::4], code[..., 3::4]
    return (c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)).to(torch.uint8)


def unpack_2bit(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [..., K//4] -> int8 [..., K] {-1,0,1} (two's complement decode)."""
    p = packed.to(torch.int32)
    c0, c1, c2, c3 = p & 0x3, (p >> 2) & 0x3, (p >> 4) & 0x3, (p >> 6) & 0x3
    code = torch.stack([c0, c1, c2, c3], dim=-1).reshape(*packed.shape[:-1], -1)
    # 2-bit two's complement: 3 -> -1, 0 -> 0, 1 -> 1 (2 unused)
    return torch.where(code >= 2, code - 4, code).to(torch.int8)


def is_bitlinear_weight(name: str, keys: set) -> bool:
    """A BitLinear weight has a sibling `<prefix>.norm.weight`."""
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
    n_bytes = 0

    for name, t in sd.items():
        if is_bitlinear_weight(name, keys):
            w = t.to(torch.float32)
            ternary, scale = weight_quant(w)            # {-1,0,1} fp32, scale fp32
            ternary_int8 = ternary.to(torch.int8)        # {-1,0,1} int8
            packed = pack_2bit(ternary_int8)             # 2-bit two's complement
            exported[name] = {
                "packed": packed.cpu(),
                "scale": float(scale),
                "shape": tuple(w.shape),
            }
            n_packed += 1
            n_bytes += packed.numel()
        else:
            exported[name] = t.cpu()
    torch.save(exported, args.out)

    bf16_bytes = sum(t.numel() * 2 for _, t in sd.items() if t.dtype == torch.bfloat16)
    print(f"packed {n_packed} BitLinear weights -> {n_bytes:,} bytes "
          f"({n_bytes/1e6:.2f} MB, 2-bit two's complement {-1,0,1})")
    print(f"original bf16: {bf16_bytes:,} bytes ({bf16_bytes/1e6:.2f} MB)  "
          f"compression: {bf16_bytes/max(n_bytes,1):.1f}x")
    print(f"saved: {args.out}")

    if not args.no_verify:
        verify(exported, sd)


def verify(exported, sd):
    """Confirm unpacked ternary matches weight_quant exactly."""
    keys = set(sd.keys())
    max_diff = 0.0
    for name, t in sd.items():
        if not is_bitlinear_weight(name, keys):
            continue
        w = t.to(torch.float32)
        ternary_ref, scale_ref = weight_quant(w)
        e = exported[name]
        ternary_rec = unpack_2bit(e["packed"]).to(torch.float32)  # {-1,0,1}
        diff = (ternary_rec - ternary_ref).abs().max().item()
        max_diff = max(max_diff, diff)
        assert abs(e["scale"] - float(scale_ref)) < 1e-8, f"{name}: scale mismatch"
        assert set(unpack_2bit(e["packed"]).unique().tolist()).issubset({-1, 0, 1}), \
            f"{name}: non-ternary values found"
    print(f"verify: max |ternary_rec - ternary_ref| = {max_diff} "
          f"({'OK' if max_diff == 0 else 'MISMATCH'}), values ⊆ {{-1,0,1}} ✓")
    assert max_diff == 0


if __name__ == "__main__":
    main()
