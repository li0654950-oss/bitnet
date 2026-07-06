#!/usr/bin/env python3
"""
2-bit packed ternary weight matmul kernel.

  ternary_linear(x_int8, w_packed, scale_x, scale_w) -> float

  x_int8   : [..., K] int8         (per-token int8 activation)
  w_packed : [N, K//4] uint8       (2-bit packed ternary {-1,0,1})
  scale_x  : [..., 1] float32      (per-token activation scale)
  scale_w  : float                 (per-tensor weight scale)
  out      : [..., N] float32

Math: out[m,n] = sum_k x_int8[m,k] * ternary[n,k] / (scale_x[m] * scale_w)

Triton kernel fuses: 2-bit unpack + bf16 matmul (tensor core) + rescale.
CPU fallback (unpack + F.linear) for testing / non-CUDA.

Self-test: python bitnet/ternary_kernel.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F

from export_ternary import unpack_2bit

HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    pass


# ───────────────────────── CPU fallback ─────────────────────────

def ternary_linear_cpu(x_int8, w_packed, scale_x, scale_w):
    """Reference: unpack to float, then F.linear. Numerically authoritative."""
    lead = x_int8.shape[:-1]
    K = x_int8.shape[-1]
    N = w_packed.shape[0]
    x = x_int8.reshape(-1, K).to(torch.float32)          # [M, K]
    ternary = unpack_2bit(w_packed).to(torch.float32)    # [N, K] {-1,0,1}
    acc = x @ ternary.T                                  # [M, N]
    sx = scale_x.reshape(-1, 1).to(torch.float32)        # [M, 1]
    return (acc / (sx * scale_w)).reshape(*lead, N)


# ───────────────────────── Triton kernel ─────────────────────────

if HAS_TRITON:

    @triton.jit
    def _ternary_mm_kernel(
        x_ptr, w_ptr, out_ptr, sx_ptr, sw,
        M, N, K, K4,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        grid_n = tl.cdiv(N, BN)
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        rm = pid_m * BM + tl.arange(0, BM)        # [BM]
        rn = pid_n * BN + tl.arange(0, BN)        # [BN]
        rk = tl.arange(0, BK)                     # [BK]

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            k = k0 + rk                            # [BK]
            # load x [BM, BK] int8 -> bf16 (int8 values are exact in bf16)
            x_ptrs = x_ptr + rm[:, None] * K + k[None, :]
            x = tl.load(x_ptrs, mask=(rm[:, None] < M) & (k[None, :] < K), other=0)
            x = x.to(tl.bfloat16)

            # load w_packed [BN, BK//4] uint8, unpack 2-bit -> ternary bf16
            kp = k // 4                             # [BK]
            w_ptrs = w_ptr + rn[:, None] * K4 + kp[None, :]
            wb = tl.load(w_ptrs, mask=(rn[:, None] < N) & (kp[None, :] < K4), other=0)
            shift = (k % 4) * 2                     # [BK]
            code = ((wb.to(tl.int32) >> shift[None, :]) & 0x3)   # [BN, BK] in {0,1,3} (2-bit two's complement)
            ternary = tl.where(code >= 2, code - 4, code).to(tl.bfloat16)  # 3->-1, 0->0, 1->1

            # acc += x @ ternary.T   -> [BM, BN]
            acc += tl.dot(x, tl.trans(ternary))

        sx = tl.load(sx_ptr + rm, mask=rm < M, other=1.0)  # [BM]
        out = acc / (sx[:, None] * sw)
        out_ptrs = out_ptr + rm[:, None] * N + rn[None, :]
        tl.store(out_ptrs, out, mask=(rm[:, None] < M) & (rn[None, :] < N))


def ternary_linear(x_int8, w_packed, scale_x, scale_w, use_triton=True):
    """Dispatch to Triton (CUDA) or CPU fallback."""
    lead = x_int8.shape[:-1]
    K = x_int8.shape[-1]
    N = w_packed.shape[0]
    assert K % 4 == 0, f"K={K} must be divisible by 4"
    assert w_packed.shape == (N, K // 4), f"bad packed shape {w_packed.shape}"

    if not (use_triton and HAS_TRITON and x_int8.is_cuda and w_packed.is_cuda):
        return ternary_linear_cpu(x_int8, w_packed, scale_x, scale_w)

    x = x_int8.reshape(-1, K).contiguous()          # [M, K]
    sx = scale_x.reshape(-1).to(torch.float32).contiguous()  # [M]
    M = x.shape[0]
    out = torch.empty(M, N, dtype=torch.float32, device=x.device)

    BM, BN, BK = 16, 64, 128
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _ternary_mm_kernel[grid](
        x, w_packed, out, sx, float(scale_w),
        M, N, K, K // 4, BM, BN, BK,
    )
    return out.reshape(*lead, N)


# ───────────────────────── self-test ─────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev} | triton: {HAS_TRITON}")

    # random ternary weight + packed, random int8 activation
    for (N, K) in [(512, 512), (65, 512), (1664, 512), (512, 1664), (256, 512)]:
        ternary = torch.randint(-1, 2, (N, K), dtype=torch.int8)  # {-1,0,1}
        from export_ternary import pack_2bit
        packed = pack_2bit(ternary)                     # [N, K//4] (two's complement)
        scale_w = 1.0 / 0.05

        M = 7
        x_int8 = torch.randint(-100, 100, (M, K), dtype=torch.int8)
        sx = torch.rand(M, 1) + 0.5                      # per-token scales

        if dev == "cuda":
            x_d = x_int8.cuda(); p_d = packed.cuda(); sx_d = sx.cuda()
            out_t = ternary_linear(x_d, p_d, sx_d, scale_w, use_triton=True)
            out_c = ternary_linear_cpu(x_d, p_d, sx_d, scale_w)
            diff = (out_t - out_c).abs().max().item()
            ok = diff < 1e-2
            print(f"(N={N:4d},K={K:4d}) triton vs cpu: max diff {diff:.2e} {'OK' if ok else 'FAIL'}")
            assert ok
        else:
            out_c = ternary_linear_cpu(x_int8, packed, sx, scale_w)
            print(f"(N={N:4d},K={K:4d}) cpu only: out {tuple(out_c.shape)} OK")

    print("self-test passed")
