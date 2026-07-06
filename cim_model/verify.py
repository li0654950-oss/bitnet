"""CIM 定点计算模型验证脚本。

三级验证:
  1. Macro 级:    64×64 int8×ternary → int32 部分和 (精确整数)
  2. BitLinear 级: tile 切分 + int32 累加 + rescale (vs fp32 参考路径)
  3. 模型级:      CIM 模型推理 vs bf16 STE 参考模型 (argmax 一致率)

运行:
  /home/li/anaconda3/envs/nanogpt-gpu/bin/python cim_model/verify.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bitnet"))

from export_ternary import pack_2bit
from cim_model.macro import macro_matmul, MACRO_SIZE
from cim_model.accumulator import Accumulator
from cim_model.bitlinear_cim import bitlinear_cim, quantize_activation


def _pack(ternary_int8):
    """int8 三值 [..., K] → uint8 2bit 补码打包 [..., K//4] (复用 export_ternary.pack_2bit, 节点态)。"""
    return pack_2bit(torch.from_numpy(np.ascontiguousarray(ternary_int8))).numpy()

CHECKPOINT_BEST = os.path.join(ROOT, "checkpoints", "bitnet_shakespeare_char_best.pt")
CHECKPOINT_TERNARY = os.path.join(ROOT, "checkpoints", "bitnet_shakespeare_char_ternary.pt")
VAL_BIN = os.path.join(ROOT, "data", "shakespeare_char", "val.bin")

# 模型配置 (训练时)
CFG = dict(d_model=512, block_size=256, n_layer=6, n_head=8, n_kv_head=4, ffn_dim=1664)


# ───────────────────────── 1. Macro 级 ─────────────────────────

def test_macro_exact():
    """Macro: int8×ternary → int32, 与 int32 参考 matmul 精确一致 (diff=0)。"""
    rng = np.random.default_rng(0)
    B = 7
    x = rng.integers(-128, 128, (B, MACRO_SIZE), dtype=np.int8)
    W = rng.integers(-1, 2, (MACRO_SIZE, MACRO_SIZE), dtype=np.int8)  # {-1,0,1}
    W_packed = _pack(W)  # uint8 [64, 16] 2bit 补码 (节点态)
    y = macro_matmul(x, W_packed)  # int32 [B, 64]
    ref = x.astype(np.int32) @ W.astype(np.int32).T
    diff = int(np.abs(y - ref).max())
    assert y.dtype == np.int32, f"dtype {y.dtype} != int32"
    assert diff == 0, f"macro diff={diff} (应精确整数)"
    print(f"  [1a] macro int8×ternary→int32: max diff={diff} (int32 精确) ✓")


def test_macro_degenerate():
    """三值退化: W=+1→+x, W=-1→-x, W=0→0。"""
    x = np.array([[1, 2, 3, -4] + [0] * 60], dtype=np.int8)  # [1, 64]
    s = int(x.sum())
    y_plus = macro_matmul(x, _pack(np.ones((64, 64), dtype=np.int8)))[0, 0]
    y_minus = macro_matmul(x, _pack(-np.ones((64, 64), dtype=np.int8)))[0, 0]
    y_zero = macro_matmul(x, _pack(np.zeros((64, 64), dtype=np.int8)))[0, 0]
    assert y_plus == s and y_minus == -s and y_zero == 0, \
        f"退化失败: +{y_plus}/{s} -{y_minus}/{-s} 0={y_zero}"
    print(f"  [1b] macro 三值退化 (+1→{y_plus}, -1→{y_minus}, 0→{y_zero}) ✓")


# ───────────────────────── 2. BitLinear 级 ─────────────────────────

def test_accumulator_rmw():
    """累加区: K 维多 tile int32 RMW 累加 == 直接大 int32 matmul (diff=0)。"""
    rng = np.random.default_rng(1)
    M, N, K = 4, 64, 192  # K=3 个 tile
    x = rng.integers(-128, 128, (M, K), dtype=np.int8)
    W = rng.integers(-1, 2, (N, K), dtype=np.int8)
    W_packed = _pack(W)  # uint8 [64, 48] 2bit 补码 (节点态)
    acc = np.zeros((M, N), dtype=np.int32)
    for kb in range(K // MACRO_SIZE):
        x_slice = x[:, kb * 64:(kb + 1) * 64]
        w_tile = W_packed[:, kb * 16:(kb + 1) * 16]  # [64, 16] 2bit 补码 tile
        psum = macro_matmul(x_slice, w_tile)
        acc += psum  # int32 RMW
    ref = x.astype(np.int32) @ W.astype(np.int32).T
    diff = int(np.abs(acc - ref).max())
    assert diff == 0, f"累加 diff={diff}"
    print(f"  [2a] 累加区 int32 RMW (3 tile): max diff={diff} ✓")


def test_bitlinear_vs_fp32_ref():
    """BitLinear CIM 定点通路 vs fp32 参考路径 (量化 + fp32 matmul + rescale)。"""
    rng = np.random.default_rng(2)
    N, K, M = 512, 512, 7
    ternary = rng.integers(-1, 2, (N, K), dtype=np.int8)  # {-1,0,1}
    ternary_packed = _pack(ternary)  # uint8 [N, K//4] 2bit 补码 (节点态)
    scale_w = 20.0
    x = rng.standard_normal((M, K)).astype(np.float32)

    y_cim = bitlinear_cim(x, ternary_packed, scale_w)  # CIM 定点 (2bit 节点)

    # fp32 参考: 同量化 + fp32 matmul + rescale (int8×ternary 是精确整数, fp32/int32 等价)
    x_int8, scale_x = quantize_activation(x)
    ref = (x_int8.astype(np.float32) @ ternary.astype(np.float32).T) / (scale_x * scale_w)

    diff = float(np.abs(y_cim - ref).max())
    assert y_cim.shape == (M, N), f"shape {y_cim.shape}"
    assert diff < 1e-3, f"bitlinear diff={diff}"
    print(f"  [2b] bitlinear CIM vs fp32 ref: shape={y_cim.shape} max diff={diff:.2e} ✓")


def test_bitlinear_padding():
    """lm_head (65, 512): N=65 非 64 倍数, pad→128, 输出截断回 65。"""
    rng = np.random.default_rng(3)
    N, K, M = 65, 512, 7
    ternary = rng.integers(-1, 2, (N, K), dtype=np.int8)
    ternary_packed = _pack(ternary)  # 2bit 补码 (节点态)
    scale_w = 20.0
    x = rng.standard_normal((M, K)).astype(np.float32)
    y = bitlinear_cim(x, ternary_packed, scale_w)
    assert y.shape == (M, N), f"padding shape {y.shape} != {(M, N)}"

    # 与未 padding 的 fp32 参考对比 (前 65 行)
    x_int8, scale_x = quantize_activation(x)
    ref = (x_int8.astype(np.float32) @ ternary.astype(np.float32).T) / (scale_x * scale_w)
    diff = float(np.abs(y - ref).max())
    assert diff < 1e-3, f"padding diff={diff}"
    print(f"  [2c] bitlinear padding (N=65→128): shape={y.shape} max diff={diff:.2e} ✓")


# ───────────────────────── 3. 模型级 ─────────────────────────

def _val_idx(seq_len=256):
    """取 val.bin 开头 seq_len token 作为推理输入 (真实数据)。"""
    val = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")
    arr = np.array(val[:seq_len], dtype=np.int64)[None, :]  # [1, seq_len]
    return torch.from_numpy(arr)


def test_model_forward():
    """CIM 模型 (fp32 三值通路) vs bf16 STE 参考模型: logits diff + argmax 一致率。"""
    from model import BitNet
    from data_char import get_meta
    from cim_model.model_cim import patch_bitlinear, unpatch_bitlinear, load_ternary_into_model

    device = "cpu"
    vocab = get_meta()["vocab_size"]
    idx = _val_idx(256).to(device)

    # ── 参考模型: bf16 原权重 + STE 伪量化 (patch 之前 forward) ──
    ref_model = BitNet(vocab, **CFG).to(device, dtype=torch.bfloat16)
    sd = torch.load(CHECKPOINT_BEST, map_location=device, weights_only=True)
    ref_model.load_state_dict(sd)
    ref_model.eval()
    with torch.no_grad():
        ref_logits, _ = ref_model(idx)  # bf16 STE 路径
    ref_logits = ref_logits.float()

    # ── CIM 模型: patch BitLinear.forward = CIM 定点, 加载三值权重 ──
    orig_fwd = patch_bitlinear()
    try:
        cim_model = BitNet(vocab, **CFG).to(device, dtype=torch.bfloat16)
        n_ternary = load_ternary_into_model(cim_model, CHECKPOINT_TERNARY, device)
        cim_model.eval()
        with torch.no_grad():
            cim_logits, _ = cim_model(idx)  # CIM 定点路径
        cim_logits = cim_logits.float()
    finally:
        unpatch_bitlinear(orig_fwd)

    # ── 对比 ──
    diff = (ref_logits - cim_logits).abs().max().item()
    ref_arg = ref_logits.argmax(-1)
    cim_arg = cim_logits.argmax(-1)
    acc = (ref_arg == cim_arg).float().mean().item()
    # 逐 token argmax 一致率 (整段 256)
    tok_acc = (ref_arg[0] == cim_arg[0]).float().mean().item()

    print(f"  [3a] 模型级: n_ternary={n_ternary} BitLinear")
    print(f"       logits max diff = {diff:.4f}")
    print(f"       argmax 一致率   = {acc:.4f}  (逐 token {tok_acc:.4f})")
    assert acc > 0.7, f"argmax 一致率过低 {acc:.4f}"
    print(f"  [3a] CIM 模型推理 argmax 一致率 {acc:.4f} > 0.7 ✓")


def test_model_generate():
    """CIM 模型生成文本 (固定 seed), 直观验证可跑通。"""
    from model import BitNet
    from data_char import get_meta, CharTokenizer
    from cim_model.model_cim import patch_bitlinear, unpatch_bitlinear, load_ternary_into_model

    device = "cpu"
    meta = get_meta()
    tok = CharTokenizer(meta)
    vocab = meta["vocab_size"]

    orig_fwd = patch_bitlinear()
    try:
        model = BitNet(vocab, **CFG).to(device, dtype=torch.bfloat16)
        load_ternary_into_model(model, CHECKPOINT_TERNARY, device)
        model.eval()
        torch.manual_seed(0)
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        print("  [3b] CIM 生成 (60 token): ", end="", flush=True)
        with torch.no_grad():
            for _ in range(60):
                idx_cond = idx[:, -CFG["block_size"]:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :].float()
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, idx_next], dim=1)
                print(tok.decode([idx_next.item()]), end="", flush=True)
        print()
    finally:
        unpatch_bitlinear(orig_fwd)
    print("  [3b] CIM 生成完成 ✓")


# ───────────────────────── main ─────────────────────────

def main():
    print("=" * 60)
    print("CIM 定点计算模型验证 (cim_mlp.md 数据通路)")
    print("=" * 60)

    print("\n[1] Macro 级 (64×64 int8×ternary → int32)")
    test_macro_exact()
    test_macro_degenerate()

    print("\n[2] BitLinear 级 (tile + int32 累加 + rescale)")
    test_accumulator_rmw()
    test_bitlinear_vs_fp32_ref()
    test_bitlinear_padding()

    print("\n[3] 模型级 (CIM 推理 vs bf16 STE 参考)")
    test_model_forward()
    test_model_generate()

    print("\n" + "=" * 60)
    print("全部验证通过 ✓")
    print("CIM 定点数据通路 (Macro int32 部分和 → 累加区 int32 acc → CPU rescale fp32)")
    print("数值正确, 端到端推理 argmax 与参考一致。")
    print("=" * 60)


if __name__ == "__main__":
    main()
