# bitnet — 15M ternary-weight BitNet on shakespeare_char

A ~15M-parameter **BitNet b1.58** style transformer trained from scratch on
nanoGPT's `shakespeare_char` dataset. Weights are stored as float32 but
**ternarized to `{-1, 0, +1}` on every forward pass** via a Straight-Through
Estimator (STE) inside `BitLinear` — no separate quantization/export step.

> tldr; **No more floats.** Just weights in `[1, 0, -1]`.

## Model

| | value |
|---|---|
| params | **15,041,280 (15.04M)** |
| `d_model` | 512 |
| `n_layer` | 6 |
| `n_head` / `n_kv_head` | 8 / 4 (GQA, head_dim=64) |
| `ffn_dim` | 1664 |
| `block_size` | 256 |
| vocab | 65 (char-level) |
| dtype | bfloat16 (train), float32 (stored weights) |

Architecture (in `bitnet/model.py`):
- **BitLinear** — STE ternarization: per-tensor abs-mean scale → `{-1, 0, +1}`, per-token int8 activation quant, built-in RMSNorm, no bias.
- **RotaryMHA** — RoPE on q/k, GQA (k/v shared across head groups), `scaled_dot_product_attention`.
- **ReLUSqFFN** — `ReLU²` activation (BitNet paper).
- **SubLayerNorm** — mean-centered, γ-only, no bias.
- Pre-norm residual blocks, no biases anywhere.

## Data

`data/shakespeare_char/` — nanoGPT's tinyshakespeare, char-level (vocab=65).
Reuses the repo's `data/input.txt` (byte-identical to nanoGPT's source).

```bash
python data/shakespeare_char/prepare.py   # -> train.bin, val.bin, meta.pkl
```

## Usage

```bash
# 0. env (torch + numpy + pytest)
source setup.sh

# 1. prepare data (vocab=65)
python data/shakespeare_char/prepare.py

# 2. train (nanoGPT loop + BitNet paper lr/wd schedule, bf16)
#    auto-saves val-optimal -> checkpoints/bitnet_shakespeare_char_best.pt
#    early-stops when val plateaus (patience=5 evals)
python bitnet/train_shakespeare_char.py
python bitnet/train_shakespeare_char.py --smoke          # 20-step sanity check

# 3. generate
python bitnet/generate_shakespeare_char.py
python bitnet/generate_shakespeare_char.py --prompt "ROMEO:" --max_tokens 500
python bitnet/generate_shakespeare_char.py --temperature 0.7 --top_k 40 --seed 0
```

Expected: best val ≈ 1.53 around step 1250 (~4 min on a laptop GPU), early-stop
around step 2500.

## Tests

```bash
PYTHONPATH=$PWD pytest -q          # BitLinear/RotaryMHA/SubLN layers + ternary quant
```

## Project layout

```
bitnet/
  model.py                      # BitNet, BitLinear (STE ternary), RotaryMHA, ReLUSqFFN
  data_char.py                  # char-level get_batch + CharTokenizer (no HF deps)
  train_shakespeare_char.py     # train (best-save + early stopping)
  generate_shakespeare_char.py  # inference (temperature / top-k / prompt)
data/
  input.txt                     # tinyshakespeare source (reused by prepare.py)
  shakespeare_char/             # prepare.py + train.bin/val.bin/meta.pkl
tests/                          # layer + quantization unit tests
SPEC.md                         # BitNet paper architecture notes
```

## References

- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764) (BitNet b1.58)
- [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)
- [nanoGPT](https://github.com/karpathy/nanoGPT) — training loop / data layout
