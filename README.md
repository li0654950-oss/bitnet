# bitnet — 基于 shakespeare_char 的 15M 三值权重 BitNet

一个约 1500 万参数的 **BitNet b1.58** 风格 Transformer，在 nanoGPT 的
`shakespeare_char` 数据集上从零训练。训练时权重以 **bfloat16** 存储，前向传播
经 `BitLinear` 内部的直通估计器（STE）**三值化为 `{-1, 0, +1}`**。
推理支持将权重**导出为 2bit 打包格式**，由自定义 Triton kernel 直接对打包
三值权重做矩阵乘——无需解包回浮点，存储 8× 压缩。

> 一句话：**不再用浮点。** 权重只有 `[1, 0, -1]`，推理时 2bit 打包。

## 模型

| 项 | 值 |
|---|---|
| 参数量 | **15,041,280（15.04M）** |
| `d_model` | 512 |
| `n_layer` | 6 |
| `n_head` / `n_kv_head` | 8 / 4（GQA，head_dim=64） |
| `ffn_dim` | 1664 |
| `block_size` | 256 |
| 词表 | 65（字符级） |
| 权重 dtype | bfloat16（训练与 checkpoint 存储均为 bf16） |
| 前向有效值 | `{-1, 0, +1}`（STE 三值化，per-tensor abs-mean 缩放） |

架构（见 `bitnet/model.py`）：
- **BitLinear** —— STE 三值化：per-tensor abs-mean 缩放 → `{-1, 0, +1}`，per-token int8 激活量化，内置 RMSNorm，无 bias。
- **RotaryMHA** —— q/k 上 RoPE，GQA（k/v 跨头组共享），`scaled_dot_product_attention`。
- **ReLUSqFFN** —— `ReLU²` 激活（BitNet 论文）。
- **SubLayerNorm** —— 仅去均值，γ 可学习，无 bias。
- Pre-norm 残差块，全工程无任何 bias。

## 数据

`data/shakespeare_char/` —— nanoGPT 的 tinyshakespeare，字符级（vocab=65）。
复用工程根目录的 `data/input.txt`（与 nanoGPT 源文件逐字节一致）。

```bash
python data/shakespeare_char/prepare.py   # 生成 train.bin, val.bin, meta.pkl
```

## 用法

```bash
# 0. 环境（torch + numpy + pytest）
source setup.sh

# 1. 准备数据（vocab=65）
python data/shakespeare_char/prepare.py

# 2. 训练（nanoGPT 训练循环 + BitNet 论文 lr/wd 调度，bf16）
#    自动保存 val 最优 -> checkpoints/bitnet_shakespeare_char_best.pt
#    val 停滞时早停（默认 patience=5 次 eval）
python bitnet/train_shakespeare_char.py
python bitnet/train_shakespeare_char.py --smoke          # 20 步冒烟测试

# 3. 生成（bf16 STE 推理，默认）
python bitnet/generate_shakespeare_char.py
python bitnet/generate_shakespeare_char.py --prompt "ROMEO:" --max_tokens 500
python bitnet/generate_shakespeare_char.py --temperature 0.7 --top_k 40 --seed 0

# 4. 导出 2bit 三值权重（bf16 -> 2bit packed, 8x 压缩, ~3.7 MB）
python bitnet/export_ternary.py

# 5. 三值推理（2bit packed + Triton kernel，无需解包）
python bitnet/generate_shakespeare_char.py --ternary --prompt "ROMEO:" --max_tokens 500
```

预期表现：best val ≈ 1.53，约在 step 1250 达到（笔记本 GPU 约 4 分钟），
约 step 2500 触发早停。

## 三值推理（2bit 打包 + Triton kernel）

训练用 STE（bf16 浮点三态）保证梯度可微；推理可切换到**纯三值路径**：

1. **导出**（`bitnet/export_ternary.py`）：每个 BitLinear 权重 → `weight_quant`
   三值化 `{-1,0,1}` → 编码 2bit（`code=ternary+1 ∈ {0,1,2}`）→ 打包 4 个
   code/byte（uint8）。存储 30 MB bf16 → 3.7 MB（8× 压缩）。
2. **kernel**（`bitnet/ternary_kernel.py`）：Triton kernel 内 **fused 解包 2bit**
   （`(byte>>shift)&0x3`）+ bf16 tensor-core dot + rescale，无需解包成大 tensor。
   附 CPU fallback（解包 + `F.linear`）。
3. **推理路径**（`BitLinear.set_inference`）：`model.py` 的 BitLinear 加推理分支，
   训练路径完全不动；`generate --ternary` 加载 packed 权重并切换各 BitLinear 到
   kernel 路径。

PyTorch 原生 int8 matmul 不支持（`addmm_cuda not implemented for Int`），故 2bit
三值 matmul 必须自定义 kernel。

**数值验证**：ternary 推理 vs bf16 STE，logits argmax 100% 一致，max diff ≈ 0.25
（logits scale ~13.5，~1.9%）。

## 测试

```bash
PYTHONPATH=$PWD pytest -q          # BitLinear/RotaryMHA/SubLN 层 + 三值量化
```

## 工程结构

```
bitnet/
  model.py                      # BitNet, BitLinear（STE 训练 + 三值推理分支）, RotaryMHA, ReLUSqFFN
  data_char.py                  # 字符级 get_batch + CharTokenizer（无 HF 依赖）
  train_shakespeare_char.py     # 训练（best 保存 + 早停）
  generate_shakespeare_char.py  # 推理（bf16 STE / --ternary 2bit kernel）
  export_ternary.py             # 导出 2bit 打包三值权重（8x 压缩）
  ternary_kernel.py             # Triton 2bit 三值 matmul kernel + CPU fallback
data/
  input.txt                     # tinyshakespeare 源文件（prepare.py 复用）
  shakespeare_char/             # prepare.py + train.bin/val.bin/meta.pkl
tests/                          # 层 + 量化单元测试
```

## 参考

- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)（BitNet b1.58）
- [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)
- [nanoGPT](https://github.com/karpathy/nanoGPT) —— 训练循环 / 数据布局
