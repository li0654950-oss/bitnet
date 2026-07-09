# bitnet - 基于 shakespeare_char 的 15M 三值权重 BitNet + CIM 异构编译前端

一个约 1500 万参数的 **BitNet b1.58** 风格 Transformer，在 nanoGPT 的
`shakespeare_char` 数据集上从零训练。训练时权重以 **bfloat16** 存储，前向传播
经 `BitLinear` 内部的直通估计器（STE）**三值化为 `{-1, 0, +1}`**。
推理支持将权重**导出为 2bit 打包格式**，由自定义 Triton kernel 直接对打包
三值权重做矩阵乘--无需解包回浮点，存储 8× 压缩。

`cim_compiler/` 是面向 CIM 硬件的异构编译前端：把 BitNet 导出为 FX graph、
划分 CPU/CIM 子图、用 torch-mlir 降级到 MLIR linalg IR。

> 一句话：**不再用浮点。** 权重只有 `[1, 0, -1]`，推理时 2bit 打包；
> 编译前端把图导成 MLIR 给 CIM 后端。

---

## 环境（重点）

### 核心 conda env：`nanogpt-gpu`

| 包 | 版本 | 说明 |
|---|---|---|
| Python | 3.11 | |
| torch | **2.12.0.dev20260407+cu128** | nightly，CUDA 12.8 |
| torch-mlir | **20260403.771** | 2026 dev wheel（nightly），`--no-deps` 装 |
| numpy | <2 | torch 2.x 要求（numpy 2.x 触发 `_ARRAY_API` 警告） |

所有命令默认在 `nanogopt-gpu` env 下运行：
```bash
conda activate nanogopt-gpu
# 或直接用绝对路径
/home/li/anaconda3/envs/nanogpt-gpu/bin/python ...
```

### torch-mlir 安装

torch-mlir **不发 PyPI**，从 GitHub `llvm/torch-mlir-release` 仓库的 `dev-wheels`
release 下载 nightly wheel，`--no-deps` 装（不覆盖 torch 的 cu128）：

```bash
pip install --no-deps \
  https://github.com/llvm/torch-mlir-release/releases/download/dev-wheels/torch_mlir-20260403.771-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

dev wheel 配 torch nightly（2.12），与 `torch 2.12.0.dev20260407` ABI 兼容。

### 为什么用 nightly + dev wheel

torch-mlir 把 BitNet 降级到 linalg IR 必须用 **`fx.export_and_import`**（高层 API）。
其他路径不可行：
- `FxImporter.import_program` + `lower_mlir_module`（手动两步）：卡 `torch.vtensor.literal`
- `torch_mlir.compile`（老 jit.script 路径）：不支持 BitNet 的 attention op

### bitnet 适配 torch-mlir 降级的 4 处改动（数值均等价）

torch-mlir 2026 对部分 op 降级不全，bitnet 代码做了适配（`cim_compiler/export/` 与 `bitnet/model.py`）：

1. **`_LogitsOnly` 包装**（`cim_compiler/export/inference_model.py`，仅 torch-mlir 降级用）：
   `BitNet.forward` 返回 `(logits, None)`，`fx.export_and_import` 不支持 None 输出；
   降级时包装只返回 logits。`export_fx.py` 直接 export BitNet（torch 2.12 的
   `torch.export` 支持 None 输出，不需包装）。
2. **div/mod 解包**（`unpack_2bit_aten`）：`bitwise_and`/`rshift` torch-mlir 降级不全；
   改用 `p%4`、`(p//4)%4` 等（p 非负，数学等价 bitwise）。
3. **float matmul 替代 `_int_mm`**（`BitLinearInference.forward`）：torch-mlir 不降级
   `_int_mm`；int8×{-1,0,1} 在 float32 精确表示（`|acc|≤65536<2²⁴`），数值等价 int32。
4. **seq f32**（`RotaryMHA`）：`torch.arange` 默认 i64，与 `inv_freq`（f32）einsum 后
   cast 失败；显式 `.to(torch.float32)`。

### 已知坑

- **dev wheel 只配 nightly**：torch-mlir 预编译只有 nightly，无 stable 版本，故用 torch nightly。
- **API 选择**：`torch_mlir.compile`（老 jit.script 路径）对 BitNet 不行；必须用
  `torch_mlir.fx.export_and_import`。
- **namespace package 初始化**：torch-mlir 2026 是 namespace（无 `__init__.py`），
  需 `import torch_mlir._mlir_libs` 触发 `_site_initialize`，再手动
  `_torchMlir.register_dialect(ctx)` 注册 torch dialect 给 `FxImporter`。

---

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
- **BitLinear** -- STE 三值化：per-tensor abs-mean 缩放 -> `{-1, 0, +1}`，per-token int8 激活量化，内置 RMSNorm，无 bias。
- **RotaryMHA** -- q/k 上 RoPE，GQA（k/v 跨头组共享），`scaled_dot_product_attention`。
- **ReLUSqFFN** -- `ReLU²` 激活（BitNet 论文）。
- **SubLayerNorm** -- 仅去均值，γ 可学习，无 bias。
- Pre-norm 残差块，全工程无任何 bias。

## 数据

`data/shakespeare_char/` -- nanoGPT 的 tinyshakespeare，字符级（vocab=65）。
复用工程根目录的 `data/input.txt`（与 nanoGPT 源文件逐字节一致）。

```bash
python data/shakespeare_char/prepare.py   # 生成 train.bin, val.bin, meta.pkl
```

## 用法

```bash
# 0. 环境
conda activate nanogpt-gpu   # torch 2.12 dev + torch-mlir 2026 dev wheel

# 1. 准备数据（vocab=65）
python data/shakespeare_char/prepare.py

# 2. 训练（nanoGPT 训练循环 + BitNet 论文 lr/wd 调度，bf16）
#    自动保存 val 最优 -> checkpoints/bitnet_shakespeare_char_best.pt
python bitnet/train_shakespeare_char.py
python bitnet/train_shakespeare_char.py --smoke          # 20 步冒烟测试

# 3. 生成（bf16 STE 推理，默认）
python bitnet/generate_shakespeare_char.py
python bitnet/generate_shakespeare_char.py --prompt "ROMEO:" --max_tokens 500

# 4. 导出 2bit 三值权重（bf16 -> 2bit packed, 8x 压缩, ~3.7 MB）
python bitnet/export_ternary.py

# 5. 三值推理（2bit packed + Triton kernel，无需解包）
python bitnet/generate_shakespeare_char.py --ternary --prompt "ROMEO:" --max_tokens 500

# 6. CIM 编译前端（见下方 cim_compiler）
python cim_compiler/export/export_fx.py           # 导出 FX graph
python cim_compiler/partition/partition.py        # 划分 CPU/CIM
```

预期表现：best val ≈ 1.53，约在 step 1250 达到（笔记本 GPU 约 4 分钟），
约 step 2500 触发早停。

## 三值推理（2bit 打包 + Triton kernel）

训练用 STE（bf16 浮点三态）保证梯度可微；推理可切换到**纯三值路径**：

1. **导出**（`bitnet/export_ternary.py`）：每个 BitLinear 权重 -> `weight_quant`
   三值化 `{-1,0,1}` -> 编码 2bit（`code=ternary+1 ∈ {0,1,2}`）-> 打包 4 个
   code/byte（uint8）。存储 30 MB bf16 -> 3.7 MB（8× 压缩）。
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

## cim_compiler（CIM 异构编译前端）

把 BitNet 导出为 FX graph + 划分 CPU/CIM 子图 + 用 torch-mlir 降级 MLIR，
为 CIM 硬件编译后端提供 IR。对应 `cim_mlp.md` 的 CIM 定点通路。

```
cim_compiler/
├── export/              # BitNet -> FX graph (.pt2) + weight blob (.bin)
│   ├── inference_model.py   # BitLinearInference (2bit 打包权重 + 原生 ATen 定点前向)
│   ├── export_fx.py          # torch.export -> .pt2 (动态 seq len)
│   ├── weight_blob.py        # CIM 权重预加载 blob (自描述二进制)
│   └── verify_export.py       # 端到端验证
├── partition/           # FX graph CPU/CIM 划分
│   ├── classify.py          # CIM 节点标记 (matmul + 权重解包链)
│   ├── partition.py         # 逻辑子图 + 边界张量 (.json)
│   └── verify_partition.py  # 验证
└── (lowering 待实现)    # MLIR -> CIM 指令
```

### 流程

1. **export**：`BitLinearInference`（2bit 打包三值权重 + 原生 ATen 定点前向：
   `norm -> per-token int8 量化 -> 2bit 解包 -> float matmul -> rescale`）
   -> `torch.export` -> `.pt2`（动态 seq len [1..256]）+ 旁路 `.bin`（CIM preload）。
2. **partition**：识别 CIM 块（`matmul` + 权重解包链）+ 边界张量（CPU->CIM `x_in`、
   CIM->CPU `acc`）-> `.json` 逻辑子图（不切分可执行图，保留 export 调用 magic）。
3. **torch-mlir 降级**：`fx.export_and_import` -> linalg-on-tensors IR
   （37 `linalg.matmul`，CPU/CIM 在 IR 里自然分离）。

### 用法

```bash
# export
python cim_compiler/export/export_fx.py          # 产出 .pt2 + .bin
python cim_compiler/export/verify_export.py       # 验证

# partition
python cim_compiler/partition/partition.py         # 产出 .json
python cim_compiler/partition/verify_partition.py  # 验证

# torch-mlir 降级 (验证降级通路)
python -c "
import torch_mlir._mlir_libs, torch_mlir.ir as ir, torch_mlir._mlir_libs._torchMlir as t
import torch_mlir.fx as fx, torch_mlir.compiler_utils as cu, torch, sys
sys.path.insert(0,'cim_compiler/export'); sys.path.insert(0,'bitnet')
from inference_model import build_inference_model, _LogitsOnly
from data_char import get_meta
m = _LogitsOnly(build_inference_model('checkpoints/bitnet_shakespeare_char_ternary.pt',
           vocab_size=get_meta()['vocab_size'])).eval()
mlir = fx.export_and_import(m, torch.zeros(1,4,dtype=torch.long),
           output_type=cu.OutputType.LINALG_ON_TENSORS, func_name='bitnet')
print(str(mlir)[:500])
"
```

### 产物

| 文件 | 内容 |
|---|---|
| `checkpoints/bitnet_ternary.pt2` | FX graph（37 matmul，动态 seq len）|
| `checkpoints/bitnet_ternary_weights.bin` | CIM preload 权重 blob（37 层 2bit 打包）|
| `checkpoints/bitnet_ternary_partition.json` | CPU/CIM 划分（节点标注 + 边界）|
| `checkpoints/bitnet_linalg.mlir` | linalg IR（37 `linalg.matmul`，torch-mlir 降级产物）|

### CIM 块边界（对应 cim_mlp.md）

CIM 节点 = `matmul` + 权重解包链（`div/mod` 取 2bit code -> `where` 补码 -> int8）。
对应 CIM Macro：2bit 节点 + 宏内解包 + 累加。CPU 节点 = norm / 激活量化 / rescale /
attention / embedding。边界：CPU->CIM = `x_in`（激活），CIM->CPU = `acc`（matmul 输出）。

> **注**：为兼容 torch-mlir 降级，CIM 块的 `matmul` 是 float（替代 `_int_mm`）。
> int8×{-1,0,1} 在 float32 精确表示，数值等价 int32；CIM compiler 后端识别
> `linalg.matmul` 后可重新引入 int8 量化语义，或改用 custom op 保留 CIM 语义。

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
cim_compiler/                   # CIM 异构编译前端
  export/                       # BitNet -> FX graph + weight blob
  partition/                    # CPU/CIM 逻辑子图划分
cim_model/                      # CIM 定点仿真模型（验证 cim_mlp.md 计算通路）
cim_mlp.md                      # CIM 协处理器架构规范
data/
  input.txt                     # tinyshakespeare 源文件（prepare.py 复用）
  shakespeare_char/             # prepare.py + train.bin/val.bin/meta.pkl
tests/                          # 层 + 量化单元测试
checkpoints/                    # 模型 checkpoint + 导出产物（.pt/.pt2/.bin/.json/.mlir）
```

## 参考

- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)（BitNet b1.58）
- [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)
- [nanoGPT](https://github.com/karpathy/nanoGPT) -- 训练循环 / 数据布局
- [torch-mlir](https://github.com/llvm/torch-mlir) -- PyTorch -> MLIR
