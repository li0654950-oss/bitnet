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

# 6. CIM 编译前端 + 系统仿真（见下方 cim_compiler）
python cim_compiler/pipeline.py                    # 一键流水线 (9 步: export->...->JIT 验证 + AOT 构建)
python cim_compiler/lowering/run_sim_text.py --prompt "ROMEO:" --num-tokens 60  # JIT 文本生成仿真
./cim_compiler/lowering/aot/run_aot.sh --prompt "ROMEO:" --n 60                 # AOT 系统仿真
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

## cim_compiler（CIM 异构编译前端 + 系统仿真）

把 BitNet 导出 FX graph -> 划分 CPU/CIM -> torch-mlir 降级 MLIR -> CIM 指令流 ->
**系统仿真**（JIT / AOT 两种模式）。对应 `cim_mlp.md` 的 CIM 定点通路、`sys_sim.md`
的系统仿真运行时。

```
cim_compiler/
├── export/              # BitNet -> FX graph (.pt2) + weight blob (.bin)
│   ├── inference_model.py   # BitLinearInference (2bit 打包 + cim.matmul custom op 定点前向)
│   ├── export_fx.py          # torch.export -> .pt2 (动态 seq len)
│   ├── cim_op.py             # cim::matmul custom op 注册 (CPU/CIM 在 IR 天然分离)
│   └── weight_blob.py        # CIM 权重预加载 blob (自描述二进制)
├── partition/           # FX graph CPU/CIM 划分
│   ├── classify.py          # CIM 节点标记 (cim.matmul op)
│   └── partition.py         # 逻辑子图 + 边界张量 (.json)
├── ir/                  # torch-mlir 降级 (MLIR linalg IR)
├── cimres/              # CIM 资源映射 + 指令流 + 仿真器
│   ├── lower_to_cimres.py / place.py / emit_instr.py  # C1/C2/C3 (IR->指令流)
│   └── hw_simulator.py   # cycle 级纯硬件仿真器 (HwCimSimulator, Macro 并行时序)
├── lowering/            # MLIR -> LLVM + 系统仿真驱动
│   ├── cim_stub.c          # 固定硬件驱动 (MMIO, 单一 cim_launch(idx), 一次编译任意规模复用)
│   ├── cim_jit.py          # JIT 系统仿真 (ExecutionEngine + ctypes 回调, L6)
│   ├── run_sim_text.py     # JIT 文本生成仿真
│   ├── to_object.py        # AOT: dump .o (L7)
│   └── aot/                # AOT 系统仿真 (cim_sim 可执行文件 + IPC server)
│       ├── cim_main.c / cim_ipc.c / cim_shm.c / cim_runtime.c  # 入口 + IPC(shm+reg) + 共享内存 + consume
│       ├── cim_sim_server.py                        # Python 仿真器 IPC server
│       ├── Makefile / run_aot.sh                    # 构建 / 一键启动
│       └── tokenizer_data.h                         # CharTokenizer C 化数据
└── pipeline.py          # 一键流水线 (9 步, 任意规模自动适配)
```

### 编译流程（9 步，`pipeline.py`）

| 步 | 阶段 | 产物 |
|---|---|---|
| 1 | export_fx | `.pt2` + `weights.bin` |
| 2 | L0 to_mlir | mlir |
| 3 | partition | `partition.json`（37 CIM 块）|
| 4 | C1 lower_to_cimres | cimres IR |
| 5 | C2 place | placed IR（容量校验 ≤4096 Macro）|
| 6 | C3 emit_instr | `forward.bin` + `preload.bin` |
| 7 | L1 cim_lowering | `placeholder.mlir`（含 `func.call @cim_launch`）|
| 8 | cim_jit --sim | JIT 数值验证（max_diff=0）|
| 9 | AOT 构建 | `cim_sim` 可执行文件 |

### 系统仿真（两种模式，共用 `cim_stub.c` + `HwCimSimulator`）

- **JIT 模式**：CPU 侧 LLVM JIT 进程内执行 + CIM 侧 ctypes 同进程回调（`cim_jit.py` / `run_sim_text.py`）。
- **AOT 模式**：CPU 侧独立可执行文件 `cim_sim` + CIM 侧 IPC（unix socket）跨进程（`cim_compiler/lowering/aot/`）。

两者数值完全一致（max_diff=0），AOT 模式更接近真实 CPU↔硬件分离。

### 运行指令

```bash
# 0. 环境
conda activate nanogpt-gpu   # 或直接 /home/li/anaconda3/envs/nanogpt-gpu/bin/python ...

# === 一键流水线 (9 步: export -> ... -> JIT 验证 + AOT 构建) ===
python cim_compiler/pipeline.py                              # 默认 shakespeare (6层512维)
python cim_compiler/pipeline.py --start-step 9               # 只跑 AOT 构建 (调试)
python cim_compiler/pipeline.py --no-sim                     # 跳过 JIT 仿真

# === 多规模回归 (任意规模 ≤4096 Macro, 随机权重验证编译兼容) ===
python cim_compiler/gen_random_model.py --n_layer 2 --d_model 256 --ffn_dim 1024 --out /tmp/small.pt
python cim_compiler/pipeline.py --ternary /tmp/small.pt --n_layer 2 --d_model 256 --ffn_dim 1024

# === JIT 系统仿真 (CPU LLVM JIT + CIM hw_simulator, ctypes 同进程) ===
python cim_compiler/lowering/cim_jit.py --sim                # 单次 forward + max_diff 验证
python cim_compiler/lowering/run_sim_text.py --prompt "ROMEO:" --num-tokens 60   # 文本生成

# === AOT 系统仿真 (CPU 可执行文件 + IPC 仿真器, 跨进程) ===
make -C cim_compiler/lowering/aot                            # 构建 cim_sim 可执行文件
./cim_compiler/lowering/aot/run_aot.sh --prompt "ROMEO:" --n 60   # 一键 (nohup server + cim_sim + kill)
# 或分两步:
python cim_compiler/lowering/aot/cim_sim_server.py &        # 仿真器 server (先启动, 循环 accept)
./cim_compiler/lowering/aot/cim_sim --prompt "ROMEO:" --n 60      # AOT 可执行文件驱动
```

### 产物

| 文件 | 内容 |
|---|---|
| `checkpoints/bitnet_ternary.pt2` | FX graph（37 cim.matmul，动态 seq len）|
| `checkpoints/bitnet_ternary_weights.bin` | CIM preload 权重 blob（37 层 2bit 打包）|
| `checkpoints/bitnet_ternary_partition.json` | CPU/CIM 划分（37 CIM 块 + 边界张量）|
| `checkpoints/bitnet_ternary_placeholder.mlir` | L1 降级产物（含 `func.call @cim_launch`）|
| `cim_compiler/cimres/checkpoints/forward.bin` | CIMF：37 段 MATMUL 指令流（按 idx 索引）|
| `cim_compiler/cimres/checkpoints/preload.bin` | CIMP：自包含 Preload（tile 数据 + PROG_WGT）|
| `cim_compiler/lowering/cim_stub.so` | 固定硬件驱动（MMIO，任意规模复用）|
| `cim_compiler/lowering/aot/cim_sim` | AOT 可执行文件（ELF）|

### 关键设计

- **cim.matmul custom op**：BitLinearInference 用 `torch.ops.cim.matmul`（int8 激活 × 2bit 打包权重 -> int32），torch.export 保留为 op 节点不内联，CPU/CIM 在 IR 天然分离。
- **固定硬件驱动**：`cim_stub.c` 单一 `cim_launch(idx, X, W)`，idx 运行时参数查 `forward.bin` 段，`.so` 一次编译任意规模复用（规模变只改 forward.bin + IR）。
- **tile 切分**：权重 `W[N,K]` 切 64×64 tile，一 tile 一 Macro（≤4096），N 维分块输出 / K 维累加 reduce（accum 字段）。
- **两阶段共享缓存**：Preload 一次性权重驻留 Macro（`preload.bin`/CIMP），Forward 流式激活（`forward.bin`/CIMF）。
- **数值/时序解耦**：dispatch 同步算 matmul+RMW（max_diff=0），`busy_until`/`page_busy` 统计 Macro 并行时序。
- **w_packed 传而不用**：`cim_launch` 不读 W 数据（Preload 已驻留 Macro），W memref 仅 shape 对即可，AOT 模式用空壳。
- **IPC 共享内存**（AOT）：shm_* 数据走 POSIX 共享内存（C/Python 共享 1MB，零拷贝零往返），reg_* 控制走 socket（同步点）。比纯 socket 快 ~40%（n=3：12010 -> 3307 socket 往返）。

详见 `sys_sim.md`（系统仿真运行时，JIT + AOT）+ `cim_mlp.md`（CIM 架构规范）。

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
cim_compiler/                   # CIM 异构编译前端 + 系统仿真
  export/ partition/ ir/        # 导出 + 划分 + torch-mlir 降级
  cimres/                       # CIM 资源映射 + 指令流 + hw_simulator (C1/C2/C3)
  lowering/                     # MLIR->LLVM + cim_stub 驱动 + JIT/AOT 系统仿真
    └── aot/                    # AOT 可执行文件 cim_sim + IPC server
  pipeline.py                   # 一键流水线 (9 步, 任意规模自动适配)
cim_model/                      # CIM 定点仿真模型 (早期, 验证 cim_mlp.md 计算通路)
sys_sim.md                      # 系统仿真运行时 (JIT + AOT 两种模式)
cim_mlp.md                      # CIM 协处理器架构规范
data/
  input.txt                     # tinyshakespeare 源文件（prepare.py 复用）
  shakespeare_char/             # prepare.py + train.bin/val.bin/meta.pkl
tests/                          # 层 + 量化单元测试
checkpoints/                    # 模型 checkpoint + 导出产物（.pt/.pt2/.bin/.json/.mlir）
cim_compiler/cimres/checkpoints/  # CIM 指令流产物（forward.bin / preload.bin）
```

## 参考

- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)（BitNet b1.58）
- [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)
- [nanoGPT](https://github.com/karpathy/nanoGPT) -- 训练循环 / 数据布局
- [torch-mlir](https://github.com/llvm/torch-mlir) -- PyTorch -> MLIR
