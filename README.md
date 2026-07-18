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

### bitnet-cim 包（editable 安装）

`pyproject.toml` 把 `bitnet/` + `cim_compiler/` 声明为 `bitnet-cim` 包（含 `ir` / `lowering.aot` regular 子包），editable 安装后全 repo 用绝对包导入，取代各脚本顶部的 `sys.path.insert` hack：

```bash
pip install -e .                       # editable 安装 (repo 根, pyproject.toml 所在)
```

安装后：

| 用法 | 说明 |
|---|---|
| `cim-pipeline` 命令 | 任意 cwd 跑全 12 步流水线（`pyproject.toml` 注册的 console script = `cim_compiler.pipeline:main`）|
| 绝对包导入 | `from cim_compiler.export.inference_model import` / `from bitnet.model import`，取代裸模块 + sys.path hack |
| 包结构 | `bitnet` + `cim_compiler`（`cimres`/`export`/`partition`/`lowering`/`ir`/`lowering.aot` 均为 regular 子包）|

**前提**：cim_compiler 各脚本已删 sys.path hack（包化重构后），**必须 editable 安装**才能跑——绝对导入依赖 `cim_compiler`/`bitnet` 是可导入包；未安装则 `import cim_compiler.X` 失败。

**外部依赖（editable 解决不了）**：`cim_jit.py`/`to_object.py`/`linalg_to_llvm.py` 用 `torch_mlir_e2e_test`（refbackend L4 pipeline），它不在 pip wheel 里，需从 torch-mlir 源码树导入——脚本内保留 `_E2E_PATH = "/home/li/workspace/torch-mlir/projects/pt1/python"` 路径 hack（机器特定，后续可改环境变量）。

**加新包后刷新**：新增 `__init__.py`（新子包）需重新 `pip install -e .` 让 setuptools 重新发现包（egg-info/SOURCES.txt 自动更新）。

### 已知坑

- **dev wheel 只配 nightly**：torch-mlir 预编译只有 nightly，无 stable 版本，故用 torch nightly。
- **API 选择**：`torch_mlir.compile`（老 jit.script 路径）对 BitNet 不行；必须用
  `torch_mlir.fx.export_and_import`。
- **namespace package 初始化**：torch-mlir 2026 是 namespace（无 `__init__.py`），
  需 `import torch_mlir._mlir_libs` 触发 `_site_initialize`，再手动
  `_torchMlir.register_dialect(ctx)` 注册 torch dialect 给 `FxImporter`。
- **推理 sdpa dtype**：model RoPE 用 float32 cos/sin 会把 q/k 提升为 float32，
  与 bfloat16 的 v 不一致致 `scaled_dot_product_attention` 报 dtype 错
  (`Expected query, key, and value to have the same dtype`)。
  `generate_shakespeare_char.py` 默认 `--dtype float32` 规避（推理 float32 精度也最高）。

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

所有命令在 `nanogpt-gpu` env 下运行（`conda activate nanogpt-gpu`，或绝对路径 `/home/li/anaconda3/envs/nanogpt-gpu/bin/python`）。CIM 编译前端依赖 editable 安装（`pip install -e .`，见「环境」章节）。

### 端到端：训练 -> 三值推理

```bash
# 1. 准备数据（vocab=65）
python data/shakespeare_char/prepare.py

# 2. 训练（nanoGPT 循环 + BitNet lr/wd 调度, bf16; 自动保存 val 最优 -> checkpoints/bitnet_shakespeare_char_best.pt）
python bitnet/train_shakespeare_char.py
python bitnet/train_shakespeare_char.py --smoke          # 20 步冒烟测试

# 3. 生成（float32 采样, 默认 temperature=0.8 top_k=40; --seed 可复现）
python bitnet/generate_shakespeare_char.py --prompt "ROMEO:" --max_tokens 128 --seed 0
python bitnet/generate_shakespeare_char.py                                              # 从 BOS 自由采样
#    注: 默认 --dtype float32 规避 sdpa dtype 坑 (见"已知坑"); greedy 对比用 _greedy_ref.py

# 4. 导出 2bit 三值权重（float32 -> 2bit packed, 8x 压缩, ~3.7 MB）
python bitnet/export_ternary.py

# 5. 三值推理（2bit packed + Triton kernel, 无需解包; 详见下方「三值推理」）
python bitnet/generate_shakespeare_char.py --ternary --prompt "ROMEO:" --max_tokens 128 --seed 0
```

预期表现：best val ≈ 1.53，约在 step 1250 达到（笔记本 GPU 约 4 分钟），约 step 2500 触发早停。

### CIM 编译前端 + 系统仿真（cim_compiler）

```bash
# === 一键流水线 (12 步: export -> ... -> JIT 验证 + AOT; 非 KV + KV 两条独立) ===
python cim_compiler/pipeline.py                              # 非 KV 全流程 (export_fx -> cim_sim, make nv)
python cim_compiler/pipeline.py --kv                         # KV 流程 (export_kv -> cim_sim_kv, make kv, 跳过 cim_jit)
python cim_compiler/pipeline.py --start-step 12              # 只跑 AOT 构建 (调试)
python cim_compiler/pipeline.py --no-sim                     # 跳过 JIT 仿真

# === 多规模回归 (任意规模 ≤4096 Macro, 随机权重验证编译兼容) ===
python cim_compiler/gen_random_model.py --n_layer 2 --d_model 256 --ffn_dim 1024 --out /tmp/small.pt
python cim_compiler/pipeline.py --ternary /tmp/small.pt --n_layer 2 --d_model 256 --ffn_dim 1024

# === JIT 系统仿真 (CPU LLVM JIT + CIM hw_simulator, ctypes 同进程) ===
python cim_compiler/lowering/cim_jit.py --sim                # 单次 forward + max_diff 验证
python cim_compiler/lowering/run_sim_text.py --prompt "ROMEO:" --num-tokens 60        # 文本生成 (全序列, O(n²))
python cim_compiler/lowering/run_sim_text.py --prompt "ROMEO:" --num-tokens 80 --kv   # KV cache 增量 (decode M=1, O(n))
# 采样解码 (默认 greedy; 加 --temperature >0 启用 softmax+top_k+multinomial, 打破 greedy 坍缩, JIT/ref 同 seed 对比):
python cim_compiler/lowering/run_sim_text.py --prompt "ROMEO:" --num-tokens 80 --kv --temperature 0.8 --top_k 40 --seed 0

# === AOT 系统仿真 (CPU 可执行文件 + IPC 仿真器, 跨进程) ===
make -C cim_compiler/lowering/aot nv                       # 构建 cim_sim (非 KV only, 不依赖 KV 产物)
make -C cim_compiler/lowering/aot kv                       # 构建 cim_sim_kv (KV only)
make -C cim_compiler/lowering/aot                          # 构建 cim_sim + cim_sim_kv (all = nv + kv, 需两者产物就绪)
./cim_compiler/lowering/aot/run_aot.sh --prompt "ROMEO:" --n 60           # 全序列一键 (nohup server + cim_sim + kill)
./cim_compiler/lowering/aot/run_aot_kv.sh --prompt "ROMEO:" --n 80        # KV cache 增量一键 (cim_sim_kv --kv)
# AOT 采样 (默认 greedy; 加 --temperature >0 启用 C 侧 softmax+top_k+multinomial; srand(seed+1) 避 glibc srand(0)==srand(1) 陷阱, 不同 seed 不同输出):
./cim_compiler/lowering/aot/run_aot.sh    --temperature 0.8 --top_k 40 --seed 0 --prompt "ROMEO:" --n 60   # 全序列采样
./cim_compiler/lowering/aot/run_aot_kv.sh --temperature 0.8 --top_k 40 --seed 0 --prompt "ROMEO:" --n 80   # KV 增量采样
# 或分两步 (server 先启动循环 accept; cim_sim 跑完断开, server 留守下一连接):
python cim_compiler/lowering/aot/cim_sim_server.py --socket /tmp/cim_sim.sock &
./cim_compiler/lowering/aot/cim_sim --socket /tmp/cim_sim.sock --prompt "ROMEO:" --n 60           # 全序列
./cim_compiler/lowering/aot/cim_sim_kv --socket /tmp/cim_sim.sock --kv --prompt "ROMEO:" --n 80   # 增量 (n=80 speedup 42x)
# PPA: cim_sim 跑完 server 自动打印架构级 PPA 报告 (28nm@1GHz: 耗时/功耗/能效/面积), 详见 sys_sim.md §7
#      小批量快速看 PPA: 先启 server, 再 ./cim_sim --prompt R --n 3

# === KV cache CIM cycle 量化 (cost_model, 不跑仿真器, 秒级) ===
python cim_compiler/cimres/cost_model.py --kv 80    # n=80: 全序列重算 vs KV cache cycle + speedup (40.5x, 实测 42x)
python cim_compiler/cimres/cost_model.py --kv       # 对比表 n=32/64/128/256 (speedup ≈ (n+1)/2, O(n²)->O(n))

# === KV cache 数值验收 (PyTorch ref: 全序列 use_cache=False vs 增量 KV cache greedy) ===
python cim_compiler/export/verify_kv.py            # n=32 greedy 完全一致; n=128 良性浮点分歧 (relu² 放大)
```

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
│   ├── inference_model.py   # BitLinearInference + RotaryMHAInference (step_stateless 增量 KV cache)
│   ├── export_fx.py          # torch.export -> .pt2 (动态 seq len)
│   ├── export_kv.py          # 增量 KV cache export (_KVCacheModel -> bitnet_ternary_kv.pt2, dynamic T cache)
│   ├── verify_kv.py          # KV cache 数值验收 (全序列 use_cache=False vs 增量 greedy)
│   ├── cim_op.py             # cim::matmul custom op 注册 (CPU/CIM 在 IR 天然分离)
│   └── weight_blob.py        # CIM 权重预加载 blob (自描述二进制)
├── partition/           # FX graph CPU/CIM 划分
│   ├── classify.py          # CIM 节点标记 (cim.matmul op)
│   └── partition.py         # 逻辑子图 + 边界张量 (.json)
├── ir/                  # torch-mlir 降级 (MLIR linalg IR)
├── cimres/              # CIM 资源映射 + 指令流 + 仿真器
│   ├── lower_to_cimres.py / place.py / emit_instr.py  # C1/C2/C3 (IR->指令流)
│   ├── hw_config.py       # ASIC 硬件参数集中定义 (C/Python 镜像, 防漂移)
│   ├── ppa_config.py      # 架构级 PPA 估算 (PPAConfig + ActivityTracker + PPAEstimator, 28nm@1GHz)
│   ├── cost_model.py      # CIM cycle 评估 (estimate + estimate_kv, S3 KV cache O(n²)->O(n))
│   └── hw_simulator.py   # cycle 级纯硬件仿真器 (HwCimSimulator, Macro 并行时序 + PPA)
├── lowering/            # MLIR -> LLVM + 系统仿真驱动
│   ├── cim_stub.c          # 固定硬件驱动 (MMIO, 单一 cim_launch(idx), 一次编译任意规模复用)
│   ├── hw_config.h         # ASIC 硬件参数 (C, 与 hw_config.py 镜像)
│   ├── buffer_kind.py      # BUFFER target -> kind 分类 (cim_jit build_inputs + gen_config 共用, B 类去重)
│   ├── cim_jit.py          # JIT 系统仿真 (ExecutionEngine + ctypes 回调, L6)
│   ├── run_sim_text.py     # JIT 文本生成仿真
│   ├── to_object.py        # AOT: dump .o (L7)
│   └── aot/                # AOT 系统仿真 (cim_sim 可执行文件 + IPC server)
│       ├── cim_main.c / cim_ipc.c / cim_shm.c / cim_runtime.c  # 入口(libffi) + IPC(shm+reg) + 共享内存 + consume
│       ├── model_config.h / gen_config.py          # 运行时模型配置 (.pt2->model_config.bin, cim_main 固定宿主)
│       ├── cim_sim_server.py                        # Python 仿真器 IPC server (跑完打印 PPA)
│       ├── Makefile / run_aot.sh / run_aot_kv.sh    # 构建 / 全序列一键 / KV 增量一键
│       └── tokenizer_data.h                         # CharTokenizer C 化数据 (旧, cim_main 改用 model_config)
└── pipeline.py          # 一键流水线 (12 步, 任意规模自动适配; --kv 增量 KV 流程)
```

### 编译流程（12 步，`pipeline.py`；`--kv` 走 KV 路径，跳过 step 11）

| 步 | 阶段 | 产物 |
|---|---|---|
| 1 | export_fx / export_kv（`--kv`）| `.pt2` + `weights.bin` |
| 2 | L0 to_mlir | mlir |
| 3 | partition | `partition.json`（25 CIM 块，qkv 合并）|
| 4 | C1 lower_to_cimres | cimres IR（tile 展开 + role 元数据）|
| 5 | cimres passes | canon + cse（逻辑层冗余消除）|
| 6 | C2 place | placed IR（容量校验 ≤4096 Macro，PAGE 布局）|
| 7 | verify | 结构校验（dest_id/accum/PAGE/a_page）|
| 8 | 调度分析 | cost_model/scheduler/page_alloc（makespan/最优性/PAGE 报告）|
| 9 | C3 emit_instr | `forward.bin` + `preload.bin`（`--kv` 用 `_kv` 后缀）|
| 10 | L1 cim_lowering | `placeholder.mlir`（含 `func.call @cim_launch`）|
| 11 | cim_jit --sim | JIT 数值验证（max_diff=0；`--kv` 跳过，经 make run_kv 验证）|
| 12 | AOT 构建 | `cim_sim` / `cim_sim_kv`（`make nv` / `make kv`）|

### 系统仿真（两种模式 × 全序列/KV cache，共用 `cim_stub.c` + `HwCimSimulator`）

- **JIT 模式**：CPU 侧 LLVM JIT 进程内执行 + CIM 侧 ctypes 同进程回调（`cim_jit.py` / `run_sim_text.py`）。
- **AOT 模式**：CPU 侧独立可执行文件 `cim_sim`/`cim_sim_kv` + CIM 侧 IPC（unix socket）跨进程（`cim_compiler/lowering/aot/`）。
- **全序列 vs KV cache 增量**：全序列每步重算前序 K/V（CIM matmul M=T，O(n²)）；KV cache 增量每步只喂新 token（M=1，O(n)）。CPU 侧 KV cache（`inference_model.py` `step_stateless`），CIM/lowering 零改动，O(n²)->O(n) 由 `cim_launch` 运行时 M 维 T->1 自然达成（`cim_stub.c` 对 M 行循环，makespan ∝ M 行数）。

两者数值完全一致（max_diff=0），AOT 模式更接近真实 CPU↔硬件分离。KV cache 实测 n=80 speedup **42x**（全序列 `cim=22819256` vs 增量 `cim=542784`，≈ n/2）。

### 产物

| 文件 | 内容 |
|---|---|
| `checkpoints/bitnet_ternary.pt2` | FX graph（37 cim.matmul，动态 seq len）|
| `checkpoints/bitnet_ternary_weights.bin` | CIM preload 权重 blob（37 层 2bit 打包）|
| `checkpoints/bitnet_ternary_partition.json` | CPU/CIM 划分（25 cim_blocks，qkv 合并 + 边界张量）|
| `checkpoints/bitnet_ternary_placeholder.mlir` | L1 降级产物（含 `func.call @cim_launch`）|
| `cim_compiler/cimres/checkpoints/forward.bin` | CIMF：25 段 MATMUL 指令流（qkv 合并后，按 idx 索引）|
| `cim_compiler/cimres/checkpoints/preload.bin` | CIMP：自包含 Preload（tile 数据 + PROG_WGT）|
| `cim_compiler/cimres/checkpoints/model_config.bin` | CIMC：运行时模型配置（超参+buffer描述+tokenizer, cim_main 固定宿主读）|
| `cim_compiler/lowering/cim_stub.so` | 固定硬件驱动（MMIO，任意规模复用）|
| `cim_compiler/lowering/aot/cim_sim` | AOT 可执行文件-全序列（ELF，libffi 运行时变参）|
| `checkpoints/bitnet_ternary_kv.pt2` | 增量 KV cache FX graph（`_KVCacheModel`，dynamic T cache，多输出 logits+new_k+new_v）|
| `cim_compiler/cimres/checkpoints/{forward,preload,model_config}_kv.bin` | 增量 KV 产物（`forward_kv.bin`=`forward.bin`，CIM 侧零改动）|
| `cim_compiler/lowering/aot/cim_sim_kv` | AOT 可执行文件-增量 KV（多输出 consume `mrf32_mrf32_mrf32`，`run_kv` 增量循环）|

### 关键设计

- **cim.matmul custom op**：BitLinearInference 用 `torch.ops.cim.matmul`（int8 激活 × 2bit 打包权重 -> int32），torch.export 保留为 op 节点不内联，CPU/CIM 在 IR 天然分离。
- **固定硬件驱动**：`cim_stub.c` 单一 `cim_launch(idx, X, W)`，idx 运行时参数查 `forward.bin` 段，`.so` 一次编译任意规模复用（规模变只改 forward.bin + IR）。
- **tile 切分**：权重 `W[N,K]` 切 64×64 tile，一 tile 一 Macro（≤4096），N 维分块输出 / K 维累加 reduce（accum 字段）。
- **两阶段共享缓存**：Preload 一次性权重驻留 Macro（`preload.bin`/CIMP），Forward 流式激活（`forward.bin`/CIMF）。
- **数值/时序解耦**：dispatch 同步算 matmul+RMW（max_diff=0），`busy_until`/`page_busy` 统计 Macro 并行时序。
- **w_packed 传而不用**：`cim_launch` 不读 W 数据（Preload 已驻留 Macro），W memref 仅 shape 对即可，AOT 模式用空壳。
- **IPC 共享内存**（AOT）：shm_* 数据走 POSIX 共享内存（C/Python 共享 1MB，零拷贝零往返），reg_* 控制走 socket（同步点）。比纯 socket 快 ~40%（n=3：12010 -> 3307 socket 往返）。
- **cim_main 固定宿主**：AOT `cim_main.c` 通用宿主，超参运行时从 `model_config.bin` 读（`gen_config.py` 从 .pt2 提取），forward 参数个数随 n_layer 变由 **libffi** 运行时变参调用解决，换模型不改 C 代码。
- **PPA 架构级估算**：`ppa_config.py` 在仿真器上加活动因子统计，28nm@1GHz 估算计算耗时/功耗能效/面积（±30~50%），cim_sim 跑完 server 自动打印报告（见 `sys_sim.md` §7）。
- **KV cache（CPU 侧，CIM 零改动）**：`inference_model.py` `RotaryMHAInference.step_stateless(h, k_in, v_in, cos, sin)` 无状态增量 forward（cache 显式 IO，cos/sin 外部传入避免动态 arange），`_KVCacheModel` 包装导出 dynamic T cache。CIM/lowering/cim_stub 零改动，O(n²)->O(n) 由 `cim_launch` 运行时 M 维 T->1（`cim_stub.c` 对 M 行循环，makespan ∝ M 行数，非 M-tile ceil(T/64)）。AOT 多输出 consume `mrf32_mrf32_mrf32`（3 个 `{i64,ptr}*` 参数，对齐 refbackend `get_ctype_func`）。实测 n=80 speedup 42x（`cost_model --kv` 量化 speedup ≈ (n+1)/2）。
- **跨 BitLinear q/k/v 合并（S6）**：同层 q/k/v 三个 BitLinear 读同一 x_int8，dest_id 不重叠（q:0-63/k:64-95/v:96-127），合并为 1 doorbell（128 MATMUL + SYNC_HALT），CIM 内不同 Macro 天然并行（§4.7.7），makespan 338->198/层，CIM cycle 6268->5428（-13.4%，spec-faithful 真提升）。IR 级合并：`cim_stub_lower.py` 用 partition.json 元数据识别 qkv（方案 B: role q/k/v + 同 blk_idx, IR walk 顺序 == cim_blocks 展开顺序; 替代旧 shape 启发式 [Nq>Nk=Nv]）+ IR 变换 pass（`_collect_movable` BFS 收集 q/k si32 的 transitive use，按 IR 顺序移到 qkv call result 后恢复 dominance）+ `cim_launch_qkv(idx,Xq,Wq,Xk,Wk,Xv,Wv)->(Q,K,V)` 多输出（Memref2D3 struct by value，LLVM sret）。forward.bin 段数 37->25，cim_sim 跑通数值不变。

### 编译器架构分析

> 上面「关键设计」讲实现机制（怎么做），本节讲架构特征、权衡与扩展边界（是什么/为什么/边界）。

**定位**：单算子存算加速器前端。所有 CIM 加速收敛到 `cim::matmul` 一个 op（int8×2bit 三值 -> int32），其余算子（norm/量化/rescale/attention/embedding）一律 CPU。扩展边界 = 这个算子语义能覆盖多少架构，非通用 NPU 编译器。

#### 三个硬假设点（架构耦合处）

| # | 位置 | 硬编码内容 |
|---|---|---|
| 1 | `inference_model.py` `_replace` | 只 isinstance 认 `BitLinear` + `RotaryMHA`，硬 `from bitnet.model import` |
| 2 | `cim_op.py` | op 签名写死 `int8[M,K] × 2bit三值[N,K//4] -> int32[M,N]`（解包/位宽/精度全硬编码）|
| 3 | `classify.py` + `cimres/dialect.py` | CIM 节点只有 `cim.matmul`；cimres IR 只有 matmul tile 类 op |

#### 数据流：权重静态全驻留 / 激活动态广播

Macro 不存激活/PSUM，只存权重；激活和 PSUM 在 1MB SharedCache（SRAM）。

| 空间 | 存什么 | 生命周期 |
|---|---|---|
| Macro RRAM（4096×64×64 2bit cell）| 权重 1 tile = 1024B | **静态全驻留**（RRAM 非易失，推理期不重载）|
| SharedCache A_PAGE（`0x010+`）| 激活 int8（64B/tile）| 动态，每 token 重写 |
| SharedCache PSUM（`0xC00+`）| 输出 int32（256B/tile）| RMW 累加，K 维多 tile 共享 |

- **权重**：3664 tile 一次性 Preload 到 3664 个 Macro（< 4096），跨层共享不重载，利用率 **89%**。dest_id 线性编号（mesh 清理后 `T_ROUT_PER_HOP=0`，无物理 2D 网格路由，所有 Macro 等价）。
- **激活**：K 维切 `k_tiles` 段存 a_page，同 k_blk 的多个 N-tile **广播共享**读入（1 份激活喂多个 Macro）。
- **PSUM**：同 n_blk 跨 K 维 RMW 累加（`accum` 字段，首拍覆盖、后续累加）。

一次 q_proj（512×512，M=1 单 token）运算的分布：

```
权重: Macro[0..63] 各持 64×64 tile (Preload 驻留, 不动)
kb=0: a_page=0x010 ─广播─> Macro[ 0.. 7] 并行 8 N-tile -> psum 0xC00..0xC07 (accum=0 覆盖)
kb=1: a_page=0x011 ─广播─> Macro[ 8..15] 并行 8 N-tile -> psum 0xC00..0xC07 (accum=1 累加)
  ...
kb=7: a_page=0x017 ─广播─> Macro[56..63] 并行 8 N-tile -> psum = 完整 512 int32 输出
```

#### 时序模型（`hw_config.py`，@1GHz -> 1 cycle = 1 ns）

| 常量 | cycle | 含义 |
|---|---|---|
| T_DISPATCH | 2 | 广播总线路由 Dest_ID |
| T_PROG_WGT | 10 | Preload：2bit tile 解包装载 |
| T_MATMUL | 64 | 运算：ADC 逐列量化 64 列（KCL 并行 1 cycle，**ADC 串行是瓶颈**，非 KCL）|
| T_WB | 4 | 写回：int32 RMW 累加 |

- 稳态一次 tile = T_DISPATCH + T_MATMUL + T_WB = **70 cycle**（70 ns）
- 调度：**k 外 n 内**（K 维串行避免 PSUM 写冲突，N 维并行共享 a_page 广播）
- 单 Macro 峰值 64 MAC/cycle = 0.128 TOPS @1GHz；4096 Macro 满载 ~524 TOPS（理论，受 dispatch/总线/利用率限制）

#### PPA 估算口径（`ppa_config.py`，28nm@1GHz，±30~50%）

- **TOPS = 2 × GMACs**：1 MAC = 2 OP 标称口径（工业标称，非 CIM 物理实际操作数）。CIM 三值权重省乘法器，能效高主要来自**能耗分母小**（e_mac=0.5pJ），非分子。
- **两口径**：稳态 `tops_w`（不含 Preload 一次性 RRAM 编程）/ amortized `tops_w_amort`（含 Preload 分摊到本次）。
- **估算偏保守**：`n_mac` / `n_prog_cell` 按全 tile 4096 计，未扣 0 权重 -> 能耗高估、能效低估（见下）。

#### 稀疏性：32% 训练 0 + 0.21% 补零

- **训练稀疏 0**：4,791,436（32.0%），模型学出，各层均匀（30~33%），+1/-1 近乎对称（34.08% / 33.92%）
- **结构补零**：32,256（0.21%），tile 对齐 ceil 补的，集中在 `lm_head` 尾部 8 个 Macro（vocab=65 非 64 整除，每 Macro 98.4% 是补零）

CIM 对 0 权重的稀疏红利分三层（核心：0 cell 电导 G=0 -> 电流 I=0，欧姆定律物理天然不贡献，非「检测后跳过」）：

| 资源 | 能否省 | 原因 |
|---|---|---|
| MAC 动态能耗 | ✅ 天然省 | RRAM G=0 -> I=0，0 cell 物理不产生电流（已发生）|
| Preload 编程能耗 | ✅ 可省（需 ISA 改）| 0 cell 可不编程（默认高阻=0），省 ~32% 写入 |
| 计算时序 T_MATMUL | ❌ 省不了 | ADC 逐列量化，列内 0 不影响；整列全 0 概率 = 0.32⁶⁴ ≈ 0 |

> 当前 PPA 按全 tile 计未反映 32% 物理红利 -> 估算保守。L1（零硬件改 `ppa_config` 统计非零）可让估算诚实化；L2（ISA 加 partial PROG_WGT + 位图）省真实 Preload 能耗。三值 2bit 稠密已紧凑，存储稀疏化不划算（元数据开销抵消收益）。

#### 架构扩展边界（变化支持矩阵）

| 架构变化 | 支持度 | 改动点 |
|---|---|---|
| 层数/维度/head/FFN/vocab/seq_len | 零改动（参数化）| - |
| GQA | 已支持 | `n_kv_head`（`RotaryMHAInference`）|
| norm/激活/attention 变体/FFN 加 gate | 小改 | 仅 `inference_model.py` 层适配，CIM/lowering 零改动 |
| 权重位宽/激活精度/累加精度 | 全链改 | op 签名 -> cimres -> hw_config -> ppa |
| 新增 CIM 算子（conv/attention 加速）| 全链改 | op -> classify -> dialect -> lower -> emit -> sim -> ppa |
| MoE 动态路由 / 跨层存算融合 | 需重写 | FX 静态图 + 单 matmul tile IR 模型不支持 |

一句话：**Transformer + Linear 范式内的演化基本能支持（只改 `inference_model`，CIM 零改动）；量化/位宽/精度或新算子是全链改；MoE/跨层融合需重写**。

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
  generate_shakespeare_char.py  # 推理（float32 采样 / --ternary 2bit kernel, --dtype 可选）
  export_ternary.py             # 导出 2bit 打包三值权重（8x 压缩）
  ternary_kernel.py             # Triton 2bit 三值 matmul kernel + CPU fallback
cim_compiler/                   # CIM 异构编译前端 + 系统仿真
  export/ partition/ ir/        # 导出 + 划分 + torch-mlir 降级
  cimres/                       # CIM 资源映射 + 指令流 + hw_simulator (C1/C2/C3)
  lowering/                     # MLIR->LLVM + cim_stub 驱动 + JIT/AOT 系统仿真
    └── aot/                    # AOT 可执行文件 cim_sim + IPC server
  pipeline.py                   # 一键流水线 (12 步, 任意规模自动适配; --kv 增量 KV 流程)
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
