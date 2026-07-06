# 极简 CIM 协处理器架构与指令集规范 — BitNet 三值推理专精版 (W1.58A8)

> **定位**：硬件架构与编译器探索阶段的极简版本，面向 **BitNet b1.58** 三值权重推理。权重以 **2bit 三值 `{-1,0,+1}`** 静态驻留于存算宏阵列，激活以 **per-token int8** 量化输入（W1.58A8）。宏内仅完成 `int8 × ternary` 累加、**输出 int32 部分和**（不做 rescale）；**rescale 由 CPU 在最后一个 tile 后统一完成**，还原浮点结果。

---

## 1. 概述

核心关注点：静态三值计算图的宏映射 (Macro Mapping)、宏间并行流、总线带宽利用率。基于 BitNet W1.58A8 做出"理想化黑盒"硬件抽象：纯静态三值权重计算，2bit 权重 + int8 激活的 BitLinear GEMM。

### 1.1 系统边界

$\text{系统边界} = \text{CPU} + \text{CIM 协处理器}$。本架构对软件暴露为统一的 **CIM 协处理器**：CPU 负责非线性、归一化、动态形状操作、激活 int8 量化**以及 rescale**；CIM 仅处理静态三值权重的**整数矩阵向量乘**（各 tile $\vec{y} = W \cdot \vec{x}_{int8}$，多 tile 在累加区 int32 累加为 $\text{acc}$），浮点还原 $\hat{y} = \text{acc} / (\text{scale\_x} \cdot \text{scale\_w})$ 在 CPU 完成。

- **CPU**：Embedding 查表、SubLayerNorm/RMSNorm、ReLU²、RoPE、GQA 广播、`scaled_dot_product_attention`（Q@K^T、Attn@V 动态 matmul）、**激活 per-token int8 量化**（产出 `x_int8` 写共享缓存 + `scale_x` **CPU 侧持有**）、**rescale**（读 int32 累加结果 + CPU 侧 `scale_x`/`scale_w` → FP32，**FP32 留 CPU 侧主存参与后续计算，不写回共享缓存**）、写指令流、敲门铃、IRQ 后读结果
- **CIM**：接收 int8 激活，对驻留 2bit 三值权重执行**矩阵向量乘**（int8 × ternary），**输出 int32 向量**写回共享缓存（不 rescale）

编译器两级边界划分：

| 边界 | 负责操作 |
|------|---------|
| **CPU 子图** | 标量控制、Embedding、SubLayerNorm/RMSNorm、ReLU²、Softmax、RoPE、Attention 动态 matmul、**激活 int8 量化**、**rescale（int32→FP32）**、数据排布重组 |
| **CIM 协处理器子图** | BitLinear 静态三值权重线性层的整数矩阵向量乘（QKV/FFN/lm_head 投影），输出 int32 向量 |

**宏驻留权重软件契约**：(1) 模型初始化阶段执行一次 **preload stream**，把所有 BitLinear 的 2bit 三值权重 tile 写入 Macro 资源（全权重驻留）；`scale_w` 由 CPU 从 checkpoint 读入主存常驻、**CPU 侧持有**，**不存入 Macro、不写入共享缓存**（CIM 不参与 rescale）。(2) Forward 期每条 `MACRO_MATMUL` 闭合到 tile 级 `placement_id`，不再搬运权重。

### 1.2 Macro 资源 (64×64, 2bit)

每个存算宏物理维度 $64 \times 64$，每个交叉点节点存储 **1 个 2bit 三值权重** $W_{i,j} \in \{-1,0,+1\}$，每个 Macro 容量 $64 \times 64 \times 2\text{bit} = 1024\text{Byte}$（1 KB）。

**Macro 计算（矩阵向量乘，单 tile 部分和）**：Macro 是 64×64 矩阵向量乘单元。输入 int8 向量 $\vec{x} \in [-128,127]^{64}$（K 维的一个 64 切片），与内部三值权重矩阵 $W \in \{-1,0,+1\}^{64\times64}$ 执行矩阵向量乘，**输出 int32 部分和向量** $\vec{y} \in \mathbb{Z}^{64}$：
$$y_j = \sum_{i=0}^{63} W_{j,i} \cdot x_i \quad (\text{int32})$$
$W_{j,i} \in \{-1,0,+1\}$（三值，乘法退化为加/减/零，无需乘法器），int32 求和（最大 $64 \times 128 = 8192$，安全）。Macro **无 rescale 单元、不存 `scale_w`**。

> **权重存储 = 节点态**：$W$ 以 2bit 补码 `{-1,0,1}` 存于 64×64 阵列交叉点（`-1→0b11`，`0→0b00`，`+1→0b01`，4 code/byte 打包 uint8）。软件传输即此格式，`MACRO_PROG_WGT` 预载直接加载到节点寄存器，无需编解码——传输格式与节点物理态完全一致。
>
> **部分和 → 完整结果 → rescale（三层）**：单 Macro 输出的 $\vec{y}$ 仅是 K 维一个 64 切片的**部分和**，**非最终结果**；K 维所有 tile 的 $\vec{y}$ 在累加区 int32 累加为 $\text{acc}$（**完整 int32 结果向量**，rescale 前）；最后 CPU 一次性 rescale $\hat{y} = \text{acc} / (\text{scale\_x} \cdot \text{scale\_w})$（`scale_x·scale_w` 对同一 n 块所有 K tile 是常数）。Macro 仅做整数矩阵向量乘、输出部分和 $\vec{y}$，硬件极简。
>
> BitNet **无偏置**（BitLinear 不含 bias），宏内不进行偏置加法；共享缓存累加区仅用于多 tile int32 部分和累加（见 §2.1）。

### 1.3 MLIR 层映射

| 操作 | 语义 |
|------|------|
| `cimres.macro_matmul` | 映射 `MACRO_MATMUL`：接收 1 PAGE（64 维 int8，`scale_x` 由 CPU 侧持有、**不进 PAGE**），与 64×64 三值权重执行**矩阵向量乘**，**输出 int32 向量**（不 rescale）；通过 `resident_weight` 属性绑定静态三值权重 logical tile |
| `cimres.local_transfer` | 表达逻辑层布局变换，不生成物理 DMA，只记录 metadata |

**定长页抽象**：基于 64×64 宏维度与 W1.58A8 格式，1 输入 PAGE = $64 \times \text{int8}$ = 64 Byte（`scale_x` 为 CPU 侧私有标量，**不进 PAGE**）；1 输出 PAGE = $64 \times \text{int32}$ = 256 Byte（CIM 输出 int32 部分和）。SRAM 寻址步长固定，维度不足 64 时由编译器 Zero-Padding（§4.8）。

---

## 2. 硬件架构

协处理器内置 **4096 个 64×64 Macro**，总三值权重存储 $4096 \times 1\text{KB} = 4\text{MB}$，恰好容纳 BitNet 15M 模型全部 2bit 三值权重（约 3.74 MB，见 §4.5）。共享缓存 **1 MB**（仅放 int8 输入 + int32 部分和 + 指令，权重驻留 Macro 内部；FP32 中间结果与 `scale_x`/`scale_w` 同为 CPU 侧私有，**不进共享缓存**）。

### 2.1 部分和累加区 (int32 ALU)

BitLinear **无偏置**，原 FP32 版本的"偏置累加区"语义不再需要。64×64 Macro 处理大权重需沿 K 维分 tile，多 tile int32 部分和必须累加。故 1MB 共享缓存分两区，累加区为 **int32 ALU**：

| 区域 | 地址范围 | 语义 |
|------|---------|------|
| **覆盖区 (Overwrite)** | PAGE 0 ~ 3071 | 纯 SRAM 写覆盖；**两阶段语义切换**（见 §4.6）：Preload 期 PAGE 0~3055 全部用于 2bit tile 预载暂存，Forward 期仅存放 int8 输入特征 + 指令（FP32 结果留 CPU 侧，不进共享缓存） |
| **部分和累加区 (Psum Accum)** | PAGE 3072 ~ 4095 | 写入端口前置 **int32 ALU**，触发原位 RMW：`MEM[Addr] += Incoming_Psum`（int32，K 维多 tile 累加，**无 bias 预写**） |

**多 tile 累加 → 完整结果向量**：对一个输出 n 块（64 个输出），K 维 $K/64$ 个 Macro 各输出一个 $k$ 切片的 int32 部分和 $\vec{y}$，依次写到同一累加区 PAGE，硬件 int32 RMW 累加为 $\text{acc}$——**$\text{acc}$ 是 K 维全部 tile 累加后的完整 int32 结果向量**（对应 BitLinear 该 n 块输出，rescale 前）。最后一个 tile 写完后 CPU 读 $\text{acc}$ 做 rescale 还原 FP32。**不再有 bias 预写步骤**。

### 2.2 物理数据流图

```
=====================================================================================
|                               主控 CPU (Host CPU)                                 |
|   rescale: int32 acc / (scale_x · scale_w) → FP32   (K 维全部 tile 累加完后)      |
=====================================================================================
         ^ IRQ_CIM 中断                              | 系统总线 AXI/APB (指令/数据/门铃)
         |                                           v
+-----------------------------------------------------------------------------------+
|                CPU+CIM 统一共享缓存 (Shared Buffer, 1MB)                          |
|-------------------------------------------------------+---------------------------|
|        覆盖区 (Overwrite, 768KB)                      | 部分和累加区 (Psum, 256KB)|
| [输入 int8] | [指令区]   (FP32→CPU 主存, 不进缓存)    |   [K 维多 tile int32 RMW] |
|                                                       |int32 ALU (+) → int32 acc  |
+-------------------------------------------------------+---------------------------+
         | 指令流 / int8 特征数据                       | int32 Psum 写回
         v                                              |
+-------------------------------------------------------|---------------------------+
|              极简 CIM 协处理器 (CIM Coprocessor)      |                           |
|                                                       |                           |
|   +---------------------------------------------+     |                           |
|   |  控制器 (指令解析 / 宏资源分配)              |    |                           |
|   +----------------------+----------------------+     |                           |
|                          v                            |                           |
|   | 多宏分发与总线驱动 (Bus Dispatcher)         |     |                           |
|   +----------------------+----------------------+     |                           |
|                          v                            |                           |
|   |  内部广播总线: [ Dest_ID 12b | 负载 ]      |      |                           |
|   +-----+-----------+-----------+---------+----+      |                           |
|         v           v           v             v       |                           |
|   +----------+  +----------+  +-------+  +----------+ |                           |
|   | Macro 0  |  | Macro 1  |  |  ...  |  |Macro4095 | |                           |
|   |64×64 2bit|  |64×64 2bit|  |       |  |64×64 2bit| |                           |
|   |int8×T MvM|  |int8×T MvM|  |       |  |int8×T MvM| |                           |
|   |int32 vec |  |int32 vec |  |       |  |int32 vec | |                           |
|   +----+-----+  +----+-----+  +-------+  +----+-----+ |                           |
|        | 1 PAGE int32 out  | 1 PAGE int32    | 1 PAGE |                           |
|        v (Psum int32)      v (Psum int32)    v (Psum) |                           |
|   |          上行总线仲裁器 (Upstream Arbiter)        |                           |
|   +---------------------------------------------------+                           |
+------------------------------------------------------------+----------------------+
```

### 2.3 门铃与中断

| 寄存器 | 地址 | 操作 | 说明 |
|--------|------|------|------|
| `DOORBELL_REG` | `0x00` | 写 | 写入指令区起始地址，唤醒分发器取指 |
| `INT_CLEAR_REG` | `0x04` | 写 | 写任意值清除中断/状态 |
| `IRQ_STATUS_REG` | `0x08` | 读 | 状态: 0=idle, 1=busy, 2=done, 3=error |
| `IRQ_CIM` | — | 物理连线 | 中断信号线 |

**Doorbell 语义**：CPU 写指令流后向 `DOORBELL_REG` 写起始地址，协处理器顺序取指执行至 `SYNC_HALT`（§3.4）终止。

---

## 3. 指令集架构 (ISA)

4096 个 Macro 需 12-bit 寻址（$2^{12}=4096$），原 32-bit 位宽不足以容纳 12-bit Dest_ID + 双 PAGE 寻址，故扩展至 **48-bit**。BitLinear 无偏置，原"地址分区决定 bias"语义移除，累加区由 `ACCUM` 标志位显式控制（§3.3）。

### 3.1 48-bit 指令格式

| 字段 | 位宽 | 说明 |
|------|------|------|
| `[47:45]` | 3 | **Opcode**（最多 8 种指令） |
| `[44:33]` | 12 | **Dest_ID**：Macro 路由地址 (0~4095) |
| `[32:21]` | 12 | **PAGE_1**：数据页负载 1（覆盖 1MB SRAM 的 4096 PAGE，PAGE=256B 步长） |
| `[20:9]` | 12 | **PAGE_2**：数据页负载 2 |
| `[8]` | 1 | **ACCUM**：0=覆盖区纯覆盖写，1=部分和累加区 int32 RMW |
| `[7:0]` | 8 | 保留 |

### 3.2 MACRO_PROG_WGT（Opcode `0x1`，静态三值权重配置）

把连续 PAGE 中的 2bit 三值权重（$64 \times 64$ 个 2bit = 1024 Byte，4 个 PAGE）配置到 `Dest_ID` 指定的 Macro。**不传 `scale_w`**（rescale 由 CPU 完成，`scale_w` 由 CPU 侧持有、**不进共享缓存**）。

- **语义**：模型初始化阶段 **Preload**。三值权重以 **2bit 补码 `{-1,0,1}`** 打包存储（`-1→0b11`，`0→0b00`，`+1→0b01`，4 code/byte uint8）；**预载时硬件直接加载该 2bit 补码到节点寄存器**，无需编解码转换（传输格式即节点态）。加载后长驻阵列。Macro **不存 `scale_w`**（无 rescale 单元）。

| Payload | 位域 | 含义 |
|---------|------|------|
| `PAGE_1` | `[32:21]` | `B_PAGE_START`：2bit 三值权重数据起始页 |
| `PAGE_2` | `[20:9]` | 保留（不用） |
| `ACCUM` | `[8]` | 忽略 |

### 3.3 MACRO_MATMUL（Opcode `0x2`，三值权重向量乘）

`Dest_ID` 指定的 Macro 接收 1 PAGE int8 特征向量 (64×1)，与内部 64×64 三值权重执行**矩阵向量乘**（int8 × ternary），**直接输出 1 PAGE int32 部分和向量** $\vec{y}$（**不 rescale**），按 `ACCUM` 写回。

**数学**（单 tile 矩阵向量乘 → 部分和）：$\vec{y} = W \cdot \vec{x}_{int8}$，$y_j = \sum_{i=0}^{63} W_{j,i} \cdot x_{int8,i}$（int32），$W_{j,i} \in \{-1,0,+1\}$（加/减/零）。$\vec{y}$ 是 K 维一个 tile 的**部分和**；K 维多 tile 的 $\vec{y}$ 在累加区 int32 RMW 累加为 $\text{acc}$（**完整结果向量**）；**CPU rescale**：$\hat{y} = \text{acc} / (\text{scale\_x} \cdot \text{scale\_w})$。

| `ACCUM` | 写回语义 |
|---------|---------|
| `0`（覆盖区） | 纯覆盖：$MEM = \vec{y}$ (int32 向量) |
| `1`（累加区） | 原位 int32 RMW：$MEM = \vec{y} + MEM[\text{PSUM\_PAGE}]$（K 维多 tile 累加为 $\text{acc}$） |

**语义**：专用于 BitLinear 三值权重投影（QKV/FFN/lm_head），**无偏置加法、无 rescale**。`scale_x`、`scale_w` 均为 **CPU 侧私有标量**（**不进共享缓存、CIM 不读**），rescale 时 CPU 直接取用。Macro 仅做整数矩阵向量乘、输出 int32 部分和向量 $\vec{y}$。

| Payload | 位域 | 含义 |
|---------|------|------|
| `PAGE_1` | `[32:21]` | `A_PAGE`：输入 int8 特征页地址（`scale_x` **不进 PAGE**，CPU 侧持有） |
| `PAGE_2` | `[20:9]` | `PSUM_PAGE`：输出 int32 写回页地址 |
| `ACCUM` | `[8]` | 0=覆盖写，1=int32 部分和累加 |

### 3.4 SYNC_HALT（Opcode `0x7`，同步屏障）

标记指令段结束。控制器等待所有已发射 Macro 操作完成并写回后，拉高 `IRQ_CIM` 唤醒 CPU。Payload 全字段忽略。

---

## 4. 计算模型

本节定义 BitNet 三值推理在 CIM 上的端到端计算模型——前向推理如何切分为 CPU 序列与 CIM 指令序列，及两者如何通过共享缓存交换数据。

### 4.1 整体数据流

```
一次前向推理 = 1 次 Preload Phase + 1 次 Forward Phase
Preload (仅一次):  所有三值权重 tile (2bit) 写入 Macro → 永久驻留; scale_w 由 CPU 从 checkpoint 读入主存常驻 (不进共享缓存)
Forward (每次推理, 逐 BitLinear):
  CPU: Embedding → SubLN → int8 量化 (x_int8 写共享缓存 + scale_x CPU 侧) → 写指令流 → 敲门铃 → 等 IRQ
  CIM: int8×ternary 矩阵向量乘 → 写回 int32 向量 (不 rescale) → 发 IRQ
  CPU: 读 int32 累加结果 + CPU 侧 scale_x/scale_w → rescale → FP32 → (下一层) int8 量化 → ...循环...
```

### 4.2 Preload Phase：三值权重预加载

对每个 BitLinear 权重 $W \in \mathbb{R}^{N \times K}$（`weight_quant` 后为三值 `{-1,0,1}`，per-tensor `scale_w`）：

```
For each BitLinear weight (q_proj, k_proj, v_proj, o_proj, fc1, fc2, lm_head):
1. CPU 将 W ∈ R^{N×K} 切分为 64×64 tile (N/64 输出块 × K/64 输入块)
   - 若 N 或 K 不足 64 → zero-pad 到 64; 记录 original/padded_shape + valid_region
2. 对每个 tile:
   a. 三值 {-1,0,1} 直接以 2bit 补码打包 (code = ternary: -1→0b11, 0→0b00, +1→0b01),
      4 code/byte → 1024 Byte (4 PAGE); 预载时硬件直接加载到节点, 无编解码
   b. tile 数据写入共享缓存覆盖区 (Preload 阶段覆盖区全部用作预载暂存, 见 §4.6)
   c. 发射 MACRO_PROG_WGT (Dest_ID=目标 Macro, PAGE_1=tile 起始页)  # 不传 scale_w
   d. 协处理器传输 2bit 权重到指定 Macro, 原地存储
3. CPU 把该 BitLinear 的 scale_w (FP32) 从 checkpoint 读入主存常驻 (CPU 侧持有, 不写共享缓存)
4. 所有 Macro 编程完成 → 协处理器空闲等待 Forward Phase
```

**驻留契约**：三值权重一旦加载即永久驻留（直到下次 Preload 或复位）；`scale_w` 由 CPU 侧主存常驻（rescale 时取用，**不进共享缓存**）；Forward 期 `MACRO_MATMUL` 不携带权重地址，仅引用 `Dest_ID`。

### 4.3 Forward Phase：BitNet Block 逐层执行

以 BitNet b1.58 一次 Transformer Block 为例（`d_model=512, n_head=8, n_kv_head=4, ffn_dim=1664`，全部 BitLinear 无 bias）。**每个 BitLinear 后 CPU 做 rescale 还原 FP32，再量化为下一层 int8 输入**。

**4.3.1 Attention** — `SubLayerNorm(CPU) → x ∈ R^{1×512} → CPU 量化: x_int8=round(x·scale_x), scale_x=127/max|x|`

```
CIM: q_proj BitLinear (512→512, 无 bias)
  Weight: 三值 W_q ∈ {-1,0,1}^{512×512}, 8×8=64 个 64×64 tile
  对每个输出 n 块 (64 n): 8 个 k-tile Macro 各算 64 维 int32 部分和
    → 写到同一累加区 PAGE (ACCUM=1, int32 RMW 累加 8 次) → 得 64 维 int32 acc
  CPU rescale: y_q = acc / (scale_x · scale_w_q)  ← 一次性, CPU 侧 scale_w
  k_proj (512→256), v_proj (512→256), o_proj (512→512): 同流程
CPU: rescale 结果 → split 为 Q/K/V → RoPE → Q@K^T → softmax → Attn@V → GQA → o_proj(CIM) → 残差 x += attn_out
```

**4.3.2 MLP** — `SubLayerNorm(CPU) → x → CPU 量化: x_int8 + scale_x`

```
CIM: fc1 BitLinear (512→1664, 无 bias)
  Weight: W_fc1 ∈ {-1,0,1}^{1664×512}, N=1664 (26 块, 1664/64=26), K=512 (8 块)
  26×8 = 208 tile → 208 Macro → MACRO_MATMUL (ACCUM=1, K 维 8 tile int32 累加) → int32 acc
CPU: rescale: y_fc = acc / (scale_x · scale_w_fc1) → ReLU²: y_relu2 = ReLU(y_fc)²    # BitNet 论文激活
CPU: 量化 y_relu2 → y_int8 + scale_y (下一层输入)
CIM: fc2 BitLinear (1664→512, 无 bias)
  Weight: W_fc2 ∈ {-1,0,1}^{512×1664}, N=512 (8 块), K=1664 (26 块) → 8×26 = 208 tile
  → MACRO_MATMUL (ACCUM=1, K 维 26 tile int32 累加) → int32 acc
CPU: rescale: y_proj = acc / (scale_y · scale_w_fc2) → 残差 x += y_proj
```

**4.3.3 lm_head** — `SubLayerNorm(CPU) → x → CPU 量化: x_int8 + scale_x`

```
CIM: lm_head BitLinear (512→65, 无 bias)
  Weight: W_lm ∈ {-1,0,1}^{65×512}, N=65 (2 块, ceil(65/64)=2), K=512 (8 块) → 2×8 = 16 tile
  → MACRO_MATMUL (ACCUM=1, K 维 int32 累加) → int32 acc
CPU: rescale: logits = acc / (scale_x · scale_w_lm) → FP32 logits ∈ R^{1×65}  (vocab=65)
```

### 4.4 指令流模板

完整 Forward Phase 的 CIM 指令序列（block0 部分）；**rescale 不占指令流，由 CPU 在 IRQ 后软件执行**：

```
// block0 attention — q_proj: 8 个 n 块, 每块 K 维 8 tile int32 累加
//   n 块 0: Macro_0..Macro_7 各算一个 k 切片, int32 输出累加到 PAGE 0xC00
MACRO_MATMUL  Dest=Macro_0  A_PAGE=0x010  PSUM_PAGE=0xC00  ACCUM=0  // 首个 tile, int32 覆盖
MACRO_MATMUL  Dest=Macro_1  A_PAGE=0x011  PSUM_PAGE=0xC00  ACCUM=1  // int32 累加
... (Macro_2..7, ACCUM=1)
//   ↑ 8 tile 完成后, CPU 读 PAGE 0xC00 (int32 acc) + CPU 侧 scale_x/scale_w → rescale → FP32 y_q
//   n 块 1: Macro_8..Macro_15 → PAGE 0xC01 ; 同模式 + CPU rescale
// (CPU: RoPE + Q@K^T + softmax + Attn@V + GQA)
// o_proj: Macro_64..Macro_127 → 各 n 块累加区 + CPU rescale
// block0 MLP — fc1: 208 tile (26 n × 8 k) → Macro_192..Macro_399 + CPU rescale + ReLU² + 量化
//              fc2: 208 tile → Macro_400..Macro_607 + CPU rescale
// (CPU: ReLU² + 残差)
// block1..5: 同上 (复用或继续分配 Macro)
// lm_head
MACRO_MATMUL  Dest=Macro_3648  A_PAGE=0x100  PSUM_PAGE=0xC10  ACCUM=1  // lm_head (2×8=16 tile, K 维累加) → int32 acc
// CPU rescale → FP32 logits (CPU 侧主存, 不进共享缓存)
SYNC_HALT                                                          // 同步屏障
```

### 4.5 Macro 资源分配与合理性验证

BitNet 15M 全部 BitLinear 权重的 64×64 tile 数：

| BitLinear | shape (N×K) | tile (N/64 × K/64) | 小计 |
|-----------|------------|-------------------|------|
| q_proj | 512×512 | 8×8 = 64 | |
| k_proj | 256×512 | 4×8 = 32 | |
| v_proj | 256×512 | 4×8 = 32 | |
| o_proj | 512×512 | 8×8 = 64 | attn 192 |
| fc1 | 1664×512 | 26×8 = 208 | |
| fc2 | 512×1664 | 8×26 = 208 | mlp 416 |
| **每层合计** | | | **608** |

- 6 层 × 608 = **3648 tile**；lm_head (65×512) = 2×8 = **16 tile**；**总计 3664 tile**（每个 tile 占 1 Macro）
- **容量验证**：3664 < 4096 Macro ✓，余 432 可用于流水/冗余
- **权重存储验证**：4096 × 1KB = 4MB ≈ BitNet 15M × 2bit = 3.75MB ✓
- **K 维累加开销**：每个输出 n 块需 K/64 次 int32 RMW 累加（q_proj 每块 8 次，fc2 每块 26 次），均在累加区完成，**最后 1 次 CPU rescale**（每 n 块一次除法，非每 tile 一次）

> **合理性结论**：64×64 Macro + 4096 个 + 2bit 节点存储，可一次性驻留整个 BitNet 15M 模型三值权重；每层 BitLinear 通过 K 维多 tile int32 RMW 累加完成任意维度 GEMM；**Macro 仅做整数矩阵向量乘、输出部分和 $\vec{y}$，K 维全 tile 累加为 $\text{acc}$ 后 CPU 一次性 rescale**——省 n-1 次除法且 int32 累加精确（无 FP32 累加误差）。模型无 bias，累加区仅承担 int32 部分和累加，无偏置预写开销。

### 4.6 共享缓存地址空间布局

共享缓存采用**两阶段动态划分**：Preload 与 Forward 在时间上不重叠，覆盖区在两阶段间**语义切换**而非物理切分。Preload 阶段覆盖区全部用于权重预载暂存（最大化单批可编程 tile 数，避免预载区静态预留拖累运行效率）；Forward 阶段权重已驻留 Macro、不再需要预载区，覆盖区仅承载 int8 输入特征 + 指令（**FP32 中间结果全程 CPU 侧主存，不进共享缓存**）。**数据区与指令区物理分离**——指令区固定占据覆盖区末尾 16 PAGE，两阶段不变。

**Preload 阶段**（覆盖区全用于 2bit tile 预载暂存）：

```
覆盖区 (Overwrite): PAGE 0x000 ~ 0xBEF (3056 pages, 764KB)
  └── 预载暂存区: 全部用于 2bit tile 暂存
     (单 tile = 64×64×2bit = 1024B = 4 PAGE; 单批可暂存 3056/4 = 764 tile,
      批量发射 MACRO_PROG_WGT 编程对应 Macro; BitNet 15M 共 3664 tile → 分 5 批)
指令区:        PAGE 0xBF0 ~ 0xBFF (16 pages, 4KB, 48-bit 指令流)
部分和累加区:  PAGE 0xC00 ~ 0xFFF (1024 pages, 256KB) — 未启用 (Forward 才用)
```

**Forward 阶段**（权重已驻留 Macro, 取消预载区, 数据区流式复用）：

```
覆盖区 (Overwrite): PAGE 0x000 ~ 0xBEF (3056 pages, 764KB)
  ├── 输入特征区 (int8): A_PAGE, CPU 量化后写入, CIM 读
  │   (scale_x 为 CPU 侧私有标量, 不进共享缓存)
  ├── 指令区:           PAGE 0xBF0 ~ 0xBFF (16 pages, 4KB)
  └── 空闲: 原 0x200~0xBEF 预载区释放, 可作多 BitLinear int8 输入流水暂存
部分和累加区 (int32): PAGE 0xC00 ~ 0xFFF (1024 pages, 256KB)
  └── K 维多 tile int32 累加区 (CIM 写 int32 acc, CPU 读后 rescale; 无 bias 预写, 运行时 int32 RMW)
```

**FP32 数据全程 CPU 侧私有**：CPU 读 int32 acc → rescale → FP32 结果直接在 CPU 主存参与后续计算（残差、RoPE、Q@K^T、softmax、ReLU²、下一层 norm+量化），**不写回共享缓存**——CIM 仅消费 int8 激活、产出 int32 部分和，FP32 对 CIM 不可见。共享缓存只承载 int8 输入 + int32 部分和 + 指令，FP32 与 `scale_x`/`scale_w` 同为 CPU 侧私有。

具体 PAGE 地址由编译器在覆盖区内动态分配（§4.4 示例 A_PAGE 地址 0x010/0x100 等均落在此区；int32 acc 写累加区 0xC00+）。

| 区域 | Preload 阶段 | Forward 阶段 |
|------|-------------|--------------|
| 覆盖区 0x000~0xBEF | 预载暂存（764KB 全用于 2bit tile） | int8 输入 + 指令（无预载区、无 FP32 输出区） |
| 指令区 0xBF0~0xBFF | 指令流 | 指令流 |
| 累加区 0xC00~0xFFF | 未启用 | int32 部分和累加（256KB） |

> **设计要点**：Preload 期可用 764KB 预载（单批 764 tile，较静态预留预载区的 636KB 提升约 20%，分批次数减少）；Forward 期释放全部预载空间给 int8 输入/流水暂存，避免预载区空占拖累运行效率。**FP32 中间结果全程 CPU 侧主存私有、不进共享缓存**（CIM 仅消费 int8、产出 int32，FP32 对 CIM 不可见）；`scale_x`/`scale_w` 同为 CPU 侧私有标量，**不进共享缓存**。

### 4.7 关键时序契约

1. **权重复位**：Preload 期间覆盖区全部用作 2bit tile 预载暂存（见 §4.6），CPU 写 tile → `MACRO_PROG_WGT`（不传 scale_w）→ 等待完成；CPU 从 checkpoint 读 `scale_w` 入主存常驻（**不写共享缓存**）；所有 Macro 编程完成后才进入 Forward Phase（覆盖区切换为输入/输出语义）
2. **激活量化**：发射 `MACRO_MATMUL` 前，CPU 必须完成 SubLayerNorm + int8 量化（`x_int8` 写入 A_PAGE，`scale_x` **CPU 侧持有**）
3. **输入就绪**：发射 `MACRO_MATMUL` 时，A_PAGE 的 int8 特征必须已写就绪（`scale_x` 在 CPU 侧，**不占 A_PAGE**）
4. **int32 部分和累加**：K 维多 tile 累加时，首个 tile `ACCUM=0`（int32 覆盖残留），后续 `ACCUM=1`（int32 RMW 累加）；CPU 读结果前不得修改对应累加区 PAGE
5. **CPU rescale**：K 维全部 tile 累加完毕（IRQ 后），CPU 读累加区 int32 acc + **CPU 侧** `scale_x`/`scale_w`，**一次性**计算 $y = \text{acc} / (\text{scale\_x} \cdot \text{scale\_w})$ → FP32；**FP32 结果留 CPU 侧主存**参与后续计算（残差/RoPE/attention/ReLU²/norm），**不写回共享缓存**；下一层 BitLinear 前由 CPU 重新量化为 int8 写入 A_PAGE
6. **输出消费**：CPU 在 IRQ 后读 int32 累加结果前不得修改累加区 PAGE；CIM 执行期间 CPU 不得读写被 `MACRO_MATMUL` 作为目标地址的覆盖区 PAGE
7. **宏间并行**：同一 Macro 串行执行 `MACRO_MATMUL`；不同 Macro 可并行；`SYNC_HALT` 后所有 Macro 保证已完成

### 4.8 Zero-Padding 约定

当 BitLinear 维度不足 64 时，编译器负责生成 zero-padding：

- **输入向量**：`x_int8 ∈ R^{1×d}` (d<64) → pad 到 `R^{1×64}`，`x'[0:d]=x`，`x'[d:64]=0`；`scale_x` 不变（per-token，基于原始 x）
- **权重矩阵**：`W ∈ {-1,0,1}^{N×K}` → pad 到 `{ceil(N/64)×64, ceil(K/64)×64}`，多余行列置 0；如 lm_head N=65 → pad 到 128（2 块），第 2 块仅前 1 行有效
- **输出向量**：int32 acc ∈ R^{1×N}，`acc[0:N]` 有效，`acc[N:64]` 为 padding 无关值（CPU rescale 后同理）
- **元数据**：IR 中记录 `original_shape` / `padded_shape` / `valid_region`，不生成实际填充数据——填充由硬件或 runtime 在数据搬运时完成
