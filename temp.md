核心前提已探明：export 对 torch custom op 保留为 op 节点，不内联——cim.matmul op 节点 1 个，_int_mm 内联 0 个。这是用 torch-mlir 做 CIM 异构的关键，路径成立。

这与之前 export inline 子模块（call_module → 0）的失败形成对比：custom op 不是子模块，export 把它当 opaque 调用保留。这正好是 torch-mlir 异构接入的标准方式。

整体思路：用 custom op 让 CPU/CIM 在 IR 里自然分离

之前（内联 _int_mm）的问题：torch-mlir 会把 _int_mm + 解包链降级成 CPU linalg 循环，CIM 块失去硬件语义，事后无法识别。

改用 custom op 后：torch-mlir 对不认识的 cim.matmul op 保留为 torch.operator（不降级），CPU op 正常降级 linalg。CPU/CIM 在 IR 里天然分离，无需后处理切分。

torch-mlir 完整路径

1. 注册 cim.matmul custom op

@torch.library.custom_op("cim::matmul", mutates_args=())
def cim_matmul(x_int8: torch.Tensor, w_packed: torch.Tensor) -> torch.Tensor:
    """CIM Macro: int8 × 2bit-packed-ternary → int32"""
    # 真实实现: 解包 + _int_mm (CPU 仿真) 或调用 CIM runtime
    ...

@cim_matmul.register_fake        # export 需要的 meta (形状/dtype)
def _(x_int8, w_packed):
    M, _ = x_int8.shape; N = w_packed.shape[0]
    return torch.empty(M, N, dtype=torch.int32)

2. BitLinearInference 用 custom op 替代内联 _int_mm

def forward(self, x):
    x_norm = self.norm(x).reshape(-1, K)
    scale_x = 127.0 / x_norm.abs().max(...).clamp_min(1e-5)
    x_int8 = (x_norm * scale_x).round().clamp(-128, 127).to(torch.int8)
    acc = torch.ops.cim.matmul(x_int8, self.w_packed)    # ← custom op, 不内联
    return (acc.to(torch.float32) / (scale_x * self.scale_w)).reshape(*lead, -1)

3. torch.export 导出

图里 CIM 块 = cim.matmul op 节点（37 个），CPU op = norm/quant/rescale/attention 等。已验证 op 保留。

4. torch-mlir 导出 MLIR

import torch_mlir
mlir_module = torch_mlir.compile(
    prog, output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)
# CPU op → linalg dialect (自动降级)
# cim.matmul → torch.operator "cim.matmul" (保留, 不降级)

5. 产出的 MLIR IR

mlir
func.func @main(%idx: !torch.tensor) {
    %x_int8 = <linalg.matmul / arith> : tensor<MxKi8>           // CPU: norm+quant
    %acc = torch.operator "cim.matmul"(%x_int8, %w_packed)
        : (tensor<MxKi8>, tensor<NxK//4xi8>) -> tensor<MxNxi32>  // CIM: 保留为 op
    %out = <arith.divf> %acc : ...                              // CPU: rescale
    ...
}
CPU 是 linalg（可降级 LLVM），CIM 是 torch.operator（待自定义 lowering）。

6. 自定义 lowering pass：torch.operator → runtime call

写一个 mlir-opt pass（或用 MLIR Python builder）：
mlir
// lowering 前
%acc = torch.operator "cim.matmul"(%x_int8, %w_packed) : ...
// lowering 后
%acc = func.call @cim_launch(%x_int8, %w_packed) : (...) -> tensor<MxNxi32>
@cim_launch 是 host runtime 函数（封装共享缓存写/doorbell/累加区读，对应 cim_mlp.md 两阶段）。

7. CPU 部分降级到底

mlir-opt --convert-linalg-to-loops --convert-arith-to-llvm --convert-func-to-llvm ...
mlir-translate --mlir-to-llvmir  →  llc  →  机器码

与计算图划分（partition）的结合

custom op 让 partition 大幅简化：

┌──────────┬───────────────────────────────────────────────┬────────────────────────────┐
│          │        现有 partition（内联 _int_mm）         │   custom op 版 partition   │
├──────────┼───────────────────────────────────────────────┼────────────────────────────┤
│ CIM 识别 │ _int_mm + 权重解包链追溯（含 list args 展开） │ cim.matmul op 节点（直接） │
├──────────┼───────────────────────────────────────────────┼────────────────────────────┤
│ CIM 块   │ 16 节点（解包 + _int_mm）                     │ 1 个 op 节点               │
├──────────┼───────────────────────────────────────────────┼────────────────────────────┤
│ 边界     │ 推断 x_int8 / acc                             │ op 的输入/输出（显式）     │
└──────────┴───────────────────────────────────────────────┴────────────────────────────┘

def mark_cim_nodes(graph):
    return {n for n in graph.nodes
            if n.op == "call_function" and "cim::matmul" in str(n.target)}
无需追溯解包链。partition 仍提供：调用点定位、边界张量、CIM 块清单（37 个 cim.matmul）。

partition 在 torch-mlir 流程里的角色：
- 导出 IR 前：partition 的 node_backend 不再必需（custom op 天然分离）
- 导出 IR 后：partition 的 CIM 块清单 → MLIR 里的 torch.operator 节点 → 自定义 lowering 的输入。即 partition 提供"哪些 cim.matmul 对应哪个 BitLinear + tile_id"的映射，给 lowering pass 用。

数据流与交互逻辑的保证

回到上一个问题"如何确保 CPU 调取 CIM"——在 torch-mlir 路径下：

1. 调用点：cim.matmul op 在 IR 里就是调用点，CPU 代码（linalg）执行到它就触发 func.call @cim_launch
2. 数据流：MLIR SSA 保证 %x_int8 在调用前算出，%acc 在调用后消费
3. 数据搬运：@cim_launch runtime 封装共享缓存写/读（cim_mlp.md）
4. 权重预载：w_packed 是 buffer，Preload 阶段注入，forward 时作参数传给 cim.matmul（lowering 后用 tile_id）

工程结构

cim_compiler/
├── export/
│   ├── cim_op.py            # 新: 注册 cim.matmul custom op (+fake)
│   ├── inference_model.py   # 改: forward 用 torch.ops.cim.matmul
│   └── ...
├── partition/
│   └── classify.py          # 简化: cim.matmul op 即 CIM
└── lowering/                # 新增
    ├── to_mlir.py            # ExportedProgram → MLIR (torch_mlir.compile)
    ├── cim_lowering.py       # torch.operator "cim.matmul" → func.call @cim_launch (mlir-opt pass)
    └── runtime.py            # @cim_launch host 实现 (共享缓存/doorbell/累加区)

实施步骤（建议顺序）

1. cim_op.py：注册 cim.matmul custom op（fake + CPU 仿真实现）
2. 改 inference_model.py：BitLinearInference.forward 用 torch.ops.cim.matmul，重新导出 .pt2
3. 改 partition/classify.py：识别 cim.matmul op（简化），重跑 partition
4. 装 torch-mlir（见下挑战）
5. lowering/to_mlir.py：torch_mlir.compile 导出 MLIR，验证 CPU→linalg、CIM→torch.operator
6. lowering/cim_lowering.py：mlir-opt pass 把 torch.operator "cim.matmul" → func.call @cim_launch
7. lowering/runtime.py：cim_launch host 实现（对接 cim_mlp.md 共享缓存语义）

torch-mlir 安装挑战（主要障碍）

torch 2.12.0.dev 是极新版本，torch-mlir 可能无对应 wheel。三个选择：
- 源码编译 torch-mlir：clone torch-mlir，对着 torch 2.12 编（要 LLVM/MLIR 源码，你 mlir-tutorial/install 里的可能能用）
- 降 torch 版本：降到 torch-mlir 支持的稳定版（如 2.4/2.5），但会丢 2.12 的新特性
- torch-mlir nightly：试 pip install --pre torch-mlir 看有无匹配

如果 torch-mlir 装不上，退路是路 C（自建 MLIR builder）：用你 mlir-tutorial 的 MLIR Python/C++ API 直接构建 IR（CIM 块手写 cim.matmul op），不依赖 torch-mlir。custom op 已验证保留，FX graph 节点可直接遍历映射到 MLIR。