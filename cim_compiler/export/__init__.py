"""export - BitNet b1.58 模型 -> cim::matmul custom op FX graph 导出工程。

把 BitNet b1.58 模型导出为「原生三值权重」FX graph, 作为 CIM 硬件 + CPU 异构计算的
编译器前端输入。

  - cim_op: 注册 cim::matmul custom op (int8 × 2bit 打包 -> int32), torch.export 保留为 op 节点
  - BitLinearInference: 2bit 打包三值权重 + cim::matmul 定点前向
      norm -> per-token int8 量化 -> cim::matmul (int32) -> rescale (FP32 留 CPU 侧)
  - torch.export -> .pt2: 部署 IR, 动态 seq len [1..block_size], 2bit 权重作常量内嵌, cim.matmul op 保留
  - 旁路 .bin: CIM 权重预加载 blob (自描述二进制, Preload 阶段用)

对应 cim_mlp.md 的 CIM 定点通路: int8 激活 × 2bit 节点 -> int32 累加 -> CPU rescale。
"""
from . import cim_op  # noqa: F401  (注册 cim::matmul custom op side-effect, torch.export 反序列化需要)

