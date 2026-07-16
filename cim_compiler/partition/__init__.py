"""partition — FX graph CPU/CIM 逻辑划分。

对导出的 export 图做 CPU/CIM 逻辑划分 (不修改图, 保留动态形状与调用 magic):
  - 标注每节点 backend (CPU/CIM)
  - 识别 CIM 块 (matmul + 权重解包链)
  - 输出 CPU↔CIM 边界张量 (x_int8 激活 / acc int32)

CIM 节点 = matmul + 权重解包链 (对应 cim_mlp.md CIM Macro: 2bit 节点 + 宏内解包 + int32 累加)。
边界: CPU→CIM = x_int8 (per-token int8 激活), CIM→CPU = acc (int32, 给 CPU rescale)。

物理可执行子图 GraphModule 在 export 图上不可行 (export 图密封: inline 子模块 / split 丢 magic /
修改破坏 magic), 故用逻辑子图 (标注 + 边界 + 数据流验证), compiler 后端遍历原图按标签调度。
"""

