#!/usr/bin/env python3
"""注册 cim::matmul custom op - CIM Macro: int8 激活 × 2bit 打包三值权重 -> int32 累加。

op 语义:
  x_int8   [M, K] int8        per-token int8 激活 (CPU 侧量化)
  w_packed [N, K//4] uint8    2bit 补码打包三值权重 (4 code/byte, CIM 预载)
  -> [M, N] int32              累加输出 (CIM->CPU 边界)

注册为 torch custom op 后:
  - torch.export 保留为 op 节点 (target str "cim.matmul.default", 不内联)
  - torch-mlir RAW 模式保留为 torch.operator "torch.cim.matmul"
  - LINALG pipeline 对 unknown torch.operator 报错 (backend_legal_ops 无效), 故 IR 导出用 RAW
  - CPU 仿真 impl 不进 export 图 (custom op 不内联), 仅 eager/数值验证用

对应 cim_mlp.md CIM Macro: 2bit 节点 + 宏内解包 + int32 累加。
"""
import torch
import torch.library


def unpack_2bit_aten(packed: torch.Tensor) -> torch.Tensor:
    """uint8[..., K//4] (2bit 补码) -> int8[..., K] {-1,0,1}。

    cim.matmul 的 CPU 仿真解包 (不进 export 图, custom op 不内联):
      div/mod 取 4 路 code -> stack/reshape -> where(code>=2, code-4, code)
    2bit 补码: 0->0, 1->+1, 3->-1 (2 未用)。
    """
    p = packed.to(torch.int32)
    c0 = p % 4
    c1 = (p // 4) % 4
    c2 = (p // 16) % 4
    c3 = (p // 64) % 4
    code = torch.stack([c0, c1, c2, c3], dim=-1).reshape(*packed.shape[:-1], -1)
    return torch.where(code >= 2, code - 4, code).to(torch.int8)


@torch.library.custom_op("cim::matmul", mutates_args=())
def cim_matmul(x_int8: torch.Tensor, w_packed: torch.Tensor) -> torch.Tensor:
    """CIM Macro (CPU 仿真): int8 [M,K] x 2bit 打包 [N,K//4] -> int32 [M,N]。

    仿真: 解包 w_packed -> w_int8 {-1,0,1} + float matmul。
    int8x{-1,0,1} 在 float32 精确 (|acc|<=65536<2^24), 数值等价 _int_mm int32。
    真实 CIM: 调用 CIM runtime (lowering 阶段对接)。
    """
    w_int8 = unpack_2bit_aten(w_packed)  # [N, K] {-1,0,1}
    acc = (x_int8.to(torch.float32) @ w_int8.to(torch.float32).t()).to(torch.int32)
    return acc


@cim_matmul.register_fake
def _(x_int8: torch.Tensor, w_packed: torch.Tensor) -> torch.Tensor:
    M, _ = x_int8.shape
    N = w_packed.shape[0]
    return torch.empty(M, N, dtype=torch.int32)
