#!/usr/bin/env python3
"""PoC T2 (gate): 验证 torch-mlir 对 custom op 保留为 torch.operator。

结论 (torch 2.12 dev + torch-mlir 20260403.771):
  - RAW 模式: cim.matmul 保留为 torch.operator "torch.cim.matmul",
              CPU op 为 torch.aten.* (未降级 linalg)  ✓ 可用
  - TORCH / LINALG_ON_TENSORS: torchdynamo-export-to-torch-backend-pipeline 把
              unknown torch.operator 标记 illegal; backend-legal-ops 选项无效
              (cim.matmul / torch.cim.matmul / cim::matmul / torch.cim.matmul.default /
               torch.operator 五种格式均失败)  ✗ 不可用

方案: IR 导出用 RAW (torch dialect IR). CPU 降级 linalg 移到 lowering 阶段
      (先 cim.matmul -> func.call 消除 unknown op, 再跑 linalg pipeline)。

运行: nanogpt-gpu python cim_compiler/ir/poc_torch_mlir.py
"""
import torch
import torch.export
from torch_mlir.fx import export_and_import
from torch_mlir.compiler_utils import OutputType


@torch.library.custom_op("cim::matmul", mutates_args=())
def cim_matmul(x_int8: torch.Tensor, w_packed: torch.Tensor) -> torch.Tensor:
    """CIM Macro (dummy): float matmul 仿真。"""
    return (x_int8.to(torch.float32) @ w_packed.to(torch.float32).t()).to(torch.int32)


@cim_matmul.register_fake
def _(x_int8: torch.Tensor, w_packed: torch.Tensor) -> torch.Tensor:
    M, _ = x_int8.shape
    N = w_packed.shape[0]
    return torch.empty(M, N, dtype=torch.int32)


class DummyModel(torch.nn.Module):
    def forward(self, x_int8, w_packed):
        y = x_int8 + 1                            # CPU
        acc = torch.ops.cim.matmul(y, w_packed)   # CIM custom op
        return acc + 2                            # CPU


def main():
    m = DummyModel().eval()
    x = torch.randint(-128, 127, (4, 8), dtype=torch.int8)
    w = torch.randint(-1, 2, (6, 8), dtype=torch.int8)
    prog = torch.export.export(m, (x, w))

    # ---- RAW 模式 (主验证): cim.matmul 保留为 torch.operator ----
    print("=== RAW 模式: 导出 torch dialect IR ===")
    mod = export_and_import(prog, output_type=OutputType.RAW, func_name="main")
    ir = str(mod)
    n_cim = ir.count('torch.operator "torch.cim.matmul"')
    n_aten = ir.count("torch.aten.")
    print(f"  torch.operator 'torch.cim.matmul': {n_cim}")
    print(f"  torch.aten.* (CPU op): {n_aten}")
    print("  IR:")
    for line in ir.split("\n"):
        print("    " + line)

    assert n_cim == 1, "RAW: cim.matmul 未保留为 torch.operator (gate 失败)"
    assert n_aten >= 2, "RAW: CPU op 未保留为 torch.aten.*"
    print("\n✓ T2 PASS (gate, RAW 模式): custom op 保留为 torch.operator, CPU op 为 torch.aten.*")

    # ---- LINALG 模式: 记录 torch-mlir 限制 ----
    print("\n=== LINALG_ON_TENSORS 模式: 预期失败 (torch-mlir 对 unknown torch.operator 限制) ===")
    try:
        export_and_import(prog, output_type=OutputType.LINALG_ON_TENSORS,
                          backend_legal_ops=["torch.cim.matmul"], func_name="main")
        print("  (意外成功 - backend_legal_ops 可能已修复)")
    except Exception as e:
        print(f"  失败 (符合预期): {type(e).__name__}")
        print("  原因: torchdynamo-export-to-torch-backend-pipeline 把 unknown torch.operator 标记 illegal")
        print("  结论: CPU 降级 linalg 移到 lowering 阶段 (cim.matmul 先转 func.call 再跑 linalg pipeline)")


if __name__ == "__main__":
    main()
