#!/usr/bin/env python3
"""L3: linalg 后插 func.call @cim_launch + canonicalize 清死链。

从 L1 产出 (placeholder.mlir) in-memory 跑 L2 pipeline (确保 tm_tensor dialect 注册, 因
Python 无 tm_tensor 注册 binding, 不能重新 parse linalg.mlir), 然后在 linalg IR 上:

对每个 placeholder 降级出的 linalg.matmul (37 个, 有 int8 cast 链), 追溯其输入到原始
int8 tensor, 替换整条 cast 链为一个 func.call @cim_launch_<idx>:

  追溯链 (placeholder 降级固定模式):
    ins[0] f32 <- linalg.generic{sitofp} <- X_si8  (tensor<?xKxi8>, 量化激活)
    ins[1] f32 <- linalg.generic{uitofp} <- linalg.transpose <- tensor.collapse_shape
                 <- linalg.generic{yield/broadcast} <- W_packed  (tensor<NxK/4xi8>, block arg)
    result  f32 -> linalg.generic{fptosi} -> si32  (tensor<?xNxi32>)

  替换为:
    %r = func.call @cim_launch_<idx>(X_si8, W_packed) : (tensor<?xKxi8>, tensor<NxK/4xi8>) -> tensor<?xNxi32>

  区别 CPU attention: attention 用 tm_tensor.attention (非 linalg.matmul), CPU 其他 matmul
  是 f32 直运 (无 sitofp/uitofp cast 链), 故 37 个 linalg.matmul 全是 placeholder 产物。

最后 canonicalize + cse 清死链 (cast generic/repeat/transpose/collapse/fill 等变 dead)。

用法:
  python cim_compiler/lowering/cim_stub_lower.py
  python cim_compiler/lowering/cim_stub_lower.py --in checkpoints/bitnet_ternary_placeholder.mlir \\
    --out checkpoints/bitnet_ternary_final.mlir
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d, func as func_d
from torch_mlir.fx import _module_lowering
from torch_mlir.compiler_utils import OutputType, run_pipeline_with_repro_report


def _owner(v):
    """Value -> defining Operation (OpResult.owner)。BlockArg 会触发下游 assert。"""
    return v.owner


def lower_linalg_to_cim_call(mod) -> int:
    """把 placeholder 降级的 linalg.matmul 替换为 func.call @cim_launch_<idx>。返回替换数。"""
    ctx = mod.context
    matmuls = []

    def find(op):
        if str(op.name) == "linalg.matmul":
            matmuls.append(op)
        return ir.WalkResult.ADVANCE

    mod.operation.walk(find)

    for idx, mm in enumerate(matmuls):
        loc = mm.location
        # ---- 追溯输入 ----
        # ins[0] f32 <- sitofp generic <- X_si8
        sitofp_gen = _owner(mm.operands[0])
        assert str(sitofp_gen.name) == "linalg.generic", f"ins0 非 generic: {sitofp_gen.name}"
        X_si8 = sitofp_gen.operands[0]
        # ins[1] f32 <- uitofp generic <- transpose <- collapse <- broadcast generic <- W_packed
        uitofp_gen = _owner(mm.operands[1])
        assert str(uitofp_gen.name) == "linalg.generic", f"ins1 非 generic: {uitofp_gen.name}"
        transpose = _owner(uitofp_gen.operands[0])
        collapse = _owner(transpose.operands[0])
        broadcast_gen = _owner(collapse.operands[0])
        W_packed = broadcast_gen.operands[0]  # block arg, 原始 packed ternary
        # result f32 -> fptosi generic -> si32
        fptosi_gen = None
        for u in mm.results[0].uses:
            if str(u.owner.name) == "linalg.generic":
                fptosi_gen = u.owner
                break
        assert fptosi_gen is not None, "matmul result 无 fptosi generic use"
        si32_result = fptosi_gen.results[0]
        # outs (init 0, 通常 linalg.fill)
        fill = None
        outs_v = mm.operands[2]
        if hasattr(outs_v.owner, "name"):  # OpResult -> Operation
            fill = outs_v.owner

        # ---- 声明 @cim_launch_<idx> (private, tensor signature) ----
        sym = f"cim_launch_{idx}"
        func_type = ir.FunctionType.get(
            inputs=[X_si8.type, W_packed.type],
            results=[si32_result.type],
            context=ctx,
        )
        with ir.InsertionPoint.at_block_begin(mod.body):
            func_d.FuncOp(name=sym, type=func_type, visibility="private", loc=loc)

        # ---- 插入 func.call (在 matmul 之前) ----
        with ir.InsertionPoint(mm):
            call = func_d.CallOp([si32_result.type], sym, [X_si8, W_packed], loc=loc)

        # ---- 替换 si32 -> call result, 删死链 ----
        # 只 erase result 无 use 的 op; fill 常被多个 matmul 共享为 init buffer (uses>0),
        # erase 有 use 的 op 会 segfault, 留给 canonicalize。
        si32_result.replace_all_uses_with(call.results[0])
        kill = [fptosi_gen, mm, uitofp_gen, transpose, collapse, broadcast_gen, sitofp_gen]
        if fill is not None:
            kill.append(fill)
        for op in reversed(kill):
            n_uses = sum(1 for r in op.results for _ in r.uses)
            if n_uses == 0:
                try:
                    op.erase()
                except Exception:
                    pass  # 仍有隐式 use, 留给 canonicalize

    return len(matmuls)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default="checkpoints/bitnet_ternary_placeholder.mlir")
    p.add_argument("--out", default="checkpoints/bitnet_ternary_final.mlir")
    p.add_argument("--save-linalg", default="checkpoints/bitnet_ternary_linalg.mlir",
                   help="保存 L2 中间 linalg IR (调试)")
    args = p.parse_args()

    src = open(args.inp).read()
    ctx = ir.Context()
    torch_d.register_dialect(ctx)
    mod = ir.Module.parse(src, ctx)

    print(f"[L3] 加载 {args.inp}", file=sys.stderr)
    print(f"[L3] 跑 LINALG pipeline (in-memory, 注册 tm_tensor)...", file=sys.stderr)
    _module_lowering(False, False, OutputType.LINALG_ON_TENSORS, mod)

    if args.save_linalg:
        with open(args.save_linalg, "w") as f:
            f.write(str(mod))
        print(f"[L3] saved linalg intermediate: {args.save_linalg}", file=sys.stderr)

    print(f"[L3] 插 func.call @cim_launch_<idx> (替换 placeholder linalg.matmul)...", file=sys.stderr)
    n = lower_linalg_to_cim_call(mod)

    print(f"[L3] canonicalize + cse 清死链...", file=sys.stderr)
    run_pipeline_with_repro_report(mod, "builtin.module(canonicalize, cse)", "L3 canonicalize+cse")
    mod.operation.verify()
    print(f"[L3] final IR verify OK ✓", file=sys.stderr)

    out = str(mod)
    with open(args.out, "w") as f:
        f.write(out)

    n_call = out.count("call @cim_launch_")
    n_decl = out.count("func.func private @cim_launch_")
    n_linalg_mm = out.count("linalg.matmul")
    n_linalg_gen = out.count("linalg.generic")
    n_op = out.count("torch.operator")
    n_aten = out.count("torch.aten.")
    print(f"[L3] {n} linalg.matmul -> func.call @cim_launch_<idx>", file=sys.stderr)
    print(f"[L3] call={n_call}, 声明={n_decl}, linalg.matmul={n_linalg_mm} (应=0), "
          f"linalg.generic={n_linalg_gen}, torch.operator={n_op}, torch.aten={n_aten}",
          file=sys.stderr)
    print(f"[L3] saved: {args.out}", file=sys.stderr)

    ok = True
    if n_call != 37:
        print(f"[L3] FAIL: call={n_call} (应=37)", file=sys.stderr); ok = False
    if n_decl != 37:
        print(f"[L3] FAIL: 声明={n_decl} (应=37)", file=sys.stderr); ok = False
    if n_linalg_mm != 0:
        print(f"[L3] FAIL: 残留 linalg.matmul={n_linalg_mm} (应=0, placeholder 死链未清)", file=sys.stderr); ok = False
    if n_op != 0:
        print(f"[L3] FAIL: 残留 torch.operator={n_op}", file=sys.stderr); ok = False
    print(f"\n[L3] {'PASS ✓ (37 func.call stub 保留, placeholder 死链清)' if ok else 'FAIL ✗'}",
          file=sys.stderr)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
