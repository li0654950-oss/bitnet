#!/usr/bin/env python3
"""L4: linalg+tm_tensor -> LLVM IR (refbackend pipeline + cim_call_to_memref)。

从 L1 产出 (placeholder.mlir) in-memory 跑 L2 (LINALG pipeline) + L3 (插 func.call
@cim_launch) + cim_call_to_memref (转 memref signature) 后, 跑 refbackend pipeline:
  tm-tensor-bufferize -> tm-tensor-to-loops (attention 降级)
  convert-linalg-to-loops (CPU linalg 降级)
  one-shot-bufferize (tensor -> memref)
  refback-munge-calling-conventions (main -> callback calling convention)
  convert-{arith,math,func,cf,...}-to-llvm -> LLVM dialect

关键处理:
1. cim_call_to_memref: @cim_launch signature tensor->memref + func.call 加
   bufferization.to_buffer (args) / to_tensor {restrict} (result), 避开 one-shot
   bufferize-function-boundaries 对 external dynamic-shape tensor func 的 segfault。
2. refbackend pipeline 去 buffer-deallocation-pipeline: 该 pass 在 external func 上
   产生不可降级的 bufferization.dealloc。one-shot (copy-before-write) 已避免 inplace,
   无需 dealloc。
3. 保留 refback-munge-calling-conventions: main 用 return callback (RefBackendInvoker
   友好), @cim_launch external 保持 ranked-memref 展开 calling convention (L5 stub 匹配)。

func.call @cim_launch_<idx> 作为 external symbol 保留 (LLVM IR 里是 llvm.func declare +
llvm.call)。@cim_launch 最终 calling convention: (2D memref X, 2D memref W) -> 2D memref
result, 每个 2D memref 展开成 (allocated_ptr, aligned_ptr, offset, size0, size1,
stride0, stride1) 7 参数。

用法:
  python cim_compiler/lowering/linalg_to_llvm.py
  python cim_compiler/lowering/linalg_to_llvm.py --in checkpoints/bitnet_ternary_placeholder.mlir \\
    --out checkpoints/bitnet_ternary_llvm.mlir
"""
import os
import sys
import argparse
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
_E2E_PATH = "/home/li/workspace/torch-mlir/projects/pt1/python"
if _E2E_PATH not in sys.path:
    sys.path.insert(0, _E2E_PATH)

from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d, bufferization as buf_d, func as func_d
from torch_mlir.fx import _module_lowering
from torch_mlir.compiler_utils import OutputType, run_pipeline_with_repro_report


def _load_cim_stub_lower():
    spec = importlib.util.spec_from_file_location(
        "cim_stub_lower", os.path.join(HERE, "cim_stub_lower.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cim_call_to_memref(mod):
    """@cim_launch FuncOp signature tensor->memref + func.call 加 to_buffer/to_tensor cast。

    避开 one-shot bufferize-function-boundaries 对 external dynamic-shape tensor func 的
    segfault。to_tensor 用 restrict=True (One-Shot Analysis 要求)。
    """
    ctx = mod.context
    calls = []

    def find(op):
        if str(op.name) == "func.call":
            ca = op.attributes.get("callee")
            callee = ca.value if hasattr(ca, "value") else str(ca).strip('"')
            if callee.startswith("cim_launch"):  # S6: cim_launch + cim_launch_qkv
                calls.append(op)
        return ir.WalkResult.ADVANCE

    mod.operation.walk(find)

    with ctx:
        # declaration: tensor -> memref
        for f in list(mod.body):
            na = f.attributes.get("sym_name")
            if na is None:
                continue
            nm = str(na).strip('"')
            if "cim_launch" in nm:
                new_ft = ir.Type.parse(
                    str(f.attributes["function_type"].value).replace("tensor", "memref"), ctx)
                f.attributes["function_type"] = ir.TypeAttr.get(new_ft)
        # func.call: args to_buffer, result to_tensor {restrict}
        for call in calls:
            loc = call.location
            ca = call.attributes["callee"]
            callee = ca.value if hasattr(ca, "value") else str(ca).strip('"')
            oat = [str(o.type) for o in call.operands]
            ort = [str(r.type) for r in call.results]
            with ir.InsertionPoint(call):
                ba = []
                for o, at in zip(call.operands, oat):
                    if "tensor" in at:
                        ba.append(buf_d.ToBufferOp(ir.Type.parse(at.replace("tensor", "memref"), ctx), o, loc=loc).results[0])
                    else:
                        ba.append(o)  # [A1] i64 idx 等标量直接传, 不 to_buffer
                mrt = [ir.Type.parse(rt.replace("tensor", "memref"), ctx) for rt in ort]
                nc = func_d.CallOp(mrt, callee, ba, loc=loc)
                tr = [buf_d.ToTensorOp(ir.Type.parse(rt, ctx), mr, restrict=True, loc=loc).results[0]
                      for mr, rt in zip(nc.results, ort)]
            for old, new in zip(call.results, tr):
                old.replace_all_uses_with(new)
            call.erase()
    return len(calls)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default="checkpoints/bitnet_ternary_placeholder.mlir")
    p.add_argument("--out", default="checkpoints/bitnet_ternary_llvm.mlir")
    p.add_argument("--partition", default="checkpoints/bitnet_ternary_partition.json",
                   help="方案B: partition.json 元数据识别 qkv")
    args = p.parse_args()

    csl = _load_cim_stub_lower()

    src = open(args.inp).read()
    ctx = ir.Context()
    torch_d.register_dialect(ctx)
    ctx.enable_multithreading(False)  # 禁多线程, 避免 diagnostic quirk
    mod = ir.Module.parse(src, ctx)

    print(f"[L4] 加载 {args.inp}", file=sys.stderr)
    print(f"[L4] L2: LINALG pipeline (in-memory)...", file=sys.stderr)
    _module_lowering(False, False, OutputType.LINALG_ON_TENSORS, mod)

    print(f"[L4] L3: 插 func.call @cim_launch + canonicalize...", file=sys.stderr)
    n = csl.lower_linalg_to_cim_call(mod, args.partition)
    run_pipeline_with_repro_report(mod, "builtin.module(canonicalize, cse)", "L3 canon+cse")

    print(f"[L4] cim_call_to_memref: {cim_call_to_memref(mod)} func.call 转 memref", file=sys.stderr)

    print(f"[L4] L4: refbackend pipeline (去 buffer-dealloc, linalg+tm_tensor -> LLVM)...", file=sys.stderr)
    from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend as rb
    pipeline = rb.lowering_pipeline(False).replace("func.func(buffer-deallocation-pipeline),", "")
    run_pipeline_with_repro_report(mod, pipeline, "L4 refbackend linalg->LLVM")
    mod.operation.verify()
    print(f"[L4] LLVM IR verify OK ✓", file=sys.stderr)

    out = str(mod)
    with open(args.out, "w") as f:
        f.write(out)

    n_call = out.count("call @cim_launch")
    n_decl = out.count("llvm.func @cim_launch") + out.count("llvm.func private @cim_launch")
    n_llvm = out.count("llvm.")
    n_generic = out.count("linalg.generic")
    n_tm = out.count("tm_tensor.")
    print(f"[L4] {n} func.call @cim_launch (L3) + refbackend pipeline", file=sys.stderr)
    print(f"[L4] call @cim_launch={n_call}, 声明={n_decl}, llvm.*={n_llvm}, "
          f"残留 linalg.generic={n_generic}, tm_tensor={n_tm}", file=sys.stderr)
    print(f"[L4] saved: {args.out}", file=sys.stderr)

    ok = (n_call == 25 and n_generic == 0 and n_tm == 0)
    print(f"\n[L4] {'PASS ✓ (LLVM IR, 25 func.call external stub 保留: 6 qkv + 19 单)' if ok else 'FAIL ✗'}",
          file=sys.stderr)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
