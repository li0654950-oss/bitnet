#!/usr/bin/env python3
"""L7: LLVM IR -> .o (ExecutionEngine.dump_to_object_file, 可选)。

从 L1 产出 (placeholder.mlir) in-memory 跑 L2+L3+cim_call_to_memref+L4 -> LLVM mod,
ExecutionEngine JIT 编译并 dump object file。.o 含 main + @cim_launch external declare
(链接 cim_stub.so) + refbackend_consume callback。

dump_to_object_file 要求所有 symbol resolved: refbackend_consume_func_return_* 是
external (runtime callback), 须先 register_runtime 注册 (Python 占位), cim_launch 由
shared_libs 提供。JIT 执行 (L6) 已验证功能, L7 .o 仅作可链接产物 (可选)。
"""
import os
import sys
import argparse
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))

# torch_mlir_e2e_test 不在 pip wheel (torch_mlir dist-info 无 e2e_test 子模块), 需从
# torch-mlir 源码树导入 (refbackend L4 pipeline 用)。非 bitnet-cim 包范围, editable
# 安装解决不了, 保留此路径 hack。
_E2E_PATH = "/home/li/workspace/torch-mlir/projects/pt1/python"
if _E2E_PATH not in sys.path:
    sys.path.insert(0, _E2E_PATH)

from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d
from torch_mlir.fx import _module_lowering
from torch_mlir.compiler_utils import OutputType, run_pipeline_with_repro_report
from torch_mlir.execution_engine import ExecutionEngine


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default="checkpoints/bitnet_ternary_placeholder.mlir")
    p.add_argument("--so", default=os.path.join(HERE, "cim_stub.so"))
    p.add_argument("--runtime-so", default=os.path.join(HERE, "aot", "cim_runtime.so"),
                   help="C 版 consume runtime (.so, 提供 refbackend_consume_func_return_*, AOT 用)")
    p.add_argument("--out", default="checkpoints/bitnet_ternary.o")
    p.add_argument("--partition", default="checkpoints/bitnet_ternary_partition.json",
                   help="方案B: partition.json 元数据识别 qkv")
    args = p.parse_args()

    csl = _load("cim_stub_lower", os.path.join(HERE, "cim_stub_lower.py"))
    ll = _load("linalg_to_llvm_mod", os.path.join(HERE, "linalg_to_llvm.py"))
    src = open(args.inp).read()
    ctx = ir.Context()
    torch_d.register_dialect(ctx)
    ctx.enable_multithreading(False)
    mod = ir.Module.parse(src, ctx)
    _module_lowering(False, False, OutputType.LINALG_ON_TENSORS, mod)
    csl.lower_linalg_to_cim_call(mod, args.partition)
    run_pipeline_with_repro_report(mod, "builtin.module(canonicalize, cse)", "canon")
    ll.cim_call_to_memref(mod)
    from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend as rb
    pipeline = rb.lowering_pipeline(False).replace("func.func(buffer-deallocation-pipeline),", "")
    run_pipeline_with_repro_report(mod, pipeline, "L4")
    mod.operation.verify()

    print(f"[L7] LLVM mod OK, JIT 编译 + dump object...", file=sys.stderr)
    # consume 符号 (refbackend_consume_func_return_*) 由 cim_runtime.so C 版提供
    # (dump_to_object_file 要求 external symbol resolved, shared_libs 满足; 替代 JIT 的 register_runtime)
    ee = ExecutionEngine(mod, shared_libs=[args.so, args.runtime_so], enable_pic=True)
    ee.dump_to_object_file(args.out)
    print(f"[L7] saved: {args.out}", file=sys.stderr)
    ok = os.path.exists(args.out) and os.path.getsize(args.out) > 0
    print(f"\n[L7] {'PASS ✓ (.o 产出, 可链接 cim_stub.so)' if ok else 'FAIL ✗'}", file=sys.stderr)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
