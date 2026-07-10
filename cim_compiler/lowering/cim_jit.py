#!/usr/bin/env python3
"""L6: 端到端 JIT 执行 + 数值验证 (ExecutionEngine + C stub .so)。

从 L1 产出 (placeholder.mlir) in-memory 跑 L2+L3+cim_call_to_memref+L4 (refbackend
pipeline -> LLVM IR), 用 ExecutionEngine JIT 执行 main:
  - shared_libs=[cim_stub.so]: 提供 37 个 @cim_launch_<idx> (C, 返回 struct by value,
    ctypes callback 不支持返回 Structure, 故用 C stub)
  - register_runtime @refbackend_consume_func_return_*: return callback (main 用 munge
    calling convention, return 走 callback)
  - main input (50 个 unranked memref): 每层 [inv_freq, causal_mask, q/k/v/o/fc1/fc2
    w_packed] x6 + lm_head.w_packed + idx (gamma/norm 等 constant-fold 内嵌)
  - invoke main -> logits, 对比 PyTorch reference (model(idx)[0])

main 的 w_packed 是 i8 (MLIR 无 ui8), C stub 按 uint8 解释。

方案 A (系统级仿真): cim_stub.c 的 cim_launch_<idx> 调 Python CIM 指令级仿真器
  (register_cim_sim_callback), 走真实 func.call + JIT 链路, 与现有架构一致。
"""
import os
import sys
import ctypes
import argparse
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
_EXPORT_DIR = os.path.join(REPO, "cim_compiler", "export")
if _EXPORT_DIR not in sys.path:
    sys.path.insert(0, _EXPORT_DIR)
_E2E_PATH = "/home/li/workspace/torch-mlir/projects/pt1/python"
if _E2E_PATH not in sys.path:
    sys.path.insert(0, _E2E_PATH)

import numpy as np
import torch
from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d
from torch_mlir.fx import _module_lowering
from torch_mlir.compiler_utils import OutputType, run_pipeline_with_repro_report
from torch_mlir.execution_engine import ExecutionEngine
from torch_mlir.runtime import unranked_memref_to_numpy, get_unranked_memref_descriptor, UnrankedMemRefDescriptor

# refbackend return callback 类型映射 (复制自 refbackend.py)
CONSUME_PREFIX = "refbackend_consume_func_return_"
ELEMENTAL = {"i1": ctypes.c_bool, "i8": ctypes.c_byte, "i64": ctypes.c_int,
             "f32": ctypes.c_float, "f64": ctypes.c_double}
MEMREF_DTYPE = {"mrf16": np.float16, "mrf32": np.float32, "mrf64": np.float64,
                "mri1": np.bool_, "mri8": np.int8, "mri16": np.int16,
                "mri32": np.int32, "mri64": np.int64}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_llvm_mod(placeholder_path):
    csl = _load("cim_stub_lower", os.path.join(HERE, "cim_stub_lower.py"))
    ll = _load("linalg_to_llvm_mod", os.path.join(HERE, "linalg_to_llvm.py"))
    src = open(placeholder_path).read()
    ctx = ir.Context()
    torch_d.register_dialect(ctx)
    ctx.enable_multithreading(False)
    mod = ir.Module.parse(src, ctx)
    _module_lowering(False, False, OutputType.LINALG_ON_TENSORS, mod)
    csl.lower_linalg_to_cim_call(mod)
    run_pipeline_with_repro_report(mod, "builtin.module(canonicalize, cse)", "canon")
    ll.cim_call_to_memref(mod)
    from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend as rb
    pipeline = rb.lowering_pipeline(False).replace("func.func(buffer-deallocation-pipeline),", "")
    run_pipeline_with_repro_report(mod, pipeline, "L4")
    mod.operation.verify()
    return mod


class CIMInvoker:
    """ExecutionEngine + cim_stub.so + return callback (复制 RefBackendInvoker, 加 shared_libs)。"""
    def __init__(self, module, so_path):
        self.ee = ExecutionEngine(module, shared_libs=[so_path])
        self.result = None
        for f in module.body:
            nm_attr = f.attributes.get("sym_name")
            if nm_attr is None:
                continue
            nm = str(nm_attr).strip('"')
            if nm.startswith(CONSUME_PREFIX):
                self._register_return(nm)

    def _register_return(self, func_name):
        ret_types = func_name[len(CONSUME_PREFIX):].split("_")
        ctypes_arg = [None]
        for t in ret_types:
            if t in ELEMENTAL:
                ctypes_arg.append(ELEMENTAL[t])
            elif t in MEMREF_DTYPE:
                ctypes_arg.append(ctypes.POINTER(UnrankedMemRefDescriptor))
            else:
                raise ValueError(f"unsupported return type: {t}")
        ctype = ctypes.CFUNCTYPE(*ctypes_arg)

        def consume(*args):
            self.result = tuple(
                arg if t in ELEMENTAL else unranked_memref_to_numpy(arg, MEMREF_DTYPE[t])
                for arg, t in zip(args, ret_types))
            if len(self.result) == 1:
                self.result = self.result[0]
        self.ee.register_runtime(func_name, ctype(consume))

    def invoke(self, function_name, *args):
        ffi_args = []
        for arg in args:
            ffi_args.append(ctypes.pointer(ctypes.pointer(get_unranked_memref_descriptor(arg))))
        self.ee.invoke(function_name, *ffi_args)
        result = self.result
        self.result = None
        return result


# ---- 方案 A: CIM 指令级仿真器回调 (cim_stub.c cim_launch_<idx> -> Python) ----
# 回调签名匹配 cim_stub.c: void(int idx, int8* x, int64 M, int64 K, uint8* w, int64 N, int64 K4, int32* out)
_CIM_SIM_CB = ctypes.CFUNCTYPE(None, ctypes.c_int,
                               ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                               ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                               ctypes.c_void_p)


def register_cim_sim_callback(so_path, sim):
    """ctypes 加载 cim_stub.so, 注册 Python 回调 py_cim_sim -> sim.simulate。
    cim_stub.c 的 cim_launch_<idx> 调此回调 (传 x/w 指针+idx), 回调写 result buffer。
    返回 lib (持有回调引用, 调用方须保持存活)。"""
    lib = ctypes.CDLL(so_path)

    @_CIM_SIM_CB
    def py_cim_sim(idx, x_ptr, M, K, w_ptr, N, K4, out_ptr):
        # 从 C 指针读 x[M,K] int8 + w[N,K4] uint8 (main 的 w_packed, i8->uint8)
        x = np.ctypeslib.as_array(ctypes.cast(x_ptr, ctypes.POINTER(ctypes.c_int8)), (M, K))
        w = np.ctypeslib.as_array(ctypes.cast(w_ptr, ctypes.POINTER(ctypes.c_uint8)), (N, K4))
        acc = sim.simulate(idx, x, w)                       # [M,N] int32 (指令级, idx->Macro)
        out = np.ctypeslib.as_array(ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_int32)), (M, N))
        out[:] = acc                                        # 写回 C malloc 的 result buffer

    lib.register_cim_simulator(py_cim_sim)
    lib._py_cim_sim = py_cim_sim                            # 保持回调对象存活 (防 GC)
    return lib


def build_inputs(model, n_layer, idx):
    """50 个 numpy input (按 main signature 顺序)。

    每层: [inv_freq f32<32>, causal_mask i1<1x1xBxB>, q/k/v/o/fc1/fc2 w_packed i8]
    + lm_head.w_packed + idx。w_packed uint8 -> view int8 (MLIR i8)。
    idx: torch.long [1, T] (自回归时可变)。
    """
    args = []
    for li in range(n_layer):
        a = model.layers[li].attn
        m = model.layers[li].mlp
        args.append(a.inv_freq.numpy().astype(np.float32))
        args.append(a.causal_mask.numpy().astype(np.bool_))
        for w in [a.q_proj.w_packed, a.k_proj.w_packed, a.v_proj.w_packed,
                  a.o_proj.w_packed, m.fc1.w_packed, m.fc2.w_packed]:
            args.append(np.ascontiguousarray(w.numpy()).view(np.int8))
    args.append(np.ascontiguousarray(model.lm_head.w_packed.numpy()).view(np.int8))
    args.append(np.ascontiguousarray(idx.numpy()))
    return args


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default="checkpoints/bitnet_ternary_placeholder.mlir")
    p.add_argument("--ternary", default="checkpoints/bitnet_shakespeare_char_ternary.pt")
    p.add_argument("--so", default=os.path.join(HERE, "cim_stub.so"))
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--sim", action="store_true",
                   help="方案 A: 注册 CIM 指令级仿真器回调 (否则用 cim_stub CPU 算 fallback)")
    args = p.parse_args()

    print("[L6] 构建 LLVM mod (in-memory L2+L3+L4)...", file=sys.stderr)
    mod = build_llvm_mod(args.inp)
    print("[L6] LLVM mod OK", file=sys.stderr)

    invoker = CIMInvoker(mod, args.so)
    print(f"[L6] ExecutionEngine + cim_stub.so 加载完成", file=sys.stderr)

    sim_lib = None
    if args.sim:
        from cim_compiler.cimres.hw_simulator import HwCimSimulator
        sim = HwCimSimulator(
            os.path.join(REPO, "cim_compiler/cimres/checkpoints/bitnet_ternary_cimres_placed.mlir"),
            os.path.join(REPO, "checkpoints/bitnet_ternary_weights.bin"),
            os.path.join(REPO, "checkpoints/bitnet_ternary_partition.json"),
        )
        sim.preload_phase()  # Preload Phase: PROG_WGT 取指预载 Macro (硬件级)
        sim_lib = register_cim_sim_callback(args.so, sim)
        print(f"[L6] 硬件级 CIM 仿真器回调已注册 (方案 A, preload_phase {len(sim.macros.macro)} Macro)",
              file=sys.stderr)

    from inference_model import build_inference_model
    model = build_inference_model(args.ternary, vocab_size=65)
    idx = torch.zeros(1, args.T, dtype=torch.long)
    inputs = build_inputs(model, 6, idx)
    print(f"[L6] {len(inputs)} input 构造完成, invoke main (T={args.T})...", file=sys.stderr)

    logits = invoker.invoke("main", *inputs)
    ref = model(idx)[0].detach().numpy()

    logits = np.asarray(logits)
    print(f"[L6] JIT logits shape={logits.shape}, ref={ref.shape}", file=sys.stderr)
    diff = np.abs(logits.astype(np.float64) - ref.astype(np.float64))
    print(f"[L6] max abs diff = {diff.max():.4f}, mean = {diff.mean():.4f}", file=sys.stderr)
    ok = diff.max() < 1.0
    mode = "方案A CIM 指令级仿真器" if args.sim else "cim_stub CPU 算 fallback"
    print(f"\n[L6] {'PASS ✓ (' + mode + ', func.call 正确接入)' if ok else 'FAIL ✗'}",
          file=sys.stderr)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
