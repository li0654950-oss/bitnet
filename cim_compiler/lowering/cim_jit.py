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

方案 A (系统级仿真): cim_stub.c 的 cim_launch_<idx> + cim_preload_init 通过 MMIO 驱动
  Python 纯硬件仿真器 (register_cim_hw_sim 注册 4 回调 shm/reg), 走真实 func.call + JIT 链路。
  真实硬件: cim_stub.c #define HW_REAL, MMIO 直接 volatile, 无 Python (架构就绪)。
"""
import os
import sys
import ctypes
import argparse
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# torch_mlir_e2e_test 不在 pip wheel (torch_mlir dist-info 无 e2e_test 子模块), 需从
# torch-mlir 源码树导入 (refbackend L4 pipeline 用)。非 bitnet-cim 包范围, editable
# 安装解决不了, 保留此路径 hack。
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
from cim_compiler.lowering.buffer_kind import (
    classify_buffer, KIND_W_PACKED, KIND_LMHEAD, KIND_INVFREQ, KIND_CAUSAL_MASK)

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


def build_llvm_mod(placeholder_path, partition_path=None):
    csl = _load("cim_stub_lower", os.path.join(HERE, "cim_stub_lower.py"))
    ll = _load("linalg_to_llvm_mod", os.path.join(HERE, "linalg_to_llvm.py"))
    src = open(placeholder_path).read()
    ctx = ir.Context()
    torch_d.register_dialect(ctx)
    ctx.enable_multithreading(False)
    mod = ir.Module.parse(src, ctx)
    _module_lowering(False, False, OutputType.LINALG_ON_TENSORS, mod)
    csl.lower_linalg_to_cim_call(mod, partition_path)
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


# ---- 方案 A: CIM 纯硬件仿真器 MMIO 回调 (cim_stub.c shm/reg -> hw_simulator) ----
# cim_stub.c 的 shm_write/shm_read/reg_write/reg_read 经此转发 hw_simulator.mmio_*
# (cim_launch_<idx> + cim_preload_init 通过 MMIO 驱动纯硬件, 硬件自己取指)
_SHM_WRITE_CB = ctypes.CFUNCTYPE(None, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64)
_SHM_READ_CB = ctypes.CFUNCTYPE(None, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64)
_REG_WRITE_CB = ctypes.CFUNCTYPE(None, ctypes.c_int64, ctypes.c_int64)
_REG_READ_CB = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int64)


def register_cim_hw_sim(so_path, sim):
    """ctypes 加载 cim_stub.so, 注册 4 个 MMIO 回调 -> sim.mmio_*。
    cim_stub.c 的 shm_write/shm_read/reg_write/reg_read 经此转发 hw_simulator (纯硬件)。
    返回 lib (持回调引用, 调用方须保持存活, 防 GC 段错误)。"""
    lib = ctypes.CDLL(so_path)

    @_SHM_WRITE_CB
    def cb_shm_write(addr, ptr, n):
        sim.mmio_shm_write(addr, ptr, n)

    @_SHM_READ_CB
    def cb_shm_read(addr, ptr, n):
        sim.mmio_shm_read(addr, ptr, n)

    @_REG_WRITE_CB
    def cb_reg_write(reg, val):
        sim.mmio_reg_write(reg, val)

    @_REG_READ_CB
    def cb_reg_read(reg):
        return sim.mmio_reg_read(reg)

    lib.register_cim_hw_sim(cb_shm_write, cb_shm_read, cb_reg_write, cb_reg_read)
    lib._cbs = (cb_shm_write, cb_shm_read, cb_reg_write, cb_reg_read)  # 持引用防 GC
    # forward.bin (按 idx 索引) + preload.bin (自包含) 加载到 cim_stub
    lib.cim_load_forward.argtypes = [ctypes.c_char_p]
    lib.cim_load_forward.restype = None
    lib.cim_preload_init.argtypes = [ctypes.c_char_p]
    lib.cim_preload_init.restype = None
    return lib


DEFAULT_PT2 = os.path.join(REPO, "checkpoints", "bitnet_ternary.pt2")
_ep_cache = {}


def _resolve_attr(model, target):
    """m.layers.0.attn.q_proj.w_packed -> model.layers[0].attn.q_proj.w_packed"""
    obj = model
    for p in target.split('.'):
        if p == 'm':
            continue
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj


def build_inputs(model, idx, exported_program=None):
    """[P0-2] 签名驱动: 从 ExportedProgram.graph_signature.input_specs 按 placeholder 顺序构造 input。

    跳过 PARAMETER (constant-fold 内嵌), BUFFER 按 target 名解析 model 属性,
    USER_INPUT 用 idx。任意 attn/mlp 拓扑 + 任意 n_layer 自动适配 (不硬编码子结构)。
    """
    if exported_program is None:
        exported_program = _ep_cache.get("default")
        if exported_program is None:
            try:
                from cim_compiler.export import cim_op  # 注册 cim::matmul (反序列化 .pt2 需要)
            except ImportError:
                pass
            exported_program = torch.export.load(DEFAULT_PT2)
            _ep_cache["default"] = exported_program
    args = []
    for s in exported_program.graph_signature.input_specs:
        kind = s.kind.name
        if kind == 'PARAMETER':
            continue  # constant-fold 内嵌, 不作 runtime input
        if kind == 'USER_INPUT':
            args.append(np.ascontiguousarray(idx.numpy()))
            continue
        # BUFFER: target 名解析 model 属性, classify_buffer 分类 kind 定类型
        t = s.target
        val = _resolve_attr(model, t)
        val = val.numpy() if hasattr(val, 'numpy') else np.asarray(val)
        kind = classify_buffer(t)
        if kind in (KIND_W_PACKED, KIND_LMHEAD):    # w_packed (含 lm_head) uint8 -> i8 view
            args.append(np.ascontiguousarray(val).view(np.int8))
        elif kind == KIND_INVFREQ:
            args.append(val.astype(np.float32))
        elif kind == KIND_CAUSAL_MASK:
            args.append(val.astype(np.bool_))
        else:
            args.append(np.ascontiguousarray(val))
    return args


def build_inputs_kv(model, idx, k_caches, v_caches, cos, sin, exported_program=None):
    """[接入点③] 增量 KV cache build_inputs: USER_INPUT(idx, k_caches, v_caches, cos, sin) + BUFFER。

    增量 .pt2 的 input_specs: PARAMETER(跳过) + BUFFER(inv_freq/causal_mask/w_packed) +
    USER_INPUT(idx, k_caches, v_caches, cos, sin, 按 forward 参数顺序填)。
    """
    if exported_program is None:
        exported_program = _ep_cache.get("kv")
        if exported_program is None:
            try:
                from cim_compiler.export import cim_op  # 注册 cim::matmul (反序列化 .pt2 需要)
            except ImportError:
                pass
            exported_program = torch.export.load(os.path.join(REPO, "checkpoints/bitnet_ternary_kv.pt2"))
            _ep_cache["kv"] = exported_program
    ui = [np.ascontiguousarray(idx), np.ascontiguousarray(k_caches),
          np.ascontiguousarray(v_caches), np.ascontiguousarray(cos), np.ascontiguousarray(sin)]
    args = []
    ui_idx = 0
    for s in exported_program.graph_signature.input_specs:
        kind = s.kind.name
        if kind == 'PARAMETER':
            continue
        if kind == 'USER_INPUT':
            args.append(ui[ui_idx]); ui_idx += 1
            continue
        t = s.target
        val = _resolve_attr(model, t)
        val = val.numpy() if hasattr(val, 'numpy') else np.asarray(val)
        kind = classify_buffer(t)
        if kind in (KIND_W_PACKED, KIND_LMHEAD):
            args.append(np.ascontiguousarray(val).view(np.int8))
        elif kind == KIND_INVFREQ:
            args.append(val.astype(np.float32))
        elif kind == KIND_CAUSAL_MASK:
            args.append(val.astype(np.bool_))
        else:
            args.append(np.ascontiguousarray(val))
    return args


def pick_token(logits, temp, top_k, rng):
    """从 logits 选 token: temp<=0 greedy argmax; 否则 softmax(temp)+top_k+multinomial。"""
    l = np.asarray(logits, dtype=np.float32).copy()
    if temp is None or temp <= 0.0:
        return int(l.argmax())
    if top_k is not None and 0 < top_k < l.shape[-1]:
        kth = np.partition(l, -top_k)[-top_k]   # top_k 大里的最小值 = 阈值
        l = np.where(l >= kth, l, -np.inf)
    l = l / temp
    l = l - l.max()                             # softmax 数值稳定
    p = np.exp(l); p /= p.sum()
    return int(rng.choice(p.shape[-1], p=p))


def generate_kv(invoker, model, idx0, n, exported_program=None, block_size=256, temp=0.0, top_k=40, rng=None):
    """[接入点③] JIT 增量 KV cache 生成: prefill prompt 逐 token + decode 单 token (M=1)。

    每步 idx[1] + cache + cos/sin(pos=cache_len), invoke main -> (logits, new_k, new_v),
    cache 更新 (new_k -> k_caches)。全程 CIM matmul M=1 (decode 单 token), O(n²)->O(n)。
    """
    n_layer = len(model.layers)
    attn0 = model.layers[0].attn
    n_kv, head_dim = attn0.n_kv, attn0.head_dim
    inv_freq = attn0.inv_freq.numpy().astype(np.float32)
    k_caches = np.zeros([n_layer, 1, 0, n_kv, head_dim], dtype=np.float32)   # T=0 首步空 cache
    v_caches = np.zeros([n_layer, 1, 0, n_kv, head_dim], dtype=np.float32)
    tokens = idx0[0].tolist()

    def step(tok):
        nonlocal k_caches, v_caches
        idx = np.array([[tok]], dtype=np.int64)
        pos = np.array([float(k_caches.shape[2])], dtype=np.float32)     # cache_len = 当前 T
        freqs = np.einsum("t,d->td", pos, inv_freq)
        cos = np.cos(freqs)[None, :, None, :].astype(np.float32)         # [1,1,1,hd/2]
        sin = np.sin(freqs)[None, :, None, :].astype(np.float32)
        inputs = build_inputs_kv(model, idx, k_caches, v_caches, cos, sin, exported_program)
        logits, k_caches, v_caches = invoker.invoke("main", *inputs)
        return logits

    logits = None
    for tok in tokens:                    # prefill prompt 逐 token (建 cache)
        logits = step(tok)
    for _ in range(n):                    # decode n token (M=1, 用 cache)
        nxt = pick_token(np.asarray(logits[0, 0]), temp, top_k, rng)
        tokens.append(nxt)
        logits = step(nxt)
    return tokens


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default="checkpoints/bitnet_ternary_placeholder.mlir")
    p.add_argument("--ternary", default="checkpoints/bitnet_shakespeare_char_ternary.pt")
    p.add_argument("--so", default=os.path.join(HERE, "cim_stub.so"))
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--sim", action="store_true",
                   help="方案 A: 注册 CIM 指令级仿真器回调 (否则用 cim_stub CPU 算 fallback)")
    # [P0-3] 模型结构参数 (任意规模, 须与 ternary.pt 一致)
    p.add_argument("--vocab_size", type=int, default=65)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_kv_head", type=int, default=4)
    p.add_argument("--ffn_dim", type=int, default=1664)
    p.add_argument("--partition", default="checkpoints/bitnet_ternary_partition.json",
                   help="方案B: partition.json 元数据识别 qkv (不传则回退 shape 启发式)")
    args = p.parse_args()

    print("[L6] 构建 LLVM mod (in-memory L2+L3+L4)...", file=sys.stderr)
    mod = build_llvm_mod(args.inp, args.partition)
    print("[L6] LLVM mod OK", file=sys.stderr)

    invoker = CIMInvoker(mod, args.so)
    print(f"[L6] ExecutionEngine + cim_stub.so 加载完成", file=sys.stderr)

    sim_lib = None
    if args.sim:
        from cim_compiler.cimres.hw_simulator import HwCimSimulator
        sim = HwCimSimulator()                       # 纯硬件 (无参数, 不读 IR/weights)
        sim_lib = register_cim_hw_sim(args.so, sim)  # 注册 4 个 MMIO 回调
        fwd = os.path.join(REPO, "cim_compiler/cimres/checkpoints/forward.bin")
        pre = os.path.join(REPO, "cim_compiler/cimres/checkpoints/preload.bin")
        sim_lib.cim_load_forward(fwd.encode())       # forward.bin 按 idx 索引 (cim_launch 查)
        sim_lib.cim_preload_init(pre.encode())       # Preload: 读 preload.bin MMIO 驱动 (一次性)
        print(f"[L6] 纯硬件 CIM 仿真器 MMIO 回调已注册 + cim_preload_init "
              f"({len(sim.macros.macro)} Macro 预载)", file=sys.stderr)

    from cim_compiler.export.inference_model import build_inference_model
    model = build_inference_model(args.ternary, vocab_size=args.vocab_size,
                                  d_model=args.d_model, block_size=args.block_size,
                                  n_layer=args.n_layer, n_head=args.n_head,
                                  n_kv_head=args.n_kv_head, ffn_dim=args.ffn_dim)
    idx = torch.zeros(1, args.T, dtype=torch.long)
    inputs = build_inputs(model, idx)
    print(f"[L6] {len(inputs)} input 构造完成, invoke main (T={args.T})...", file=sys.stderr)

    logits = invoker.invoke("main", *inputs)
    if args.sim:
        st = sim.stats_snapshot()
        print(f"[L6] cim_cycle={st['cim_cycle']}, mmio_cycle={st['mmio_cycle']} "
              f"(n_macro={st['n_macro']})", file=sys.stderr)
    ref = model(idx)[0].detach().numpy()

    logits = np.asarray(logits)
    print(f"[L6] JIT logits shape={logits.shape}, ref={ref.shape}", file=sys.stderr)
    diff = np.abs(logits.astype(np.float64) - ref.astype(np.float64))
    print(f"[L6] max abs diff = {diff.max():.4f}, mean = {diff.mean():.4f}", file=sys.stderr)
    ok = diff.max() < 1.0
    mode = "方案A 纯硬件 MMIO 仿真器" if args.sim else "cim_stub CPU 算 fallback"
    print(f"\n[L6] {'PASS ✓ (' + mode + ', func.call 正确接入)' if ok else 'FAIL ✗'}",
          file=sys.stderr)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
