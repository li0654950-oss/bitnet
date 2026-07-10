#!/usr/bin/env python3
"""L6+: JIT 自回归生成 token (降级完的程序跑推理出 token)。

用 L6 的 CIMInvoker (ExecutionEngine + cim_stub.so) 自回归生成:
  循环: logits = invoke(main, 权重..., idx); next = argmax(logits[0,-1]); idx = cat(idx, next)
对比 PyTorch reference 自回归生成, 验证降级程序能正确出 token。

用法:
  python cim_compiler/lowering/generate.py --n 12
"""
import os
import sys
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


def _load_cim_jit():
    spec = importlib.util.spec_from_file_location("cim_jit", os.path.join(HERE, "cim_jit.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def jit_generate(invoker, model, prompt, n, block_size):
    idx = torch.tensor([[prompt]], dtype=torch.long)
    tokens = [prompt]
    for i in range(n):
        if idx.shape[1] >= block_size:
            idx = idx[:, -block_size:]
        inputs = _build_inputs(model, 6, idx)
        logits = np.asarray(invoker.invoke("main", *inputs))
        nxt = int(np.argmax(logits[0, -1]))
        tokens.append(nxt)
        idx = torch.cat([idx, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
    return tokens


def ref_generate(model, prompt, n, block_size):
    idx = torch.tensor([[prompt]], dtype=torch.long)
    tokens = [prompt]
    with torch.no_grad():
        for i in range(n):
            if idx.shape[1] >= block_size:
                idx = idx[:, -block_size:]
            logits = model(idx)[0]
            nxt = int(logits[0, -1].argmax())
            tokens.append(nxt)
            idx = torch.cat([idx, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
    return tokens


def _build_weights(model, n_layer):
    """权重 memref (49 个, 不含 idx), 复用避免每步重建。"""
    w = []
    for li in range(n_layer):
        a = model.layers[li].attn
        m = model.layers[li].mlp
        w.append(a.inv_freq.numpy().astype(np.float32))
        w.append(a.causal_mask.numpy().astype(np.bool_))
        for x in [a.q_proj.w_packed, a.k_proj.w_packed, a.v_proj.w_packed,
                  a.o_proj.w_packed, m.fc1.w_packed, m.fc2.w_packed]:
            w.append(np.ascontiguousarray(x.numpy()).view(np.int8))
    w.append(np.ascontiguousarray(model.lm_head.w_packed.numpy()).view(np.int8))
    return w


def _build_inputs(model, n_layer, idx):
    """复用 cim_jit.build_inputs (每步重建权重, n 小可接受)。"""
    import cim_jit as _cj  # 已由 _load_cim_jit 注入
    return _cj.build_inputs(model, n_layer, idx)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default="checkpoints/bitnet_ternary_placeholder.mlir")
    p.add_argument("--ternary", default="checkpoints/bitnet_shakespeare_char_ternary.pt")
    p.add_argument("--so", default=os.path.join(HERE, "cim_stub.so"))
    p.add_argument("--prompt", type=int, default=0, help="prompt token id")
    p.add_argument("--n", type=int, default=12, help="生成 token 数")
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--sim", action="store_true",
                   help="方案 A: 注册 CIM 指令级仿真器回调 (否则 cim_stub CPU 算 fallback)")
    args = p.parse_args()

    cim_jit = _load_cim_jit()
    sys.modules["cim_jit"] = cim_jit  # 供 _build_inputs import

    print("[gen] 构建 LLVM mod (in-memory L2+L3+L4)...", file=sys.stderr)
    mod = cim_jit.build_llvm_mod(args.inp)
    invoker = cim_jit.CIMInvoker(mod, args.so)
    print("[gen] ExecutionEngine + cim_stub.so 就绪", file=sys.stderr)

    sim_lib = None
    if args.sim:
        from cim_compiler.cimres.hw_simulator import HwCimSimulator
        sim = HwCimSimulator()                       # 纯硬件 (无参数)
        sim_lib = cim_jit.register_cim_hw_sim(args.so, sim)  # 须保存 (持回调引用防 GC 段错误)
        fwd = os.path.join(REPO, "cim_compiler/cimres/checkpoints/forward.bin")
        pre = os.path.join(REPO, "cim_compiler/cimres/checkpoints/preload.bin")
        sim_lib.cim_load_forward(fwd.encode())       # forward.bin 按 idx 索引
        sim_lib.cim_preload_init(pre.encode())       # Preload: 读 preload.bin MMIO 驱动 (一次性)
        print(f"[gen] 纯硬件 CIM 仿真器 MMIO 回调已注册 + cim_preload_init "
              f"({len(sim.macros.macro)} Macro 预载)", file=sys.stderr)

    from inference_model import build_inference_model
    model = build_inference_model(args.ternary, vocab_size=65)

    print(f"[gen] JIT 自回归生成 {args.n} token (prompt={args.prompt})...", file=sys.stderr)
    jit_tokens = jit_generate(invoker, model, args.prompt, args.n, args.block_size)
    print(f"[gen] PyTorch reference 生成...", file=sys.stderr)
    ref_tokens = ref_generate(model, args.prompt, args.n, args.block_size)

    print(f"\nJIT tokens:      {jit_tokens}")
    print(f"PyTorch tokens:  {ref_tokens}")
    match = jit_tokens == ref_tokens
    print(f"完全一致: {'✓ YES' if match else '✗ NO'}")
    sys.exit(0 if match else 1)


if __name__ == "__main__":
    main()
