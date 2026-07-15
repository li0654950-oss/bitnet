#!/usr/bin/env python3
"""系统级仿真文本生成: 文本 prompt -> JIT 仿真自回归 -> 输出文本。

复用 generate.py 的 CIM 仿真链路 (cim_jit + hw_simulator --sim), 但:
  - prompt 为文本 (CharTokenizer encode 成 token 序列作初始 context)
  - 输出 decode 成文本
  - 同时跑 PyTorch reference (greedy) 对比

用法:
  conda run -n nanogpt-gpu python cim_compiler/lowering/run_sim_text.py --prompt "ROMEO:" --n 60
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


def generate(invoker, model, idx0, n, block_size, ref=False, temp=0.0, top_k=40, rng=None):
    """greedy/采样 自回归生成 n token, 返回 token 列表 (含 prompt)。"""
    import cim_jit as _cj
    idx = idx0.clone()
    tokens = idx[0].tolist()
    for i in range(n):
        if idx.shape[1] >= block_size:
            idx = idx[:, -block_size:]
        if ref:
            with torch.no_grad():
                logits = model(idx)[0]
        else:
            inputs = _build_inputs(model, idx)
            logits = np.asarray(invoker.invoke("main", *inputs))
        nxt = _cj.pick_token(logits[0, -1], temp, top_k, rng)
        tokens.append(nxt)
        idx = torch.cat([idx, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
        if (i + 1) % 10 == 0:
            print(f"  ...已生成 {i+1}/{n} token", file=sys.stderr)
    return tokens


def _build_inputs(model, idx):
    import cim_jit as _cj
    return _cj.build_inputs(model, idx)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default="checkpoints/bitnet_ternary_placeholder.mlir")
    p.add_argument("--ternary", default="checkpoints/bitnet_shakespeare_char_ternary.pt")
    p.add_argument("--so", default=os.path.join(HERE, "cim_stub.so"))
    p.add_argument("--prompt", default="ROMEO:", help="起始文本 (须为 vocab 内字符)")
    p.add_argument("--prompt-int", dest="prompt_int", type=int, default=None,
                   help="int prompt 模式 (跳过 tokenizer, 直接用 token id; 合并自 generate.py)")
    p.add_argument("--num-tokens", dest="n", type=int, default=60, help="生成 token 数")
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--partition", default="checkpoints/bitnet_ternary_partition.json",
                   help="方案B: partition.json 元数据识别 qkv")
    p.add_argument("--sim", action="store_true", default=True, help="用 hw_simulator MMIO 仿真")
    p.add_argument("--no-ref", action="store_true", help="跳过 PyTorch reference")
    p.add_argument("--kv", action="store_true", help="接入点③: 增量 KV cache (decode M=1, O(n²)->O(n))")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="采样温度; <=0 greedy (默认, token 级验证); >0 启用采样 (softmax+top_k+multinomial)")
    p.add_argument("--top_k", type=int, default=40, help="采样 top_k 截断")
    p.add_argument("--seed", type=int, default=0, help="采样随机种子 (JIT/ref 各用独立同 seed rng)")
    args = p.parse_args()
    if args.kv and args.inp == "checkpoints/bitnet_ternary_placeholder.mlir":
        args.inp = "checkpoints/bitnet_ternary_kv_placeholder.mlir"

    # ---- tokenizer (文本 prompt) 或 int prompt (合并自 generate.py) ----
    tok = None
    if args.prompt_int is not None:
        idx0 = torch.tensor([[args.prompt_int]], dtype=torch.long)
        print(f"[prompt] int mode: token id = {args.prompt_int}", file=sys.stderr)
    else:
        from bitnet.data_char import CharTokenizer
        tok = CharTokenizer()
        vocab = tok.itos
        bad = [c for c in args.prompt if c not in tok.stoi]
        if bad:
            print(f"[err] prompt 含 vocab 外字符: {bad}", file=sys.stderr)
            print(f"      vocab={sorted(vocab.values())}", file=sys.stderr)
            sys.exit(1)
        prompt_ids = tok.encode(args.prompt)
        if not prompt_ids:
            prompt_ids = [0]  # BOS
        idx0 = torch.tensor([prompt_ids], dtype=torch.long)
        print(f"[prompt] {args.prompt!r} -> {len(prompt_ids)} token: {prompt_ids}", file=sys.stderr)

    # ---- 构建 JIT + 仿真器 ----
    cim_jit = _load_cim_jit()
    sys.modules["cim_jit"] = cim_jit
    print("[run] 构建 LLVM mod (in-memory L2+L3+L4)...", file=sys.stderr)
    mod = cim_jit.build_llvm_mod(args.inp, args.partition)
    invoker = cim_jit.CIMInvoker(mod, args.so)
    print("[run] ExecutionEngine + cim_stub.so 就绪", file=sys.stderr)

    if args.sim:
        from cim_compiler.cimres.hw_simulator import HwCimSimulator
        sim = HwCimSimulator()
        sim_lib = cim_jit.register_cim_hw_sim(args.so, sim)
        fwd = os.path.join(REPO, "cim_compiler/cimres/checkpoints/forward.bin")
        pre = os.path.join(REPO, "cim_compiler/cimres/checkpoints/preload.bin")
        sim_lib.cim_load_forward(fwd.encode())
        sim_lib.cim_preload_init(pre.encode())
        print(f"[run] CIM 仿真器就绪 (MMIO + cycle 时序, {len(sim.macros.macro)} Macro 预载)",
              file=sys.stderr)

    from inference_model import build_inference_model
    model = build_inference_model(args.ternary, vocab_size=65)

    # ---- JIT 仿真生成 ----
    mode = "采样" if args.temperature > 0 else "greedy"
    rng_jit = np.random.default_rng(args.seed)
    rng_ref = np.random.default_rng(args.seed)      # 同 seed 独立 rng: logits 一致则采样一致
    print(f"[run] JIT 仿真自回归生成 {args.n} token... [{mode}]" + (" [KV cache 增量]" if args.kv else ""), file=sys.stderr)
    if args.kv:
        ep = torch.export.load(os.path.join(REPO, "checkpoints/bitnet_ternary_kv.pt2"))
        jit_tokens = cim_jit.generate_kv(invoker, model, idx0, args.n, ep, args.block_size,
                                         temp=args.temperature, top_k=args.top_k, rng=rng_jit)
    else:
        jit_tokens = generate(invoker, model, idx0, args.n, args.block_size, ref=False,
                              temp=args.temperature, top_k=args.top_k, rng=rng_jit)
    if args.sim:
        st = sim.stats_snapshot()
        print(f"[run] JIT {'KV增量' if args.kv else '全序列'} cim_cycle={st['cim_cycle']}, "
              f"mmio_cycle={st['mmio_cycle']}", file=sys.stderr)

    # ---- PyTorch reference ----
    ref_tokens = None
    if not args.no_ref:
        print("[run] PyTorch reference 生成...", file=sys.stderr)
        ref_tokens = generate(invoker, model, idx0, args.n, args.block_size, ref=True,
                              temp=args.temperature, top_k=args.top_k, rng=rng_ref)

    # ---- 输出 (int 模式: token 列表; 文本模式: decode 文本) ----
    print("\n" + "=" * 60, file=sys.stderr)
    if args.prompt_int is not None:
        print(f"prompt  : token id = {args.prompt_int}")
        print(f"JIT 仿真输出 ({len(jit_tokens)} token): {jit_tokens}")
        if ref_tokens is not None:
            match = jit_tokens == ref_tokens
            print(f"PyTorch ref 输出: {ref_tokens}")
            print(f"JIT==ref ({mode} token 级): {'✓ YES' if match else '✗ NO'}")
    else:
        jit_text = tok.decode(jit_tokens)
        print(f"prompt  : {args.prompt!r}")
        print(f"JIT 仿真输出 ({len(jit_tokens)} token):")
        print(jit_text)
        if ref_tokens is not None:
            ref_text = tok.decode(ref_tokens)
            match = jit_tokens == ref_tokens
            print("-" * 60)
            print(f"PyTorch ref 输出:")
            print(ref_text)
            print("-" * 60)
            print(f"JIT==ref ({mode} token 级): {'✓ YES' if match else '✗ NO'}")
            if not match and args.temperature > 0:
                print("(采样模式: 编译降级 logits 微差在敏感 token 翻转采样, != 不代表 bug)")
    print("=" * 60, file=sys.stderr)


if __name__ == "__main__":
    main()
