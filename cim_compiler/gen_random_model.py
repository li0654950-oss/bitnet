#!/usr/bin/env python3
"""[P3-7] 构造随机初始化 BitNet (指定规模) + export_ternary -> ternary.pt。

用于多规模回归测试: 随机权重验证编译正确性 (JIT vs PyTorch max_diff=0), 非模型精度。
权重随机三值化, 只要 JIT 和 PyTorch 用同样权重, max_diff=0 即验证流水线对规模兼容。

用法:
  python cim_compiler/gen_random_model.py --n_layer 2 --d_model 256 --ffn_dim 1024 --out /tmp/small.pt
  python cim_compiler/pipeline.py --ternary /tmp/small.pt --n_layer 2 --d_model 256 --ffn_dim 1024
"""
import os
import sys
import math
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BITNET_DIR = os.path.join(REPO, "bitnet")
sys.path.insert(0, BITNET_DIR)

import torch
from model import BitNet
from export_ternary import weight_quant, pack_2bit, is_bitlinear_weight


def gen(vocab_size, d_model, block_size, n_layer, n_head, n_kv_head, ffn_dim, out, seed=42):
    torch.manual_seed(seed)
    model = BitNet(vocab_size, d_model, block_size, n_layer, n_head, n_kv_head, ffn_dim)
    model.eval()
    sd = model.state_dict()
    keys = set(sd.keys())
    exported = {}
    n_bl = 0
    n_tile = 0
    for name, t in sd.items():
        if is_bitlinear_weight(name, keys):
            w = t.to(torch.float32)
            ternary, scale = weight_quant(w)
            packed = pack_2bit(ternary.to(torch.int8))
            exported[name] = {"packed": packed.cpu(), "scale": float(scale), "shape": tuple(w.shape)}
            n_bl += 1
            N, K = w.shape
            n_tile += math.ceil(N / 64) * math.ceil(K / 64)
        else:
            exported[name] = t.cpu()
    torch.save(exported, out)
    print(f"[gen] BitNet(vocab={vocab_size} d={d_model} L={n_layer} h={n_head} "
          f"kv={n_kv_head} ffn={ffn_dim} bs={block_size})")
    print(f"[gen] {n_bl} BitLinear, {n_tile} tile "
          f"({'< 4096 Macro OK' if n_tile < 4096 else '>= 4096 超 Macro, 容量校验会报错'})")
    print(f"[gen] saved: {out}")
    return n_tile


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vocab_size", type=int, default=65)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_kv_head", type=int, default=4)
    p.add_argument("--ffn_dim", type=int, default=1664)
    p.add_argument("--out", default="/tmp/random_bitnet.pt")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.d_model % args.n_head != 0:
        print(f"[err] d_model {args.d_model} 必须整除 n_head {args.n_head}", file=sys.stderr)
        sys.exit(1)
    if args.n_head % args.n_kv_head != 0:
        print(f"[err] n_head {args.n_head} 必须整除 n_kv_head {args.n_kv_head}", file=sys.stderr)
        sys.exit(1)
    gen(args.vocab_size, args.d_model, args.block_size, args.n_layer,
        args.n_head, args.n_kv_head, args.ffn_dim, args.out, args.seed)


if __name__ == "__main__":
    main()
