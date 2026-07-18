#!/usr/bin/env python3
"""cimres pass 编排 (S0): canonicalize + cse (C1 输出 -> C2 输入)。

pipeline step 5: C1 cimres.mlir -> [canon + cse] -> C2 place
canon/cse 在逻辑层 (cimres.mlir, 占位 PAGE), 处理 dest_id/sync_halt/preload 冗余。
当前规整 IR 消除 0 (框架就位, 供后续 fusion/调度产生冗余时用)。

用法:
  python cim_compiler/cimres/run_passes.py --in <cimres.mlir> --out <cimres.mlir>
"""
import sys
import argparse

from cim_compiler.cimres.passes.common import load_cimres, save
from cim_compiler.cimres.passes.canonicalize import canonicalize
from cim_compiler.cimres.passes.cse import cse


def run_passes(inp, out=None):
    mod, _ = load_cimres(inp)
    nc = canonicalize(mod)
    ncs = cse(mod)
    out = out or inp
    save(mod, out)
    return nc, ncs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default="cim_compiler/cimres/checkpoints/bitnet_ternary_cimres.mlir")
    p.add_argument("--out", default=None, help="输出 (默认原地)")
    args = p.parse_args()
    nc, ncs = run_passes(args.inp, args.out)
    print(f"[passes] canon 消除 {nc}, cse 消除 {ncs} -> {args.out or args.inp}", file=sys.stderr)


if __name__ == "__main__":
    main()
