#!/usr/bin/env python3
"""cimres 规范化 pass (S0)。

安全的规范化, 不改语义 (round-trip 不坏数值):
  1. dedup_preload:    同 dest_id 的重复 preload_weight 去重 (保留首个)
  2. merge_sync_halt:  func 内连续多个 sync_halt 合并 (保留首个, 删后续)
  3. dead_preload:     dest_id 无对应 macro_matmul 的死 preload 消除

preload_weight / sync_halt 都是 0 operand 0 result 的元数据/副作用 op, erase 无 use 风险。

用法:
  python cim_compiler/cimres/passes/canonicalize.py --in <cimres.mlir> --out <out.mlir>
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
CIM_COMPILER = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(CIM_COMPILER)
EXPORT_DIR = os.path.join(CIM_COMPILER, "export")
for _p in (REPO, EXPORT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cim_compiler.cimres.passes.common import walk_ops, func_blocks, matmuls_in_func


def canonicalize(mod) -> int:
    """规范化, 返回消除的 op 数。"""
    n = 0
    n += _dedup_preload(mod)
    n += _merge_sync_halt(mod)
    n += _remove_dead_preload(mod)
    return n


def _dedup_preload(mod):
    """同 dest_id 的重复 preload_weight 去重 (Macro 1:1, 重复预载冗余)。"""
    seen = set()
    to_erase = []
    for op in walk_ops(mod, "cimres.preload_weight"):
        d = int(op.attributes["dest_id"].value)
        if d in seen:
            to_erase.append(op)
        else:
            seen.add(d)
    for op in to_erase:
        op.erase()   # 0 result, 无 uses
    return len(to_erase)


def _merge_sync_halt(mod):
    """func 内连续多个 sync_halt 合并为一个 (中间无其他 op 才合并)。"""
    n = 0
    for _, blk in func_blocks(mod):
        prev_halt = False
        to_erase = []
        for op in list(blk.operations):
            if op.operation.name == "cimres.sync_halt":
                if prev_halt:
                    to_erase.append(op)
                else:
                    prev_halt = True
            else:
                prev_halt = False   # 非 halt 打断, 只合并真正连续的
        for op in to_erase:
            op.erase()
            n += 1
    return n


def _remove_dead_preload(mod):
    """dest_id 无对应 macro_matmul 的死 preload 消除 (Macro 预载了却不用)。"""
    used = set()
    for func_op, _ in func_blocks(mod):
        for m in matmuls_in_func(func_op):
            used.add(m["dest_id"])
    n = 0
    for op in walk_ops(mod, "cimres.preload_weight"):
        d = int(op.attributes["dest_id"].value)
        if d not in used:
            op.erase()
            n += 1
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default="cim_compiler/cimres/checkpoints/bitnet_ternary_cimres_placed.mlir")
    p.add_argument("--out", default=None, help="输出 (默认原地 round-trip 验证)")
    args = p.parse_args()

    from cim_compiler.cimres.passes.common import load_cimres, save
    mod, _ = load_cimres(args.inp)
    n = canonicalize(mod)
    out = args.out or args.inp
    save(mod, out)
    print(f"[canon] 消除 {n} 个冗余 op -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
