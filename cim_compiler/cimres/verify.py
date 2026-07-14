#!/usr/bin/env python3
"""cimres 形式化验证 pass (S0 安全网)。

检查 cimres IR (cimres.mlir 或 placed.mlir) 的结构正确性, 基于调度语义
(C1: 外层 k_blk 串行 / 内层 n_blk 并行):

  1. dest_id 全局唯一 (preload 集合, Macro 1:1 映射) + macro_matmul dest_id 属于 preload
  2. accum 链: per (bitlinear, n_blk) k_blk=0 accum=false (覆盖), k_blk>0 accum=true (累加),
     psum_page 全同 (K 维累加同 PAGE), k_blk 连续 0..k_tiles-1
  3. PAGE 并行冲突: 同 k_blk 不同 n_blk 的 psum_page 互不相同 (并行写同 PAGE 会冲突)
  4. a_page 一致: 同 k_blk 的 a_page 相同 (共享输入页广播)

verify(mod) -> issues: list[str], 空=通过。
verify_or_raise(mod): 失败抛 ValueError (pipeline gate 用)。

用法:
  python cim_compiler/cimres/verify.py --in <cimres.mlir>
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
CIM_COMPILER = os.path.dirname(HERE)
REPO = os.path.dirname(CIM_COMPILER)
EXPORT_DIR = os.path.join(CIM_COMPILER, "export")   # cim_op 所在 (cim_compiler/__init__ 触发 export import)
for _p in (REPO, EXPORT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cim_compiler.cimres.passes.common import walk_ops, func_blocks, matmuls_in_func


def verify(mod) -> list:
    """返回 issue 列表, 空=通过。"""
    issues = []
    issues += _check_dest_id_unique(mod)
    preload_dests = _preload_dest_set(mod)
    for func_op, _ in func_blocks(mod):
        issues += _check_func(func_op, preload_dests)
    issues += _check_double_buffer_layout(mod)
    return issues


def _check_double_buffer_layout(mod):
    """S2: double buffer bank0/bank1 PAGE 布局不冲突 (基于 IR max k/n tiles)。

    bank0: [A_PAGE_BASE, +max_k), bank1: [A_PAGE_BANK1_BASE, +max_k)。
    检查 bank0/bank1 不重叠, bank1 不越界指令区。PSUM 同理。
    """
    from cim_compiler.cimres.hw_config import (
        A_PAGE_BASE, A_PAGE_BANK1_BASE, PSUM_PAGE_BASE, PSUM_BANK1_BASE, INSTR_BASE)
    issues = []
    max_a = max_p = 0   # S6: 用实际 a_page/psum_page (qkv 错开后 max 反映真实占用)
    for func_op, _ in func_blocks(mod):
        for m in matmuls_in_func(func_op):
            max_a = max(max_a, m["a_page"])
            max_p = max(max_p, m["psum_page"])
    if max_a == 0:
        return issues
    a_bank_off = A_PAGE_BANK1_BASE - A_PAGE_BASE   # double buffer bank1 偏移
    if max_a >= A_PAGE_BANK1_BASE:
        issues.append(f"A_PAGE bank0 越界 bank1 (max a_page=0x{max_a:x} >= 0x{A_PAGE_BANK1_BASE:x})")
    if max_a + a_bank_off >= INSTR_BASE:
        issues.append(f"A_PAGE bank1 越界指令区 0x{INSTR_BASE:x} (max a_page=0x{max_a:x})")
    if max_p >= PSUM_BANK1_BASE:
        issues.append(f"PSUM bank0 越界 bank1 (max psum_page=0x{max_p:x} >= 0x{PSUM_BANK1_BASE:x})")
    return issues


def verify_or_raise(mod):
    """verify 失败抛 ValueError。pipeline gate 用。"""
    issues = verify(mod)
    if issues:
        raise ValueError("cimres verify 失败:\n  " + "\n  ".join(issues))
    return True


def _preload_dest_set(mod):
    return {int(op.attributes["dest_id"].value)
            for op in walk_ops(mod, "cimres.preload_weight")}


def _check_dest_id_unique(mod):
    """1. dest_id 唯一 (preload, Macro 1:1)。"""
    issues = []
    seen = {}
    for op in walk_ops(mod, "cimres.preload_weight"):
        d = int(op.attributes["dest_id"].value)
        seen[d] = seen.get(d, 0) + 1
    dups = {d: c for d, c in seen.items() if c > 1}
    if dups:
        issues.append(f"dest_id 重复 preload (Macro 1:1 映射破坏): {dups}")
    return issues


def _check_func(func_op, preload_dests):
    """2-4. per func: dest_id 属于 preload + accum 链 + PAGE 冲突 + a_page 一致。"""
    issues = []
    ms = matmuls_in_func(func_op)
    if not ms:
        return issues
    # S6: qkv 合并 func 含多个 bitlinear_name (q/k/v), 按 name 分组检查
    # 1. dest_id 属于 preload (每 matmul 必有对应 Macro 预载)
    for m in ms:
        if m["dest_id"] not in preload_dests:
            issues.append(
                f"[{m['bitlinear_name']}] matmul dest_id={m['dest_id']} (n={m['n_blk']},k={m['k_blk']}) 无对应 preload")

    # 2. accum 链 per (name, n_blk): k_blk 连续 + psum_page 同 + accum 正确
    by_nn = {}
    for m in ms:
        by_nn.setdefault((m["bitlinear_name"], m["n_blk"]), []).append(m)
    for (name, n), group in by_nn.items():
        group.sort(key=lambda x: x["k_blk"])
        kblks = [m["k_blk"] for m in group]
        if kblks != list(range(len(kblks))):
            issues.append(f"[{name}] n_blk={n} k_blk 不连续 0..{len(kblks) - 1}: {kblks}")
        psums = {m["psum_page"] for m in group}
        if len(psums) > 1:
            issues.append(f"[{name}] n_blk={n} K维累加 psum_page 不一致: {sorted(psums)}")
        for m in group:
            expect = m["k_blk"] > 0
            if m["accum"] != expect:
                issues.append(
                    f"[{name}] n_blk={n} k_blk={m['k_blk']} accum={m['accum']} 应={expect}")

    # 3. PAGE 并行冲突 per (name, k_blk): 同 name 同 k_blk 不同 n_blk psum_page 互不相同
    by_nk = {}
    for m in ms:
        by_nk.setdefault((m["bitlinear_name"], m["k_blk"]), []).append(m)
    for (name, k), group in by_nk.items():
        owner = {}
        for m in group:
            if m["psum_page"] in owner:
                issues.append(
                    f"[{name}] k_blk={k} 并行冲突: psum_page={m['psum_page']} "
                    f"被 n_blk={owner[m['psum_page']]} 与 n_blk={m['n_blk']} 共用 (并行写冲突)")
            else:
                owner[m["psum_page"]] = m["n_blk"]

    # 4. a_page 一致 per (name, k_blk): 同 name 同 k_blk a_page 同 (qkv a_page 错开, 跨 name 不同)
    for (name, k), group in by_nk.items():
        apages = {m["a_page"] for m in group}
        if len(apages) > 1:
            issues.append(
                f"[{name}] k_blk={k} a_page 不一致 (广播输入页应同): {sorted(apages)}")

    return issues


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default="cim_compiler/cimres/checkpoints/bitnet_ternary_cimres_placed.mlir")
    args = p.parse_args()

    from cim_compiler.cimres.passes.common import load_cimres
    mod, _ = load_cimres(args.inp)
    issues = verify(mod)
    if issues:
        print(f"[verify] FAIL ✗ ({len(issues)} issue):", file=sys.stderr)
        for s in issues[:20]:
            print(f"  {s}", file=sys.stderr)
        if len(issues) > 20:
            print(f"  ... 还有 {len(issues) - 20} 条", file=sys.stderr)
        sys.exit(1)
    print(f"[verify] PASS ✓ (cimres IR 结构正确: dest_id 唯一 / accum 链 / PAGE 无冲突 / a_page 一致)",
          file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    main()
