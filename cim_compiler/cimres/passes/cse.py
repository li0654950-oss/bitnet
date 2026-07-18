#!/usr/bin/env python3
"""cimres 公共子表达式消除 pass (S0, 保守)。

macro_matmul 有副作用 (写 PSUM_PAGE), 标准 CSE 不能直接套。保守规则:
  只 CSE 所有属性全同 (dest_id/a_page/psum_page/n_blk/k_blk/bitlinear_name)
  且 accum=false (覆盖写, 幂等) 的冗余 op。
  accum=true (累加, 不幂等) 不碰, 避免误消除有意累加。

当前 C1 生成规整 IR 不触发 (每 tile 一次), 此 pass 建框架供后续 fusion/调度
产生重复 macro_matmul 时用。

用法:
  python cim_compiler/cimres/passes/cse.py --in <cimres.mlir> --out <out.mlir>
"""
import sys
import argparse

from cim_compiler.cimres.passes.common import func_blocks


def cse(mod) -> int:
    """保守 CSE, 返回消除的冗余 matmul 数。"""
    n = 0
    for _, blk in func_blocks(mod):
        seen = {}        # key -> first op result
        to_replace = []  # (op, first_result)
        for op in list(blk.operations):
            if op.operation.name != "cimres.macro_matmul":
                continue
            accum = bool(op.attributes["accum"].value)
            if accum:
                continue   # 累加不幂等, 跳过 (保守)
            key = (
                int(op.attributes["dest_id"].value),
                int(op.attributes["a_page"].value),
                int(op.attributes["psum_page"].value),
                int(op.attributes["n_blk"].value),
                int(op.attributes["k_blk"].value),
                str(op.attributes["bitlinear_name"].value),
            )
            if key in seen:
                to_replace.append((op, seen[key]))
            else:
                seen[key] = op.results[0]
        for op, first_res in to_replace:
            op.results[0].replace_all_uses_with(first_res)
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
    n = cse(mod)
    out = args.out or args.inp
    save(mod, out)
    print(f"[cse] 消除 {n} 个冗余 matmul -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
