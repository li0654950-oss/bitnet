#!/usr/bin/env python3
"""C2: 资源映射 (Macro Dest_ID + PAGE 布局)。

填 C1 cimres IR 的物理 PAGE 属性 (C1 占位 0):
  a_page       : Forward 输入区 (覆盖区 0x010+), a_page = 0x010 + k_blk
                 (每 k_slice 64 int8 = 64B; 每 BitLinear 复用, Forward 期串行不冲突)
  psum_page    : 累加区 (0xC00+), psum_page = 0xC00 + n_blk
                 (每 n_blk 1 PAGE = 64 int32 = 256B; K 维多 tile RMW 累加同 PAGE)
  b_page_start : Preload 区 (覆盖区 0x000~0xBEF, 分 5 批每批 764 tile)
                 b_page_start = (dest_id % 764) * 4  (每 tile 2bit = 1024B = 4 PAGE, 批内复用)

dest_id (Macro 分配) C1 已全局分配 (0~3663), C2 确认 < 4096 Macro (§4.5)。
累加区 RMW 串行模型: C1 调度外层 k 串行保证同 PSUM_PAGE 不并行 (§4.7.7)。

输出: 物理绑定 cimres IR。

用法:
  nanogpt-gpu python cim_compiler/cimres/place.py
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
CIM_COMPILER = os.path.dirname(HERE)
REPO = os.path.dirname(CIM_COMPILER)
EXPORT_DIR = os.path.join(CIM_COMPILER, "export")
for _p in (REPO, EXPORT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from torch_mlir import ir
from cim_compiler.cimres.dialect import register_cimres, i32_attr

A_PAGE_BASE = 0x010       # Forward 输入区 (覆盖区, int8 特征)
PSUM_PAGE_BASE = 0xC00    # 部分和累加区 (int32, 256B/PAGE = 64 int32)
PRELOAD_BATCH = 764       # 每批 764 tile (764×4 PAGE = 3056 = 0x000~0xBEF, §4.6)


def place(cimres_in, out_path):
    ctx = ir.Context()
    ctx.load_all_available_dialects()
    register_cimres(ctx)
    with ctx, ir.Location.unknown():
        mod = ir.Module.parse(open(cimres_in).read(), ctx)
        n_preload = 0
        n_matmul = 0
        max_dest = 0
        for op in list(mod.body):
            if op.operation.name == "cimres.preload_weight":
                dest_id = int(op.attributes["dest_id"].value)
                b_page = (dest_id % PRELOAD_BATCH) * 4   # 批内偏移, 每 tile 4 PAGE
                op.attributes["b_page_start"] = i32_attr(b_page)
                n_preload += 1
                max_dest = max(max_dest, dest_id)
            elif op.operation.name == "func.func":
                blk = op.regions[0].blocks[0]
                for inner in list(blk.operations):
                    if inner.operation.name != "cimres.macro_matmul":
                        continue
                    nb = int(inner.attributes["n_blk"].value)
                    kb = int(inner.attributes["k_blk"].value)
                    inner.attributes["a_page"] = i32_attr(A_PAGE_BASE + kb)
                    inner.attributes["psum_page"] = i32_attr(PSUM_PAGE_BASE + nb)
                    n_matmul += 1
        mod.operation.verify()
        with open(out_path, "w") as f:
            f.write(str(mod))
        print(f"[C2] {n_preload} preload + {n_matmul} matmul 物理绑定, "
              f"max dest_id={max_dest} (< 4096: {'OK' if max_dest < 4096 else 'OVER'})",
              file=sys.stderr)
        print(f"[C2] A_PAGE=0x{A_PAGE_BASE:x}+k_blk, PSUM_PAGE=0x{PSUM_PAGE_BASE:x}+n_blk, "
              f"b_page=(dest%764)*4", file=sys.stderr)
        print(f"[C2] saved: {out_path}", file=sys.stderr)
    return mod


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default="cim_compiler/cimres/checkpoints/bitnet_ternary_cimres.mlir")
    p.add_argument("--out", default="cim_compiler/cimres/checkpoints/bitnet_ternary_cimres_placed.mlir")
    args = p.parse_args()
    place(args.inp, args.out)
