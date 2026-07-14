#!/usr/bin/env python3
"""C2: 资源映射 (Macro Dest_ID + PAGE 布局)。

填 C1 cimres IR 的物理 PAGE 属性 (C1 占位 0):
  a_page       : Forward 输入区 (覆盖区 0x010+), a_page = 0x010 + k_blk
                 (每 k_slice 64 int8 = 64B; 每 BitLinear 复用, Forward 期串行不冲突)
  psum_page    : 累加区 (0xC00+), psum_page = 0xC00 + n_blk
                 (每 n_blk 1 PAGE = 64 int32 = 256B; K 维多 tile RMW 累加同 PAGE)
  b_page_start : Preload 区 (覆盖区 0x000~0xBEF, 分批每批 681 tile)
                 b_page_start = (dest_id % 681) * 4  (每 tile 2bit = 1024B = 4 PAGE, 批内复用)
                 (批大小受指令区容量约束: 4KB/6B=682 条 48-bit, 留 1 SYNC_HALT -> 681;
                  §4.6 的 764 是覆盖区约束, 与指令区 682 矛盾, 取保守 681)

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
from cim_compiler.cimres.hw_config import A_PAGE_BASE, PSUM_PAGE_BASE, PRELOAD_BATCH  # ASIC 硬件参数 (集中定义)


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
                    # S6: qkv 合并 func 内 q/k/v 各自 x_int8 不同 -> a_page 错开 (3 组 A_PAGE);
                    # PSUM_PAGE 错开 (q:0-7/k:8-11/v:12-15, 16 PAGE/层, bank0 32 PAGE 够)
                    bl_name = str(inner.attributes["bitlinear_name"].value)
                    if bl_name.endswith("k.proj"):
                        a_off, p_off = 8, 8     # k: A_PAGE+8 (0x018+kb), PSUM+8 (0xC08+nb)
                    elif bl_name.endswith("v.proj"):
                        a_off, p_off = 16, 12    # v: A_PAGE+16 (0x020+kb), PSUM+12 (0xC0C+nb)
                    else:
                        a_off, p_off = 0, 0      # q + 非 qkv: 原 A_PAGE+kb, PSUM+nb
                    inner.attributes["a_page"] = i32_attr(A_PAGE_BASE + a_off + kb)
                    inner.attributes["psum_page"] = i32_attr(PSUM_PAGE_BASE + p_off + nb)
                    n_matmul += 1
        if max_dest >= 4096:
            raise ValueError(
                f"[C2] max dest_id={max_dest} >= Macro 上限 4096 (§4.5)。"
                f"tile 总数 {max_dest + 1} 超 Macro 数, 降低模型规模或启用 Macro 复用")
        mod.operation.verify()
        with open(out_path, "w") as f:
            f.write(str(mod))
        print(f"[C2] {n_preload} preload + {n_matmul} matmul 物理绑定, "
              f"max dest_id={max_dest} (< 4096: OK)",
              file=sys.stderr)
        print(f"[C2] A_PAGE=0x{A_PAGE_BASE:x}+k_blk, PSUM_PAGE=0x{PSUM_PAGE_BASE:x}+n_blk, "
              f"b_page=(dest%{PRELOAD_BATCH})*4", file=sys.stderr)
        print(f"[C2] saved: {out_path}", file=sys.stderr)
    return mod


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default="cim_compiler/cimres/checkpoints/bitnet_ternary_cimres.mlir")
    p.add_argument("--out", default="cim_compiler/cimres/checkpoints/bitnet_ternary_cimres_placed.mlir")
    args = p.parse_args()
    place(args.inp, args.out)
