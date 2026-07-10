#!/usr/bin/env python3
"""C3: 48-bit 指令编码 (cimres -> MACRO_PROG_WGT/MATMUL/SYNC_HALT)。

遍历 placed cimres IR, 编码 48-bit 指令 (§3.1):
  preload_weight -> MACRO_PROG_WGT (opcode=0x1, dest_id, PAGE_1=b_page_start)
  macro_matmul   -> MACRO_MATMUL   (opcode=0x2, dest_id, PAGE_1=a_page, PAGE_2=psum_page, ACCUM)
  sync_halt      -> SYNC_HALT       (opcode=0x7)

48-bit 字段: [47:45]opcode | [44:33]dest_id | [32:21]page1 | [20:9]page2 | [8]accum | [7:0]保留
word = (opcode<<45)|(dest_id<<33)|(page1<<21)|(page2<<9)|(accum<<8), 每条 6 字节小端。

输出:
  preload.bin : 5 批, 每批 764 PROG_WGT + SYNC_HALT (批间同步, §4.6 Preload 分批)
  forward.bin : 3664 MATMUL (按 IR 调度顺序) + SYNC_HALT (段末)

用法:
  nanogpt-gpu python cim_compiler/cimres/emit_instr.py
"""
import os
import sys
import struct
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
CIM_COMPILER = os.path.dirname(HERE)
REPO = os.path.dirname(CIM_COMPILER)
EXPORT_DIR = os.path.join(CIM_COMPILER, "export")
for _p in (REPO, EXPORT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from torch_mlir import ir
from cim_compiler.cimres.dialect import register_cimres

OP_PROG_WGT = 0x1
OP_MATMUL = 0x2
OP_SYNC_HALT = 0x7
PRELOAD_BATCH = 764   # §4.6: 每批 764 tile (3056 PAGE)


def encode(opcode, dest_id=0, page1=0, page2=0, accum=0):
    word = (opcode << 45) | (dest_id << 33) | (page1 << 21) | (page2 << 9) | (accum << 8)
    return word & ((1 << 48) - 1)


def word_to_bytes(word):
    return struct.pack("<Q", word)[:6]   # 48-bit = 6 字节, 小端


def emit(placed_path, preload_out, forward_out):
    ctx = ir.Context()
    ctx.load_all_available_dialects()
    register_cimres(ctx)
    with ctx:
        mod = ir.Module.parse(open(placed_path).read(), ctx)

    preloads = []   # [(dest_id, b_page_start)]
    matmuls = []    # [(dest_id, a_page, psum_page, accum)]
    for op in list(mod.body):
        if op.operation.name == "cimres.preload_weight":
            d = int(op.attributes["dest_id"].value)
            b = int(op.attributes["b_page_start"].value)
            preloads.append((d, b))
        elif op.operation.name == "func.func":
            blk = op.regions[0].blocks[0]
            for inner in list(blk.operations):
                if inner.operation.name != "cimres.macro_matmul":
                    continue
                d = int(inner.attributes["dest_id"].value)
                a = int(inner.attributes["a_page"].value)
                p = int(inner.attributes["psum_page"].value)
                acc = 1 if bool(inner.attributes["accum"].value) else 0
                matmuls.append((d, a, p, acc))
    preloads.sort(key=lambda x: x[0])   # 按 dest_id (全局 tile 索引) 排序, 保证批内连续

    # preload.bin: 5 批, 每批 764 PROG_WGT + SYNC_HALT (批间同步)
    with open(preload_out, "wb") as f:
        for bs in range(0, len(preloads), PRELOAD_BATCH):
            for d, b in preloads[bs:bs + PRELOAD_BATCH]:
                f.write(word_to_bytes(encode(OP_PROG_WGT, dest_id=d, page1=b)))
            f.write(word_to_bytes(encode(OP_SYNC_HALT)))

    # forward.bin: 所有 MATMUL (按 IR 调度顺序) + SYNC_HALT (段末)
    with open(forward_out, "wb") as f:
        for d, a, p, acc in matmuls:
            f.write(word_to_bytes(encode(OP_MATMUL, dest_id=d, page1=a, page2=p, accum=acc)))
        f.write(word_to_bytes(encode(OP_SYNC_HALT)))

    n_batch = (len(preloads) + PRELOAD_BATCH - 1) // PRELOAD_BATCH
    print(f"[C3] preload: {len(preloads)} PROG_WGT 分 {n_batch} 批 (每批≤{PRELOAD_BATCH}+SYNC_HALT)"
          f" -> {preload_out}", file=sys.stderr)
    print(f"[C3] forward: {len(matmuls)} MATMUL + SYNC_HALT -> {forward_out}", file=sys.stderr)
    print(f"[C3] preload.bin = {(len(preloads)+n_batch)*6} 字节, "
          f"forward.bin = {(len(matmuls)+1)*6} 字节", file=sys.stderr)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default="cim_compiler/cimres/checkpoints/bitnet_ternary_cimres_placed.mlir")
    p.add_argument("--preload", default="cim_compiler/cimres/checkpoints/preload.bin")
    p.add_argument("--forward", default="cim_compiler/cimres/checkpoints/forward.bin")
    args = p.parse_args()
    emit(args.inp, args.preload, args.forward)
