#!/usr/bin/env python3
"""C3: 48-bit 指令编码 (cimres -> MACRO_PROG_WGT/MATMUL/SYNC_HALT)。

遍历 placed cimres IR, 编码 48-bit 指令 (§3.1):
  preload_weight -> MACRO_PROG_WGT (opcode=0x1, dest_id, PAGE_1=b_page_start)
  macro_matmul   -> MACRO_MATMUL   (opcode=0x2, dest_id, PAGE_1=a_page, PAGE_2=psum_page, ACCUM)
  sync_halt      -> SYNC_HALT       (opcode=0x7)

48-bit 字段: [47:45]opcode | [44:33]dest_id | [32:21]page1 | [20:9]page2 | [8]accum | [7:0]保留
word = (opcode<<45)|(dest_id<<33)|(page1<<21)|(page2<<9)|(accum<<8), 每条 6 字节小端。

产物格式 (供 cim_stub.c 硬件驱动骨架加载, 小端):
  forward.bin (按 idx 索引, cim_launch_<idx> 用 idx 查段):
    header: magic "CIMF" | n_idx(u32) | offsets[n_idx](u32) | lengths[n_idx](u32)
    data:   段[idx] = MATMUL...(6B/条) + SYNC_HALT(6B)
            (idx 顺序 = func.func 顺序 = partition cim_blocks 顺序 = cim_launch_<idx> 的 IDX)
  preload.bin (自包含 tile 数据, cim_preload_init 读一个文件驱动 Preload):
    header: magic "CIMP" | n_batch(u32) | batch_offsets[n_batch](u32)
    body:   每批 = n_tile(u32) | tile_data(n_tile*1024B) | prog_wgt(n_tile*6B) | sync_halt(6B)
            (681 tile/批, 指令区容量约束; tile 2bit packed, PROG_WGT page1=b_page_start=i*4)

用法:
  nanogpt-gpu python cim_compiler/cimres/emit_instr.py
"""
import os
import sys
import math
import struct
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
CIM_COMPILER = os.path.dirname(HERE)
REPO = os.path.dirname(CIM_COMPILER)
EXPORT_DIR = os.path.join(CIM_COMPILER, "export")
for _p in (REPO, EXPORT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from torch_mlir import ir
from cim_compiler.cimres.dialect import register_cimres
from cim_compiler.export.weight_blob import read_weight_blob

# CIM ASIC 硬件参数集中定义 (cim_compiler/cimres/hw_config.py, C/Python 镜像)
from cim_compiler.cimres.hw_config import *   # noqa: F401  OP_*/TILE/PRELOAD_BATCH/MACRO_MAX/SEG_MAX
FORWARD_MAGIC = b"CIMF"
PRELOAD_MAGIC = b"CIMP"


def encode(opcode, dest_id=0, page1=0, page2=0, accum=0):
    word = (opcode << 45) | (dest_id << 33) | (page1 << 21) | (page2 << 9) | (accum << 8)
    return word & ((1 << 48) - 1)


def word_to_bytes(word):
    return struct.pack("<Q", word)[:6]   # 48-bit = 6 字节, 小端


def _norm(name):
    return name.replace("_", ".")


def emit(placed_path, weights_path, preload_out, forward_out):
    # ---- 读 placed IR ----
    ctx = ir.Context()
    ctx.load_all_available_dialects()
    register_cimres(ctx)
    with ctx:
        mod = ir.Module.parse(open(placed_path).read(), ctx)

    # 按 func.func 顺序遍历 (idx = func 序号 = cim_launch_<idx> IDX)
    forward_segs = []   # idx -> [48-bit MATMUL 字 (+SYNC_HALT)]
    seg_names = []      # [P1-5] idx -> BitLinear 名 (容量校验报错用)
    preload_list = []   # [(dest_id, b_page_start)]
    dest_meta = {}      # dest_id -> (name, nb, kb)
    for op in list(mod.body):
        if op.operation.name == "cimres.preload_weight":
            d = int(op.attributes["dest_id"].value)
            b = int(op.attributes["b_page_start"].value)
            preload_list.append((d, b))
        elif op.operation.name == "func.func":
            seg = []
            seg_name = "?"
            blk = op.regions[0].blocks[0]
            for inner in list(blk.operations):
                if inner.operation.name != "cimres.macro_matmul":
                    continue
                d = int(inner.attributes["dest_id"].value)
                a = int(inner.attributes["a_page"].value)
                p = int(inner.attributes["psum_page"].value)
                acc = 1 if bool(inner.attributes["accum"].value) else 0
                name = str(inner.attributes["bitlinear_name"].value)
                nb = int(inner.attributes["n_blk"].value)
                kb = int(inner.attributes["k_blk"].value)
                if seg_name == "?":
                    seg_name = name   # [P1-5] 段对应 BitLinear 名
                seg.append(encode(OP_MATMUL, dest_id=d, page1=a, page2=p, accum=acc))
                dest_meta[d] = (name, nb, kb)
            seg.append(encode(OP_SYNC_HALT))
            forward_segs.append(seg)
            seg_names.append(seg_name)
    preload_list.sort(key=lambda x: x[0])   # 按 dest_id 排序, 批内连续

    # [P1-5] 容量校验 (硬件约束, 友好报错)
    n_tile = len(preload_list)
    if n_tile > MACRO_MAX:
        raise ValueError(
            f"[C3] tile 总数 {n_tile} > Macro 上限 {MACRO_MAX} (§4.5)。"
            f"降低模型规模 (n_layer/d_model/ffn) 或启用 Macro 复用")
    for i, seg in enumerate(forward_segs):
        n_mm = len(seg) - 1   # 减 SYNC_HALT
        if n_mm > SEG_MAX:
            print(f"[C3] BitLinear '{seg_names[i]}' (idx={i}) 单段 MATMUL {n_mm} "
                  f"> {SEG_MAX}, 启用大段分块 (P2-6, cim_launch 多块门铃)",
                  file=sys.stderr)

    # ---- 读 weights, 建 dest_id -> tile_2bit (1024B) ----
    weights = read_weight_blob(weights_path)
    wmap = {_norm(w.name): w for w in weights}
    tile_of = {}   # dest_id -> 1024B 2bit packed
    for d, (name, nb, kb) in dest_meta.items():
        we = wmap[name]
        N, K = we.N, we.K
        n_tiles = math.ceil(N / TILE)
        k_tiles = math.ceil(K / TILE)
        Np, Kp = n_tiles * TILE, k_tiles * TILE
        packed = np.frombuffer(we.packed, dtype=np.uint8).reshape(N, K // 4)
        packed_pad = np.zeros((Np, Kp // 4), dtype=np.uint8)
        packed_pad[:N, :K // 4] = packed
        tile = packed_pad[nb * TILE:(nb + 1) * TILE, kb * (TILE // 4):(kb + 1) * (TILE // 4)]
        tile_of[d] = tile.tobytes()   # 64*16 = 1024B

    # ---- forward.bin (按 idx 索引) ----
    n_idx = len(forward_segs)
    seg_bytes = [b"".join(word_to_bytes(w) for w in seg) for seg in forward_segs]
    offsets, lengths, cur = [], [], 0
    for sb in seg_bytes:
        offsets.append(cur)
        lengths.append(len(sb))
        cur += len(sb)
    with open(forward_out, "wb") as f:
        f.write(FORWARD_MAGIC)
        f.write(struct.pack("<I", n_idx))
        f.write(struct.pack(f"<{n_idx}I", *offsets))
        f.write(struct.pack(f"<{n_idx}I", *lengths))
        for sb in seg_bytes:
            f.write(sb)

    # ---- preload.bin (自包含 tile 数据, 681/批) ----
    batches = [preload_list[bs:bs + PRELOAD_BATCH]
               for bs in range(0, len(preload_list), PRELOAD_BATCH)]
    batch_bytes = []
    for batch in batches:
        n_tile = len(batch)
        buf = struct.pack("<I", n_tile)
        for d, b in batch:                  # tile 数据 (1024B/tile, 写覆盖区)
            buf += tile_of[d]
        for d, b in batch:                  # PROG_WGT 指令 (page1=b_page_start=i*4)
            buf += word_to_bytes(encode(OP_PROG_WGT, dest_id=d, page1=b))
        buf += word_to_bytes(encode(OP_SYNC_HALT))
        batch_bytes.append(buf)
    b_offsets, cur = [], 0
    for bb in batch_bytes:
        b_offsets.append(cur)
        cur += len(bb)
    with open(preload_out, "wb") as f:
        f.write(PRELOAD_MAGIC)
        f.write(struct.pack("<I", len(batches)))
        f.write(struct.pack(f"<{len(batches)}I", *b_offsets))
        for bb in batch_bytes:
            f.write(bb)

    fwd_size = 4 + 4 + n_idx * 8 + sum(lengths)
    pre_size = 4 + 4 + len(batches) * 4 + sum(len(b) for b in batch_bytes)
    print(f"[C3] forward: {n_idx} 段 (按 idx 索引), {sum(lengths) // 6} 条指令 "
          f"-> {forward_out} ({fwd_size} 字节)", file=sys.stderr)
    print(f"[C3] preload: {len(preload_list)} tile 分 {len(batches)} 批 (自包含, 681/批) "
          f"-> {preload_out} ({pre_size} 字节)", file=sys.stderr)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default="cim_compiler/cimres/checkpoints/bitnet_ternary_cimres_placed.mlir")
    p.add_argument("--weights", default="checkpoints/bitnet_ternary_weights.bin")
    p.add_argument("--preload", default="cim_compiler/cimres/checkpoints/preload.bin")
    p.add_argument("--forward", default="cim_compiler/cimres/checkpoints/forward.bin")
    args = p.parse_args()
    emit(args.inp, args.weights, args.preload, args.forward)
