#!/usr/bin/env python3
"""C1: cim.matmul -> cimres tile 序列 (逻辑展开)。

输入: partition.json (37 CIM 块) + weights.bin (read_weight_blob)
对每个 BitLinear (按 bitlinear_name 匹配 WeightEntry 拿 N,K):
  tile 切 ceil(N/64) × ceil(K/64)
  生成 macro_matmul tile 序列 (单 token M=1)
  K 维 ACCUM (首个 k_blk=0 覆盖, 后续=1 累加), zero-pad (ceil 补 0, valid_region 记录)
  调度: 外层 k_blk 串行 / 内层 n_blk 并行 (多 Macro 并行, 避 PSUM_PAGE 写冲突)
  dest_id = 全局 tile 索引 (Macro 分配), a_page/psum_page/b_page_start 占位 0 (C2 填物理)
输出: cimres IR (.mlir)

不改前期工程: 只读 partition.json + weights.bin, 独立路径。

用法:
  nanogpt-gpu python cim_compiler/cimres/lower_to_cimres.py
"""
import os
import sys
import json
import math
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))        # cim_compiler/cimres/
CIM_COMPILER = os.path.dirname(HERE)                     # cim_compiler/
REPO = os.path.dirname(CIM_COMPILER)                     # repo root
EXPORT_DIR = os.path.join(CIM_COMPILER, "export")        # cim_op.py 所在 (inference_model 顶层 import cim_op)
for _p in (REPO, EXPORT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from torch_mlir import ir
from torch_mlir.dialects import func as func_d
from cim_compiler.cimres.dialect import (
    register_cimres, make_macro_matmul, make_preload_weight,
    make_sync_halt, make_tile_group, int32_vec,
)
from cim_compiler.export.weight_blob import read_weight_blob

TILE = 64  # Macro 物理维度 64×64


def _norm(name):
    """规范化 BitLinear 名。partition 把 _ 变 . (q.proj), weights 保留 _ (q_proj), 统一为 .。"""
    return name.replace("_", ".")


def lower_to_cimres(partition_path, weights_path, out_path):
    part = json.load(open(partition_path))
    weights = read_weight_blob(weights_path)
    wmap = {_norm(w.name): w for w in weights}

    ctx = ir.Context()
    ctx.load_all_available_dialects()
    register_cimres(ctx)

    total_tiles = 0
    with ctx, ir.Location.unknown():
        mod = ir.Module.create()
        with ir.InsertionPoint(mod.body):
            for blk in part["cim_blocks"]:
                name = blk["bitlinear_name"]
                we = wmap.get(_norm(name))
                assert we is not None, f"权重未找到: {name} (norm={_norm(name)})"
                N, K = we.N, we.K
                n_tiles = math.ceil(N / TILE)   # ceil(N/64), 不足 zero-pad
                k_tiles = math.ceil(K / TILE)

                make_tile_group(name, N, K)

                # preload_weight: 每 tile 一个, dest_id = base + n_blk*k_tiles + k_blk
                for nb in range(n_tiles):
                    for kb in range(k_tiles):
                        did = total_tiles + nb * k_tiles + kb
                        make_preload_weight(dest_id=did, b_page_start=0,
                                            bitlinear_name=name)

                # func: 输入 x_int8 (tensor<Kxi8>), 返回 tensor<64xi32> (占位; 模拟器遍历 op 不依赖返回)
                x_ty = ir.Type.parse(f"tensor<{K}xi8>", ctx)
                r_ty = int32_vec(ctx)
                func_ty = ir.FunctionType.get([x_ty], [r_ty], context=ctx)
                fname = "cim_" + name.replace(".", "_")
                f = func_d.FuncOp(name=fname, type=func_ty)
                entry = f.add_entry_block()
                with ir.InsertionPoint(entry):
                    x = entry.arguments[0]
                    last = None
                    # 调度: 外层 k_blk 串行 (累加冲突), 内层 n_blk 并行 (共享 A_PAGE 广播)
                    for kb in range(k_tiles):
                        for nb in range(n_tiles):
                            did = total_tiles + nb * k_tiles + kb
                            accum = kb > 0   # 首个 k_blk ACCUM=0 覆盖, 后续=1 累加
                            last = make_macro_matmul(
                                ctx, x, dest_id=did, a_page=0, psum_page=0,
                                accum=accum, n_blk=nb, k_blk=kb,
                                bitlinear_name=name)
                    make_sync_halt()
                    func_d.ReturnOp([last])

                total_tiles += n_tiles * k_tiles

        mod.operation.verify()
        with open(out_path, "w") as f:
            f.write(str(mod))
        print(f"[C1] {len(part['cim_blocks'])} BitLinear -> cimres IR, "
              f"{total_tiles} tile (< 4096 Macro: {'OK' if total_tiles < 4096 else 'OVER'})",
              file=sys.stderr)
        print(f"[C1] saved: {out_path}", file=sys.stderr)
    return mod, total_tiles


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--partition", default="checkpoints/bitnet_ternary_partition.json")
    p.add_argument("--weights", default="checkpoints/bitnet_ternary_weights.bin")
    p.add_argument("--out", default="cim_compiler/cimres/checkpoints/bitnet_ternary_cimres.mlir")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    lower_to_cimres(args.partition, args.weights, args.out)


if __name__ == "__main__":
    main()
