#!/usr/bin/env python3
"""L1: cim.matmul -> placeholder (aten.mm 仿真占位, 让 LINALG pipeline 过)。

把 RAW IR 的 torch.operator "torch.cim.matmul" 替换为可降级 placeholder:
  %f   = torch.constant.int 6        (float32 dtype 编码)
  %i   = torch.constant.int 3        (int32   dtype 编码)
  %xf  = torch.prims.convert_element_type %X, %f   : [?,K]si8  -> [?,K]f32
  %rep = torch.prim.ListConstruct %1, %4            : -> !torch.list<int>   (repeat dims [1,4])
  %wr  = torch.aten.repeat %W, %rep                 : [N,K/4]ui8 -> [N,K]ui8
  %wt  = torch.aten.t %wr                           : [N,K]ui8 -> [K,N]ui8
  %wf  = torch.prims.convert_element_type %wt, %f   : [K,N]ui8 -> [K,N]f32
  %mm  = torch.aten.mm %xf, %wf                     : [?,K]f32 x [K,N]f32 -> [?,N]f32
  %res = torch.prims.convert_element_type %mm, %i  : [?,N]f32 -> [?,N]si32

数值不重要 (L3 会替换为 func.call @cim_launch); 目标是 placeholder 全为标准 aten/prims
op, 能被 LINALG_ON_TENSORS pipeline 降级到 linalg.matmul。降级后的 linalg.matmul 会有
int8 cast 链 (sitofp si8->f32 / uitofp ui8->f32), L3 据此模式匹配, 区别于 CPU attention
的 f32 直运 matmul。

cim.matmul 语义: int8 激活 [?,K] x packed-ternary-uint8 权重 [N,K/4] -> int32 累加 [?,N]。
placeholder 把 W 当 uint8 repeat 4x (K/4->K) 模拟 unpack, 数值错但形状对 (L3 替换)。

用法:
  python cim_compiler/lowering/cim_lowering.py
  python cim_compiler/lowering/cim_lowering.py --in checkpoints/bitnet_ternary.mlir \\
    --out checkpoints/bitnet_ternary_placeholder.mlir
"""
import os
import sys
import re
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d

# torch.dtype -> torch.constant.int 的整数值 (见 cim_op.py / torch enum)
DTYPE_INT = {"f32": 6, "si32": 3, "si8": 1, "ui8": 0, "f64": 7, "si64": 4}

_VTENSOR_RE = re.compile(r"!torch\.vtensor<\[([^\]]+)\],\s*([^>]+)>")


def parse_vtensor(t):
    s = str(t)
    m = _VTENSOR_RE.match(s)
    if not m:
        raise ValueError(f"not a vtensor type: {s}")
    dims = [d.strip() for d in m.group(1).split(",")]
    dtype = m.group(2).strip()
    return dims, dtype


def vtensor(dims, dtype, ctx):
    return ir.Type.parse(f'!torch.vtensor<[{",".join(dims)}],{dtype}>', ctx)


def lower_cim_matmul(mod) -> int:
    """把 torch.operator "torch.cim.matmul" 替换为可降级 placeholder。返回替换数。"""
    ctx = mod.context
    targets = []

    def find(op):
        if str(op.name) == "torch.operator":
            name_attr = op.attributes.get("name")
            if name_attr is not None:
                val = name_attr.value if hasattr(name_attr, "value") else str(name_attr).strip('"')
                if val == "torch.cim.matmul":
                    targets.append(op)
        return ir.WalkResult.ADVANCE

    mod.operation.walk(find)

    list_int_ty = ir.Type.parse("!torch.list<int>", ctx)

    for op in targets:
        loc = op.location
        x, w = op.operands
        x_dims, _ = parse_vtensor(x.type)          # [?, K], si8
        w_dims, w_dt = parse_vtensor(w.type)        # [N, K/4], ui8
        r_dims, _ = parse_vtensor(op.results[0].type)  # [?, N], si32
        K = x_dims[1]
        N = w_dims[0]

        with ir.InsertionPoint(op):
            # dtype 常量
            f_const = torch_d.ConstantIntOp(DTYPE_INT["f32"], loc=loc).results[0]
            i_const = torch_d.ConstantIntOp(DTYPE_INT["si32"], loc=loc).results[0]
            one = torch_d.ConstantIntOp(1, loc=loc).results[0]
            four = torch_d.ConstantIntOp(int(K) // int(w_dims[1]), loc=loc).results[0]  # K/(K/4)=4

            # 1. X si8 -> f32
            xf_ty = vtensor(x_dims, "f32", ctx)
            xf = torch_d.PrimsConvertElementTypeOp(xf_ty, x, f_const, loc=loc).results[0]

            # 2. W [N,K/4]ui8 -> repeat [1,4] -> [N,K]ui8
            rep = torch_d.PrimListConstructOp(list_int_ty, [one, four], loc=loc).results[0]
            wr_ty = vtensor([N, K], w_dt, ctx)
            wr = torch_d.AtenRepeatOp(wr_ty, w, rep, loc=loc).results[0]

            # 3. W [N,K] -> t [K,N]
            wt_ty = vtensor([K, N], w_dt, ctx)
            wt = torch_d.AtenTOp(wt_ty, wr, loc=loc).results[0]

            # 4. W ui8 -> f32
            wf_ty = vtensor([K, N], "f32", ctx)
            wf = torch_d.PrimsConvertElementTypeOp(wf_ty, wt, f_const, loc=loc).results[0]

            # 5. mm: [?,K]f32 x [K,N]f32 -> [?,N]f32
            mm_ty = vtensor(r_dims, "f32", ctx)
            mm = torch_d.AtenMmOp(mm_ty, xf, wf, loc=loc).results[0]

            # 6. mm f32 -> si32
            res_ty = vtensor(r_dims, "si32", ctx)
            res = torch_d.PrimsConvertElementTypeOp(res_ty, mm, i_const, loc=loc).results[0]

        op.results[0].replace_all_uses_with(res)
        op.erase()

    return len(targets)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default="checkpoints/bitnet_ternary.mlir")
    p.add_argument("--out", default="checkpoints/bitnet_ternary_placeholder.mlir")
    args = p.parse_args()

    src = open(args.inp).read()
    ctx = ir.Context()
    torch_d.register_dialect(ctx)
    mod = ir.Module.parse(src, ctx)

    n = lower_cim_matmul(mod)

    out = str(mod)
    with open(args.out, "w") as f:
        f.write(out)

    n_op = out.count('torch.operator "torch.cim.matmul"')
    n_mm = out.count("torch.aten.mm")
    n_rep = out.count("torch.aten.repeat")
    n_cet = out.count("torch.prims.convert_element_type")
    print(f"[L1] {n} cim.matmul -> placeholder (aten.mm 仿真)", file=sys.stderr)
    print(f"[L1] 残留 torch.operator cim.matmul={n_op}, aten.mm={n_mm}, "
          f"aten.repeat={n_rep}, convert_element_type={n_cet}", file=sys.stderr)
    print(f"[L1] saved: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
