#!/usr/bin/env python3
"""L3: linalg 后插 func.call @cim_launch + canonicalize 清死链。

从 L1 产出 (placeholder.mlir) in-memory 跑 L2 pipeline (确保 tm_tensor dialect 注册, 因
Python 无 tm_tensor 注册 binding, 不能重新 parse linalg.mlir), 然后在 linalg IR 上:

对每个 placeholder 降级出的 linalg.matmul (37 个, 有 int8 cast 链), 追溯其输入到原始
int8 tensor, 替换整条 cast 链为一个 func.call @cim_launch_<idx>:

  追溯链 (placeholder 降级固定模式):
    ins[0] f32 <- linalg.generic{sitofp} <- X_si8  (tensor<?xKxi8>, 量化激活)
    ins[1] f32 <- linalg.generic{uitofp} <- linalg.transpose <- tensor.collapse_shape
                 <- linalg.generic{yield/broadcast} <- W_packed  (tensor<NxK/4xi8>, block arg)
    result  f32 -> linalg.generic{fptosi} -> si32  (tensor<?xNxi32>)

  替换为:
    %r = func.call @cim_launch_<idx>(X_si8, W_packed) : (tensor<?xKxi8>, tensor<NxK/4xi8>) -> tensor<?xNxi32>

  区别 CPU attention: attention 用 tm_tensor.attention (非 linalg.matmul), CPU 其他 matmul
  是 f32 直运 (无 sitofp/uitofp cast 链), 故 37 个 linalg.matmul 全是 placeholder 产物。

最后 canonicalize + cse 清死链 (cast generic/repeat/transpose/collapse/fill 等变 dead)。

用法:
  python cim_compiler/lowering/cim_stub_lower.py
  python cim_compiler/lowering/cim_stub_lower.py --in checkpoints/bitnet_ternary_placeholder.mlir \\
    --out checkpoints/bitnet_ternary_final.mlir
"""
import sys
import json
import argparse

from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d, func as func_d, arith as arith_d, tensor as tensor_d
from torch_mlir.fx import _module_lowering
from torch_mlir.compiler_utils import OutputType, run_pipeline_with_repro_report


def _owner(v):
    """Value -> defining Operation (OpResult.owner)。BlockArg 会触发下游 assert。"""
    return v.owner


def _block_ops_before(target_op):
    """target_op 所在 block 中 target_op 之前的 op id set (S6 阶段2 IR 变换用)。"""
    p = target_op.operation.parent
    # MLIR Python: parent 可能是 Block (有 operations) 或 enclosing Operation (func.func)
    block = p if hasattr(p, 'operations') else p.regions[0].blocks[0]
    s = set()
    target = target_op.operation
    for o in block.operations:
        if o == target:
            break
        s.add(o)  # Operation hash/eq 基于 pointer (跨 wrapper)
    return s


def _collect_movable(si32_vals, v_mm):
    """BFS 收集 si32_vals 的 transitive use op, 仅含 v_mm 前的 op (需移到 call result 后)。

    S6 阶段2 dominance 修复: q/k proj 后下游链 (si32->sitofp->RMSNorm div->reshape->
    expand_shape) 在 v proj 前, qkv call 在 v proj 前插入, call result 不 dominate 这些
    下游 op。收集 q/k si32 的 transitive use (仅 v_mm 前的), 移到 call result 后恢复
    dominance。非 si32 依赖的 operand (RMSNorm weight gamma / 形状计算 from_elements)
    不收集, 仍留在 v_mm 前, dominate 移动后的 op。
    """
    before = _block_ops_before(v_mm)
    seen = set()
    queue = list(si32_vals)
    val_seen = set(si32_vals)
    movable = []
    while queue:
        v = queue.pop(0)
        for use in v.uses:
            op = use.owner
            if op in seen:
                continue
            if op not in before:  # 只移 v_mm 前的 (v 链已在 v_mm 后, 不动)
                continue
            seen.add(op)
            movable.append(op)
            for r in op.results:
                if r not in val_seen:
                    val_seen.add(r)
                    queue.append(r)
    return movable


def _load_partition_meta(partition_path):
    """读 partition.json, 展开 cim_blocks 为 37 matmul 条目 (按 IR 顺序 = L3 walk 顺序)。

    方案B: 用 partition.json 元数据识别 qkv, 替代 shape 启发式 (Nq>Nk=Nv)。
    每条目 {name, is_qkv, role, blk_idx}; qkv 块展开 q/k/v 三个 (role=q/k/v),
    非 qkv 块 role=None。顺序 = cim_blocks idx 顺序 (qkv 展开), 与 L3 walk linalg.matmul
    顺序一致 (都按 IR 拓扑序, L1/L2 降级不改 op 顺序)。

    注: partition.json 的 w_packed_arg (input_specs 顺序) 与 MLIR %argN (torch-mlir 重排)
    不一致, 故不用 arg_number 映射, 改用 matmul 顺序映射 (IR walk 顺序 == cim_blocks 展开顺序)。
    """
    with open(partition_path) as f:
        d = json.load(f)
    entries = []
    for b in d["cim_blocks"]:
        if b.get("is_qkv"):
            entries.append({"name": b["bitlinear_name"], "is_qkv": True, "role": "q", "blk_idx": b["idx"]})
            entries.append({"name": b["bitlinear_name_k"], "is_qkv": True, "role": "k", "blk_idx": b["idx"]})
            entries.append({"name": b["bitlinear_name_v"], "is_qkv": True, "role": "v", "blk_idx": b["idx"]})
        else:
            entries.append({"name": b["bitlinear_name"], "is_qkv": False, "role": None, "blk_idx": b["idx"]})
    return entries


def lower_linalg_to_cim_call(mod, partition_path=None) -> int:
    """把 placeholder 降级的 linalg.matmul 替换为 func.call @cim_launch / @cim_launch_qkv。
    S6 shape 启发式: 3 连续 matmul W shape [Nq>Nk=Nv] (GQA) 识别 qkv, 合并为 cim_launch_qkv。返回 call 数。"""
    ctx = mod.context
    matmuls = []

    def find(op):
        if str(op.name) == "linalg.matmul":
            matmuls.append(op)
        return ir.WalkResult.ADVANCE

    mod.operation.walk(find)

    # declare @cim_launch (单输出) + @cim_launch_qkv (S6: 6 输入 Xq/Wq/Xk/Wk/Xv/Wv -> 3 输出 Q/K/V)
    i64_type = ir.IntegerType.get_signless(64, ctx)
    dyn_i8 = ir.Type.parse("tensor<?x?xi8>", ctx)
    dyn_i32 = ir.Type.parse("tensor<?x?xi32>", ctx)
    launch_ft = ir.FunctionType.get(inputs=[i64_type, dyn_i8, dyn_i8], results=[dyn_i32], context=ctx)
    launch_qkv_ft = ir.FunctionType.get(inputs=[i64_type, dyn_i8, dyn_i8, dyn_i8, dyn_i8, dyn_i8, dyn_i8],
                                        results=[dyn_i32, dyn_i32, dyn_i32], context=ctx)
    with ctx, ir.InsertionPoint.at_block_begin(mod.body):
        func_d.FuncOp(name="cim_launch", type=launch_ft, visibility="private", loc=ir.Location.unknown(ctx))
        func_d.FuncOp(name="cim_launch_qkv", type=launch_qkv_ft, visibility="private", loc=ir.Location.unknown(ctx))

    def parse_mm(mm):
        """追溯 matmul 的 X_si8 / W_packed(block arg) / si32_result + kill 链 ops。"""
        loc = mm.location
        sitofp_gen = _owner(mm.operands[0])
        X_si8 = sitofp_gen.operands[0]
        uitofp_gen = _owner(mm.operands[1])
        transpose = _owner(uitofp_gen.operands[0])
        collapse = _owner(transpose.operands[0])
        broadcast_gen = _owner(collapse.operands[0])
        W_packed = broadcast_gen.operands[0]   # block arg (BlockArgument)
        fptosi_gen = None
        for u in mm.results[0].uses:
            if str(u.owner.name) == "linalg.generic":
                fptosi_gen = u.owner
                break
        si32_result = fptosi_gen.results[0]
        fill = None
        outs_v = mm.operands[2]
        if hasattr(outs_v.owner, "name"):
            fill = outs_v.owner
        return {"X": X_si8, "W": W_packed, "si32": si32_result,
                "kill": [fptosi_gen, mm, uitofp_gen, transpose, collapse, broadcast_gen, sitofp_gen] + ([fill] if fill else []),
                "loc": loc}

    parsed = [parse_mm(mm) for mm in matmuls]   # 按 IR walk 顺序
    # 方案B: 用 partition.json 元数据 (role q/k/v + 同 blk_idx) 识别 qkv (shape 启发式已移除)
    if not partition_path:
        raise ValueError("方案B 需要 partition_path (partition.json 元数据识别 qkv)")
    entries = _load_partition_meta(partition_path)
    if len(entries) != len(parsed):
        raise ValueError(f"partition.json 展开 {len(entries)} 条目 != L3 walk {len(parsed)} matmul (顺序映射失效)")
    def kill_chain(p):
        for op in reversed(p["kill"]):
            if sum(1 for r in op.results for _ in r.uses) == 0:
                try: op.erase()
                except Exception: pass

    n_call = 0; i = 0; idx = 0
    while i < len(parsed):
        # qkv triplet 判定: partition.json 元数据 (方案B, role q/k/v + 同 blk_idx)
        is_qkv = False
        if i + 2 < len(parsed):
            e0, e1, e2 = entries[i], entries[i+1], entries[i+2]
            if (e0["is_qkv"] and e0["role"] == "q" and e1["role"] == "k" and e2["role"] == "v"
                    and e0["blk_idx"] == e1["blk_idx"] == e2["blk_idx"]):
                is_qkv = True
        if is_qkv:
                q, k, v = parsed[i], parsed[i+1], parsed[i+2]
                v_mm = matmuls[i+2]
                # S6 阶段2 IR 变换 pass: q/k proj 后下游链 (si32->sitofp->RMSNorm->reshape)
                # 在 v proj 前, qkv call result 不 dominate。收集 q/k si32 transitive use
                # (v_mm 前), 移到 call result 后恢复 dominance。
                movable = _collect_movable([q["si32"], k["si32"]], v_mm)
                _p = v_mm.operation.parent
                _blk = _p if hasattr(_p, 'operations') else _p.regions[0].blocks[0]
                order = {o: i for i, o in enumerate(_blk.operations)}
                with ir.InsertionPoint(v_mm):
                    idx_const = arith_d.ConstantOp(i64_type, ir.IntegerAttr.get(i64_type, idx), loc=q["loc"])
                    qd = tensor_d.CastOp(dyn_i8, q["X"], loc=q["loc"]).result
                    qw = tensor_d.CastOp(dyn_i8, q["W"], loc=q["loc"]).result
                    kd = tensor_d.CastOp(dyn_i8, k["X"], loc=k["loc"]).result
                    kw = tensor_d.CastOp(dyn_i8, k["W"], loc=k["loc"]).result
                    vd = tensor_d.CastOp(dyn_i8, v["X"], loc=v["loc"]).result
                    vw = tensor_d.CastOp(dyn_i8, v["W"], loc=v["loc"]).result
                    call = func_d.CallOp([dyn_i32, dyn_i32, dyn_i32], "cim_launch_qkv",
                                         [idx_const.result, qd, qw, kd, kw, vd, vw], loc=q["loc"])
                    q_res = tensor_d.CastOp(q["si32"].type, call.results[0], loc=q["loc"]).result
                    k_res = tensor_d.CastOp(k["si32"].type, call.results[1], loc=k["loc"]).result
                    v_res = tensor_d.CastOp(v["si32"].type, call.results[2], loc=v["loc"]).result
                    # 按 IR 顺序移 q/k 链 op 到 v_res 后 (q_res/k_res/v_res 在前, dominate)
                    anchor = v_res.owner
                    for op in sorted(movable, key=lambda o: order.get(o, 0)):
                        op.move_after(anchor)
                        anchor = op
                q["si32"].replace_all_uses_with(q_res)
                k["si32"].replace_all_uses_with(k_res)
                v["si32"].replace_all_uses_with(v_res)
                for p in (q, k, v): kill_chain(p)
                i += 3; idx += 1; n_call += 1; continue
        p = parsed[i]
        with ir.InsertionPoint(matmuls[i]):
            idx_const = arith_d.ConstantOp(i64_type, ir.IntegerAttr.get(i64_type, idx), loc=p["loc"])
            x_dyn = tensor_d.CastOp(dyn_i8, p["X"], loc=p["loc"]).result
            w_dyn = tensor_d.CastOp(dyn_i8, p["W"], loc=p["loc"]).result
            call = func_d.CallOp([dyn_i32], "cim_launch", [idx_const.result, x_dyn, w_dyn], loc=p["loc"])
            res_back = tensor_d.CastOp(p["si32"].type, call.results[0], loc=p["loc"]).result
        p["si32"].replace_all_uses_with(res_back)
        kill_chain(p)
        i += 1; idx += 1; n_call += 1

    return n_call


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default="checkpoints/bitnet_ternary_placeholder.mlir")
    p.add_argument("--out", default="checkpoints/bitnet_ternary_final.mlir")
    p.add_argument("--save-linalg", default="checkpoints/bitnet_ternary_linalg.mlir",
                   help="保存 L2 中间 linalg IR (调试)")
    p.add_argument("--partition", default="checkpoints/bitnet_ternary_partition.json",
                   help="S6 方案B: partition.json (建 arg->bitlinear_name 识别 qkv 合并)")
    args = p.parse_args()

    src = open(args.inp).read()
    ctx = ir.Context()
    torch_d.register_dialect(ctx)
    mod = ir.Module.parse(src, ctx)

    print(f"[L3] 加载 {args.inp}", file=sys.stderr)
    print(f"[L3] 跑 LINALG pipeline (in-memory, 注册 tm_tensor)...", file=sys.stderr)
    _module_lowering(False, False, OutputType.LINALG_ON_TENSORS, mod)

    if args.save_linalg:
        with open(args.save_linalg, "w") as f:
            f.write(str(mod))
        print(f"[L3] saved linalg intermediate: {args.save_linalg}", file=sys.stderr)

    print(f"[L3] 插 func.call @cim_launch_<idx> (替换 placeholder linalg.matmul)...", file=sys.stderr)
    n = lower_linalg_to_cim_call(mod, args.partition)
    try:
        mod.operation.verify()
        print(f"[L3] lower 后 verify OK, n_call={n}", file=sys.stderr)
    except Exception as e:
        print(f"[L3] lower 后 verify FAIL: {e}", file=sys.stderr)

    print(f"[L3] canonicalize + cse 清死链...", file=sys.stderr)
    run_pipeline_with_repro_report(mod, "builtin.module(canonicalize, cse)", "L3 canonicalize+cse")
    mod.operation.verify()
    print(f"[L3] final IR verify OK ✓", file=sys.stderr)

    out = str(mod)
    with open(args.out, "w") as f:
        f.write(out)

    n_call = out.count("call @cim_launch")
    n_decl = out.count("func.func private @cim_launch")
    n_linalg_mm = out.count("linalg.matmul")
    n_linalg_gen = out.count("linalg.generic")
    n_op = out.count("torch.operator")
    n_aten = out.count("torch.aten.")
    print(f"[L3] {n} linalg.matmul -> func.call @cim_launch_<idx>", file=sys.stderr)
    print(f"[L3] call={n_call}, 声明={n_decl}, linalg.matmul={n_linalg_mm} (应=0), "
          f"linalg.generic={n_linalg_gen}, torch.operator={n_op}, torch.aten={n_aten}",
          file=sys.stderr)
    print(f"[L3] saved: {args.out}", file=sys.stderr)

    # S6: 37 matmul -> 6 qkv 合并 (cim_launch_qkv) + 19 单 (cim_launch) = 25 call, 2 声明
    n_qkv = out.count("call @cim_launch_qkv")
    n_single = n_call - n_qkv
    print(f"[L3] 其中 cim_launch_qkv={n_qkv}, cim_launch(单)={n_single}", file=sys.stderr)
    ok = True
    if n_call != 25:
        print(f"[L3] FAIL: call={n_call} (应=25: 6 qkv + 19 单)", file=sys.stderr); ok = False
    if n_decl != 2:
        print(f"[L3] FAIL: 声明={n_decl} (应=2: cim_launch + cim_launch_qkv)", file=sys.stderr); ok = False
    if n_linalg_mm != 0:
        print(f"[L3] FAIL: 残留 linalg.matmul={n_linalg_mm} (应=0, placeholder 死链未清)", file=sys.stderr); ok = False
    if n_op != 0:
        print(f"[L3] FAIL: 残留 torch.operator={n_op}", file=sys.stderr); ok = False
    print(f"\n[L3] {'PASS ✓ (37 func.call stub 保留, placeholder 死链清)' if ok else 'FAIL ✗'}",
          file=sys.stderr)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
