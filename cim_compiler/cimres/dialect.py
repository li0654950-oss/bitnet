#!/usr/bin/env python3
"""cimres 方言 - CIM 协处理器指令流降级的 IR 中间表示 (IRDL 注册, 纯 Python 零编译)。

cimres op 对应 cim_mlp.md §1.3 MLIR 层映射 + §3 ISA:
  cimres.macro_matmul   : 映射 MACRO_MATMUL (§3.3)。1 PAGE int8 向量 × 64×64 三值权重 -> int32 部分和向量。
                          attrs: dest_id(Macro), a_page(输入页), psum_page(输出页),
                                 accum(0=覆盖/1=累加), n_blk, k_blk, bitlinear_name, role(q/k/v/none)
  cimres.preload_weight : 映射 MACRO_PROG_WGT (§3.2)。2bit 三值 tile 预载到 Macro。
                          attrs: dest_id, b_page_start(权重数据起始页), bitlinear_name
  cimres.sync_halt      : 映射 SYNC_HALT (§3.4)。同步屏障, 等 Macro 完成 + IRQ。
  cimres.tile_group     : 元数据 op, 标记一个 BitLinear 的 tile 序列边界。
                          attrs: bitlinear_name, N, K

注册方式: IRDL (irdl.load_dialects), 无需 C++ 编译。
  - torch-mlir wheel (LLVM 23) 与 mlir-tutorial (LLVM 18) ABI 不兼容, C++ dialect 不可行
  - IRDL 纯 Python, irdl.load_dialects 真正注册 (is_registered_operation=True), 有约束验证
IRDL 约束语义: 同一约束变量被多处引用 = 值相等约束, 故每个属性用独立 irdl.any。
operands/results 用带标签语法 irdl.operands(a:%ta) (LLVM 23 wheel 版本)。

运行 (self-test): nanogpt-gpu python cim_compiler/cimres/dialect.py
"""
from torch_mlir import ir
from torch_mlir.dialects import irdl

from cim_compiler.cimres.hw_config import TILE  # Macro 物理维度 (IR ABI 默认 n=TILE, 改 TILE 即改 op 签名 -> placed.mlir 失效需重跑 C1-C3)

# IRDL 文本定义 cimres dialect。每个属性独立 irdl.any (避免相等约束)。
CIMRES_IRDL = r'''irdl.dialect @cimres {
  irdl.operation @macro_matmul {
    %a1 = irdl.any
    %a2 = irdl.any
    %a3 = irdl.any
    %a4 = irdl.any
    %a5 = irdl.any
    %a6 = irdl.any
    %a7 = irdl.any
    %a8 = irdl.any
    %ta = irdl.any
    %tr = irdl.any
    irdl.attributes {
      "dest_id" = %a1,
      "a_page" = %a2,
      "psum_page" = %a3,
      "accum" = %a4,
      "n_blk" = %a5,
      "k_blk" = %a6,
      "bitlinear_name" = %a7,
      "role" = %a8
    }
    irdl.operands(a:%ta)
    irdl.results(r:%tr)
  }
  irdl.operation @preload_weight {
    %a1 = irdl.any
    %a2 = irdl.any
    %a3 = irdl.any
    irdl.attributes {
      "dest_id" = %a1,
      "b_page_start" = %a2,
      "bitlinear_name" = %a3
    }
  }
  irdl.operation @sync_halt {
  }
  irdl.operation @tile_group {
    %a1 = irdl.any
    %a2 = irdl.any
    %a3 = irdl.any
    irdl.attributes {
      "bitlinear_name" = %a1,
      "N" = %a2,
      "K" = %a3
    }
  }
}
'''


def register_cimres(ctx: "ir.Context"):
    """在 ctx 注册 cimres dialect (IRDL)。须在 `with ctx:` 作用域内调用。"""
    m = ir.Module.parse(CIMRES_IRDL, context=ctx)
    irdl.load_dialects(m)
    for opn in ("cimres.macro_matmul", "cimres.preload_weight",
                "cimres.sync_halt", "cimres.tile_group"):
        assert ctx.is_registered_operation(opn), f"{opn} 未注册"
    return ctx


# ---- 类型 helper ----
def int8_vec(ctx, n=TILE):
    """tensor<nxi8> : Macro 输入 int8 特征向量 (1 PAGE 装 TILE int8, TILE<=PAGE=256)。"""
    return ir.Type.parse(f"tensor<{n}xi8>", ctx)


def int32_vec(ctx, n=TILE):
    """tensor<nxi32> : Macro 输出 int32 部分和向量 (PSUM 跨 PSUM_PAGES_PER_NBLK PAGE, TILE>64 跨页)。"""
    return ir.Type.parse(f"tensor<{n}xi32>", ctx)


# ---- 属性 helper (在 `with ctx:` 作用域调用) ----
def i32_attr(v):
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), int(v))


def bool_attr(v):
    return ir.BoolAttr.get(bool(v))


def str_attr(s):
    return ir.StringAttr.get(str(s))


# ---- op 构造 helper (须在 `with ctx:` + InsertionPoint 作用域) ----
def make_macro_matmul(ctx, x, dest_id, a_page, psum_page, accum,
                      n_blk, k_blk, bitlinear_name, role):
    """构造 cimres.macro_matmul。x = int8 向量 value, 返回 int32 result value。
    role: "q"/"k"/"v" (qkv 合并 func 内) 或 "none" (普通 BitLinear), C2 place 据此错开 PAGE。"""
    op = ir.Operation.create(
        "cimres.macro_matmul",
        results=[int32_vec(ctx)], operands=[x],
        attributes={
            "dest_id": i32_attr(dest_id),
            "a_page": i32_attr(a_page),
            "psum_page": i32_attr(psum_page),
            "accum": bool_attr(accum),
            "n_blk": i32_attr(n_blk),
            "k_blk": i32_attr(k_blk),
            "bitlinear_name": str_attr(bitlinear_name),
            "role": str_attr(role),
        },
    )
    return op.result


def make_preload_weight(dest_id, b_page_start, bitlinear_name):
    """构造 cimres.preload_weight (0 operand 0 result)。返回 op。"""
    op = ir.Operation.create(
        "cimres.preload_weight", results=[], operands=[],
        attributes={
            "dest_id": i32_attr(dest_id),
            "b_page_start": i32_attr(b_page_start),
            "bitlinear_name": str_attr(bitlinear_name),
        },
    )
    return op


def make_sync_halt():
    """构造 cimres.sync_halt (0 operand 0 result)。返回 op。"""
    return ir.Operation.create("cimres.sync_halt", results=[], operands=[])


def make_tile_group(bitlinear_name, N, K):
    """构造 cimres.tile_group (元数据, 0 operand 0 result)。返回 op。"""
    op = ir.Operation.create(
        "cimres.tile_group", results=[], operands=[],
        attributes={
            "bitlinear_name": str_attr(bitlinear_name),
            "N": i32_attr(N),
            "K": i32_attr(K),
        },
    )
    return op


# ---- 解析 helper (遍历 cimres IR 读属性) ----
def attr_i32(op, name):
    return int(op.attributes[name].value)


def attr_bool(op, name):
    return bool(op.attributes[name].value)


def attr_str(op, name):
    return str(op.attributes[name].value)


def _self_test():
    """构造一个简化 tile 序列 (2 tile K 维累加 + preload + sync_halt), verify + round-trip。"""
    ctx = ir.Context()
    ctx.load_all_available_dialects()
    register_cimres(ctx)
    from torch_mlir.dialects import func as func_d

    with ctx, ir.Location.unknown():
        mod = ir.Module.create()
        with ir.InsertionPoint(mod.body):
            make_tile_group("layers.0.attn.q.proj", 512, 512)
            make_preload_weight(dest_id=0, b_page_start=0x000, bitlinear_name="layers.0.attn.q.proj")
            make_preload_weight(dest_id=1, b_page_start=0x004, bitlinear_name="layers.0.attn.q.proj")

            func_ty = ir.FunctionType.get([int8_vec(ctx)], [int32_vec(ctx)], context=ctx)
            f = func_d.FuncOp(name="q_proj", type=func_ty)
            entry = f.add_entry_block()
            with ir.InsertionPoint(entry):
                x = entry.arguments[0]
                # K 维 2 tile 累加: 首个 ACCUM=0 覆盖, 后续 ACCUM=1 累加
                y0 = make_macro_matmul(ctx, x, dest_id=0, a_page=0x010,
                                       psum_page=0xC00, accum=False,
                                       n_blk=0, k_blk=0,
                                       bitlinear_name="layers.0.attn.q.proj", role="q")
                y1 = make_macro_matmul(ctx, x, dest_id=1, a_page=0x011,
                                       psum_page=0xC00, accum=True,
                                       n_blk=0, k_blk=1,
                                       bitlinear_name="layers.0.attn.q.proj", role="q")
                make_sync_halt()
                func_d.ReturnOp([y1])

        mod.operation.verify()
        print("[cimres] dialect self-test: verify OK ✓")
        ir.Module.parse(str(mod), context=ctx)
        print("[cimres] round-trip OK ✓")
        print("---- cimres IR ----")
        print(mod)


if __name__ == "__main__":
    _self_test()
