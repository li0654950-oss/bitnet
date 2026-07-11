"""cimres pass 共用基础设施: IR 加载 / 遍历 / 保存。

封装 torch-mlir Python binding 的 Context 创建 + cimres dialect 注册 + 遍历,
让 canonicalize/cse/verify 聚焦逻辑而非样板代码。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))          # cim_compiler/cimres/passes/
CIMRES = os.path.dirname(HERE)                             # cim_compiler/cimres/
CIM_COMPILER = os.path.dirname(CIMRES)                     # cim_compiler/
REPO = os.path.dirname(CIM_COMPILER)                       # repo root
for _p in (REPO,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from torch_mlir import ir
from cim_compiler.cimres.dialect import register_cimres


def load_cimres(path):
    """加载 cimres IR 文件 -> (mod, ctx)。注册 cimres dialect (IRDL)。"""
    ctx = ir.Context()
    ctx.load_all_available_dialects()
    register_cimres(ctx)
    with ctx:
        mod = ir.Module.parse(open(path).read(), ctx)
    return mod, ctx


def parse_cimres(src):
    """从字符串解析 cimres IR -> (mod, ctx)。错误注入测试用。"""
    ctx = ir.Context()
    ctx.load_all_available_dialects()
    register_cimres(ctx)
    with ctx:
        mod = ir.Module.parse(src, ctx)
    return mod, ctx


def save(mod, path):
    """序列化 mod 到文件 (round-trip 可读)。"""
    with open(path, "w") as f:
        f.write(str(mod))


def walk_ops(mod, name):
    """递归收集模块内所有名为 name 的 op (含 func.func 内)。

    walk 回调收到 Operation; op.name 返回 identifier (如 "cimres.macro_matmul")。
    """
    out = []

    def cb(op):
        if str(op.name) == name:
            out.append(op)
        return ir.WalkResult.ADVANCE

    mod.operation.walk(cb)
    return out


def func_blocks(mod):
    """遍历所有 func.func -> (func_op, entry_block)。pass 修改 func 内序列用。"""
    result = []
    for op in list(mod.body):
        if op.operation.name == "func.func":
            result.append((op, op.regions[0].blocks[0]))
    return result


def matmuls_in_func(func_op):
    """收集 func.func 内所有 macro_matmul (按 IR 顺序) 的属性 dict。"""
    from cim_compiler.cimres.dialect import attr_i32, attr_bool, attr_str
    blk = func_op.regions[0].blocks[0]
    out = []
    for op in list(blk.operations):
        if op.operation.name != "cimres.macro_matmul":
            continue
        out.append({
            "op": op,
            "dest_id": attr_i32(op, "dest_id"),
            "a_page": attr_i32(op, "a_page"),
            "psum_page": attr_i32(op, "psum_page"),
            "accum": attr_bool(op, "accum"),
            "n_blk": attr_i32(op, "n_blk"),
            "k_blk": attr_i32(op, "k_blk"),
            "bitlinear_name": attr_str(op, "bitlinear_name"),
        })
    return out
