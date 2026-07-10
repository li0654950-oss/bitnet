"""cimres - CIM 协处理器指令流降级 (cim.matmul -> 48-bit 指令流)。

独立编译路径: 从 partition.json + weights.bin 出发, 不改导出/划分/CPU降级 (L1-L6)。
动态 M (seq_len) 由运行时循环吸收, 单 token 指令流编译期静态 (方案 C, 不改导出)。

模块:
  dialect         : cimres IRDL 方言定义 + 注册 + op 构造 helper (阶段0)
  lower_to_cimres : C1 逻辑 tile 展开 (cim.matmul -> macro_matmul tile 序列)
  place           : C2 资源映射 (Macro Dest_ID + PAGE 布局)
  emit_instr      : C3 48-bit 指令编码 (cimres -> MACRO_PROG_WGT/MATMUL/SYNC_HALT)
  simulator       : C4 CIM 指令流模拟器 (取指执行 + 对齐 PyTorch 参考)
  runtime         : C5 运行时动态 M 驱动
"""
from .dialect import (
    CIMRES_IRDL, register_cimres,
    int8_vec, int32_vec, i32_attr, bool_attr, str_attr,
    make_macro_matmul, make_preload_weight, make_sync_halt, make_tile_group,
    attr_i32, attr_bool, attr_str,
)

__all__ = [
    "CIMRES_IRDL", "register_cimres",
    "int8_vec", "int32_vec", "i32_attr", "bool_attr", "str_attr",
    "make_macro_matmul", "make_preload_weight", "make_sync_halt", "make_tile_group",
    "attr_i32", "attr_bool", "attr_str",
]
