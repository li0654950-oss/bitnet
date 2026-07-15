"""cim_compiler.lowering: IR 降级工程 (L1 cim_lowering -> L4 linalg_to_llvm + AOT)。

各 .py 多为独立可运行脚本 (python cim_compiler/lowering/X.py) 或被 _load 动态加载;
本 __init__.py 仅作包标记, 使 from cim_compiler.lowering.buffer_kind import 可用。
"""
