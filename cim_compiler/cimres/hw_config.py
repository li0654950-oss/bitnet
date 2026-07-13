"""CIM ASIC 硬件参数 (流片固化, 集中定义单一事实来源)。

Python 镜像 C (cim_compiler/lowering/hw_config.h), 仿真器 (hw_simulator) /
代码生成器 (emit_instr / place) 共用, 防止同一硬件事实散落多处漂移。
对应 cim_mlp.md §2.3 (寄存器) / §4.5 (Macro) / §4.6 (共享缓存三区) / §3.1 (指令)。

ASIC 固化参数: 值永不变, 集中是为 DRY + 仿真器/生成器对齐 (非可重定向)。
无前缀命名: 各引用文件原常量同名, from import 后删本地定义, 使用处零改动。
"""

# tile / page (§4.6)
TILE = 64
PAGE = 256
TILE_BYTES = TILE * TILE // 4   # 2bit packed tile = 1024B = 4 PAGE

# 共享缓存 1MB 三区 (§4.6, page 索引)
SHARED_SIZE = 1 << 20           # 1MB
OVERWRITE_BASE = 0x000          # 覆盖区 (Preload 暂存 / Forward int8 输入)
A_PAGE_BASE = 0x010             # Forward 输入区 (覆盖区内, bank0)
INSTR_BASE = 0xBF0              # 指令区 (16 PAGE, 4KB)
PSUM_PAGE_BASE = 0xC00          # 部分和累加区 (累加区, bank0)
# double buffer (S2 流水, A_PAGE/PSUM 各 2 套 ping-pong, m/m+1 隔离, §4.6)
A_PAGE_BANK1_BASE = 0x030       # A_PAGE bank1 (bank0 0x010+kb 后, 留余量)
PSUM_BANK1_BASE = 0xC20         # PSUM bank1 (bank0 0xC00+nb 后)
A_BANK_OFF = A_PAGE_BANK1_BASE - A_PAGE_BASE   # cim_stub patch: a_page += A_BANK_OFF
P_BANK_OFF = PSUM_BANK1_BASE - PSUM_PAGE_BASE  # cim_stub patch: psum_page += P_BANK_OFF
ACCUM_BASE = PSUM_PAGE_BASE     # 累加区别名 (hw_simulator 用)

# 指令区容量 (48-bit 指令, 6B/条, §3.1)
INSTR_PAGES = 16
INSTR_CAPACITY = INSTR_PAGES * PAGE // 6   # 4KB/6B = 682 条
PRELOAD_BATCH = INSTR_CAPACITY - 1         # 681, 留 1 SYNC_HALT
SEG_MAX = PRELOAD_BATCH                    # 大段分块阈值 (cim_launch)

# Macro (§4.5)
MACRO_MAX = 4096               # Macro 总数上限

# 寄存器 (§2.3, 绝对地址 = REG_BASE + 偏移)
REG_BASE = 0x20000000
REG_BASE_DEFAULT = REG_BASE     # hw_simulator 用
DOORBELL_OFF = 0x00
INT_CLEAR_OFF = 0x04
IRQ_STATUS_OFF = 0x08
IRQ_DONE = 2

# IRQ 状态机 (§2.3)
IDLE, BUSY, DONE, ERROR = 0, 1, 2, 3

# 指令 opcode (48-bit, §3.1)
OP_PROG_WGT = 0x1
OP_MATMUL = 0x2
OP_SYNC_HALT = 0x7

# ===== S4 布局延迟模型 (2D Mesh NoC, 研究性假设, 非 spec) =====
# 注: cim_mlp.md §2.2 定义 CIM 为内部广播总线 (BusDispatcher 同时广播全部 4096 Macro,
# T_DISPATCH=2 固定, 与 dest 位置无关), 规范无 2D Mesh/路由器。故:
#   T_ROUT_PER_HOP = 0   -> spec-faithful 广播总线基线 (cycle=6268, 真实模型, 默认)
#   T_ROUT_PER_HOP > 0   -> "广播总线换 Mesh NoC" 研究假设 (无 spec 依据, cycle 上升非下降)
# Mesh+layout 在所有 T_ROUT 下劣于广播总线; hotspot -9.6% 是相对 Mesh 假设基线的减损,
# 非相对真实硬件提升。保留作 layout 优化研究 + autotuner 框架, 不产生实际性能收益。
MESH_DIM = 64                      # 64x64 Mesh = 4096 Macro (研究假设拓扑, 非 spec)
T_ROUT_PER_HOP = 0                 # 0=spec 广播总线基线; >0=Mesh NoC 研究假设 (路由跳数 cycle)
LAYOUT_MAP = None                  # {dest_id: (x,y)} 重映射 (None=线性扫描; macro_layout 设置)

def dest_to_xy(dest_id, mesh_dim=MESH_DIM):
    """dest_id -> (x, y) 物理坐标 (LAYOUT_MAP 设置则重映射, 否则线性扫描)。"""
    if LAYOUT_MAP is not None:
        xy = LAYOUT_MAP.get(dest_id)
        if xy is not None:
            return xy
    return dest_id % mesh_dim, dest_id // mesh_dim

def dest_origin_hops(dest_id, mesh_dim=MESH_DIM):
    """dest 到原点(0,0)的 Manhattan 距离 = 指令分发路由跳数 (总线入口假设在原点)。"""
    x, y = dest_to_xy(dest_id, mesh_dim)
    return x + y
