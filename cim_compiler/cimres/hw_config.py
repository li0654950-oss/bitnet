"""CIM ASIC 硬件参数 (流片固化, 集中定义单一事实来源)。

Python 镜像 C (cim_compiler/lowering/hw_config.h), 仿真器 (hw_simulator) /
代码生成器 (emit_instr / place) 共用, 防止同一硬件事实散落多处漂移。
对应 cim_mlp.md §2.3 (寄存器) / §4.5 (Macro) / §4.6 (共享缓存三区) / §3.1 (指令)。

ASIC 固化参数: 值永不变, 集中是为 DRY + 仿真器/生成器对齐 (非可重定向)。
无前缀命名: 各引用文件原常量同名, from import 后删本地定义, 使用处零改动。
"""

import math as _math   # _前缀: 不被 from hw_config import * 带入, 避免污染消费方命名空间

# tile / page (§4.6)
TILE = 64

# 权重/数据编码派生常量 (消除散落 //4 *4 1024 魔数; §4.5 2bit 三值打包, §3.2 PROG_WGT)
BIT_PER_WEIGHT = 2                          # 三值权重 {-1,0,1} 2bit 补码打包
CODES_PER_BYTE = 8 // BIT_PER_WEIGHT        # 4: 每字节 4 个 2bit code (emit_instr K//4 / hw_simulator reshape TILE//4 源此)
I32_BYTES = 4                               # sizeof(int32) (PSUM_BYTES=TILE*4 源此; 与 CODES_PER_BYTE=4 含义不同, 勿混)
TILE_BYTES = TILE * TILE // CODES_PER_BYTE  # 2bit packed tile = 1024B @TILE=64

# 时序派生 (T_MATMUL = ADC 串行列扫, 每 cycle 量化 1 列, TILE 列 = TILE cycle; §3.3)
T_MATMUL = TILE                             # 64 @TILE=64 (原散落 hw_simulator.py:43, 移此使 cost_model 随 TILE 联动)

# PAGE 派生: PAGE = TILE*I32_BYTES (PSUM 恰好 1 PAGE/n_blk 无浪费, PAGE 随 TILE 变; §4.6)
PAGE = TILE * I32_BYTES   # 4*TILE; @TILE=64->256, @TILE=128->512, @TILE=32->128

# PAGE 布局派生 (PAGE=4*TILE -> PSUM 1 PAGE/n_blk 不跨页, 计算简便高效)
PAGES_PER_TILE = TILE_BYTES // PAGE         # 权重 tile 占 PAGE 数 = TILE/16; @TILE=64->4, @TILE=128->8
PSUM_PAGES_PER_NBLK = 1                     # PAGE=4*TILE -> PSUM(4*TILE)=1 PAGE, 恒不跨页 (无跨页 RMW)

# 48-bit 指令 page 字段位宽 (扩 14 bit 借保留位 [7:4]; §3.1, 支持 TILE=16 PAGE=64 -> 16384 PAGE)
PAGE_BITS = 14
PAGE_MASK = (1 << PAGE_BITS) - 1            # 0x3FFF = 16384

# 物理不变量守护
assert TILE % CODES_PER_BYTE == 0, \
    f"TILE={TILE} 必须整除 CODES_PER_BYTE={CODES_PER_BYTE} (hw_simulator reshape 整除)"
assert TILE & (TILE - 1) == 0, \
    f"TILE={TILE} 必须为 2 的幂 (PAGE=4*TILE 硬件 page*PAGE 移位译码; 16/32/64/128/256)"

# 共享缓存 1MB 三区 (§4.6, byte 边界固定, page 索引随 PAGE 派生)
SHARED_SIZE = 1 << 20           # 1MB
OVERWRITE_BYTE = 0x00000        # 覆盖区起点 (Preload 暂存 / Forward int8 输入)
A_PAGE_BYTE    = 0x01000        # A_PAGE 区 (覆盖区内, bank0; 4KB)
INSTR_BYTE     = 0xBF000        # 指令区起点 (4KB)
PSUM_BYTE      = 0xC0000        # 累加区起点 (256KB)
# page 索引 (随 PAGE 派生; @PAGE=256 同原 0x010/0xBF0/0xC00)
OVERWRITE_BASE = OVERWRITE_BYTE // PAGE
A_PAGE_BASE    = A_PAGE_BYTE // PAGE
INSTR_BASE     = INSTR_BYTE // PAGE
PSUM_PAGE_BASE = PSUM_BYTE // PAGE
# double buffer (S2 流水, A_PAGE/PSUM 各 2 套 ping-pong, m/m+1 隔离, §4.6; byte 偏移固定)
A_PAGE_BANK1_BYTE = A_PAGE_BYTE + 0x5000   # bank1 (容 qkv 3*k_tiles; @PAGE=256 -> 0x060)
PSUM_BANK1_BYTE   = PSUM_BYTE + 0x6000     # bank1 (容 max n_tiles; @PAGE=256 -> 0xC60)
A_PAGE_BANK1_BASE = A_PAGE_BANK1_BYTE // PAGE
PSUM_BANK1_BASE   = PSUM_BANK1_BYTE // PAGE
A_BANK_OFF = A_PAGE_BANK1_BASE - A_PAGE_BASE   # cim_stub patch: a_page += A_BANK_OFF
P_BANK_OFF = PSUM_BANK1_BASE - PSUM_PAGE_BASE  # cim_stub patch: psum_page += P_BANK_OFF

# 指令区容量 (4KB byte 固定, 48-bit 6B/条, §3.1; 不依赖 PAGE)
INSTR_BYTE_SIZE = 0x1000        # 4KB
INSTR_CAPACITY = INSTR_BYTE_SIZE // 6   # 682 条
PRELOAD_BATCH = min(INSTR_CAPACITY - 1, (INSTR_BASE - OVERWRITE_BASE) // PAGES_PER_TILE)  # min(指令区, 覆盖区/tile)
SEG_MAX = PRELOAD_BATCH                    # 大段分块阈值 (cim_launch)

# Macro (§4.5)
MACRO_MAX = 65536              # Macro 总数上限 (Dest_ID 16b 方案B 借保留位 [3:0]; 原 4096=12b)

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
