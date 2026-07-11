#ifndef HW_CONFIG_H
#define HW_CONFIG_H
#include <stdint.h>

/* CIM ASIC 硬件参数 (流片固化, 集中定义单一事实来源)。
 *
 * C (本文件) / Python (cim_compiler/cimres/hw_config.py) 镜像, 仿真器 (hw_simulator) /
 * 代码生成器 (emit_instr/place) / 驱动 (cim_stub) 共用, 防止同一硬件事实散落多处漂移。
 * 对应 cim_mlp.md §2.3 (寄存器) / §4.5 (Macro) / §4.6 (共享缓存三区) / §3.1 (指令)。
 *
 * ASIC 固化参数: 值永不变, 集中是为 DRY + 仿真器/生成器对齐 (非可重定向)。
 * 无前缀命名: 各引用文件原 #define / 常量同名, include 后删本地定义, 使用处零改动。
 */

/* tile / page (§4.6) */
#define TILE          64
#define PAGE          256
#define TILE_BYTES    (TILE * TILE / 4)   /* 2bit packed tile = 1024B = 4 PAGE */

/* 共享缓存 1MB 三区 (§4.6, page 索引) */
#define SHARED_SIZE   (1 << 20)           /* 1MB */
#define OVERWRITE_BASE 0x000              /* 覆盖区 (Preload 暂存 / Forward int8 输入) */
#define A_PAGE_BASE    0x010              /* Forward 输入区 (覆盖区内, bank0) */
#define INSTR_BASE     0xBF0              /* 指令区 (16 PAGE, 4KB) */
#define PSUM_PAGE_BASE 0xC00              /* 部分和累加区 (累加区, bank0) */
/* double buffer (S2 流水, A_PAGE/PSUM 各 2 套 ping-pong, m/m+1 隔离, §4.6) */
#define A_PAGE_BANK1_BASE  0x030          /* A_PAGE bank1 (bank0 0x010+kb 后, 留余量) */
#define PSUM_BANK1_BASE    0xC20          /* PSUM bank1 (bank0 0xC00+nb 后) */
#define A_BANK_OFF   (A_PAGE_BANK1_BASE - A_PAGE_BASE)   /* cim_stub patch: a_page += A_BANK_OFF */
#define P_BANK_OFF   (PSUM_BANK1_BASE - PSUM_PAGE_BASE)  /* cim_stub patch: psum_page += P_BANK_OFF */

/* 指令区容量 (48-bit 指令, 6B/条, §3.1) */
#define INSTR_PAGES    16
#define INSTR_CAPACITY (INSTR_PAGES * PAGE / 6)   /* 4KB/6B = 682 条 */
#define PRELOAD_BATCH  (INSTR_CAPACITY - 1)       /* 681, 留 1 SYNC_HALT */
#define SEG_MAX        PRELOAD_BATCH              /* 大段分块阈值 (cim_launch) */

/* Macro (§4.5) */
#define MACRO_MAX      4096                /* Macro 总数上限 */

/* 寄存器 (§2.3, 绝对地址 = REG_BASE + 偏移) */
#define REG_BASE       0x20000000ULL
#define DOORBELL_OFF   0x00
#define INT_CLEAR_OFF  0x04
#define IRQ_STATUS_OFF 0x08
#define IRQ_DONE       2                   /* IRQ_STATUS = done */

/* 指令 opcode (48-bit, §3.1) */
#define OP_PROG_WGT    0x1
#define OP_MATMUL      0x2
#define OP_SYNC_HALT   0x7

#endif /* HW_CONFIG_H */
