/* L5: @cim_launch 硬件驱动 (C, .so, 返回 struct by value, MMIO 读写驱动 hw_simulator)。
 *
 * @cim_launch_<idx> LLVM calling convention (L4 产出):
 *   (X 2D memref, W 2D memref) -> result 2D memref
 *   每个 2D memref 7 参数: (allocated, aligned, offset, size0, size1, stride0, stride1)
 *   X: si8 [M, K]  per-token int8 激活
 *   W: ui8 [N, K/4]  2bit 补码打包三值权重 (Forward 不用, Preload 已驻留 Macro)
 *   -> result: si32 [M, N]  累加输出
 *
 * MMIO 驱动纯硬件仿真器 (hw_simulator.py): cim_launch_<idx> / cim_preload_init 通过
 *   MMIO 读写 (shm_write/shm_read/reg_write/reg_read) 操作 hw_simulator:
 *   写共享缓存 A_PAGE/指令区, 写门铃, poll IRQ_STATUS, 读 PSUM_PAGE acc (§2.3/§4.7)。
 *   MMIO 回调由 register_cim_hw_sim 注册 (cim_jit.py)。
 *
 * 地址布局对齐 cim_mlp.md:
 *   寄存器 (§2.3, 相对 REG_BASE): DOORBELL=0x00 INT_CLEAR=0x04 IRQ_STATUS=0x08
 *   共享缓存 (§4.6, PAGE=256B): 覆盖区 0x000~0xBEF | 指令区 0xBF0~0xBFF | 累加区 0xC00~0xFFF
 *   (shm 用 byte offset 索引 hw_simulator 共享缓存; reg 用绝对地址路由寄存器)
 *
 * 产物加载 (C3 产出):
 *   cim_load_forward(forward.bin)  - 按 idx 索引的 MATMUL 指令段 (cim_launch_<idx> 用 idx 查)
 *   cim_preload_init(preload.bin)  - 自包含 Preload (tile 数据 + PROG_WGT, 一次性, §4.2)
 *
 * fallback: 未注册 MMIO 回调 (HW_READY=0) 时 cim_launch_impl CPU 算 matmul (兼容无仿真路径)。
 *
 * 编译: cc -O2 -shared -fPIC cim_stub.c -o cim_stub.so
 *        + cim_jit.py register_cim_hw_sim(...) + cim_load_forward + cim_preload_init
 */

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    void *allocated;
    void *aligned;
    int64_t offset;
    int64_t size0, size1;
    int64_t stride0, stride1;
} Memref2D;

/* ===== 地址布局 (对齐 cim_mlp.md §2.3/§4.6) =====
 * 寄存器 (§2.3, 相对 REG_BASE, reg 路由用): DOORBELL=0x00 INT_CLEAR=0x04 IRQ_STATUS=0x08
 * 共享缓存 (§4.6, 1MB, PAGE=256B, shm 用 byte offset = page*PAGE):
 *   覆盖区 0x000~0xBEF (3056 PAGE) | 指令区 0xBF0~0xBFF (16 PAGE, 4KB) | 累加区 0xC00~0xFFF
 */
#define REG_BASE        0x20000000ULL
#define DOORBELL_REG    (REG_BASE + 0x00)   /* §2.3 写: 指令区起始 byte addr, 唤醒取指 */
#define INT_CLEAR_REG   (REG_BASE + 0x04)   /* §2.3 写: 清中断 */
#define IRQ_STATUS_REG  (REG_BASE + 0x08)   /* §2.3 读: 0=idle 1=busy 2=done 3=error */
#define OVERWRITE_BASE  0x000               /* §4.6 覆盖区 (Preload 暂存 / Forward int8 输入) */
#define INSTR_BASE      0xBF0               /* §4.6 指令区 (16 PAGE, 4KB) */
#define A_PAGE_BASE     0x010               /* Forward 输入区 (覆盖区, 与 place.py 一致) */
#define PSUM_PAGE_BASE  0xC00               /* 部分和累加区 (累加区, 与 place.py 一致) */
#define PAGE            256
#define TILE            64
#define IRQ_DONE        2

/* ===== MMIO 抽象层 (回调转发 hw_simulator) =====
 * shm: byte offset 索引 hw_simulator 共享缓存 (§4.6 三区)
 * reg: 绝对地址路由寄存器 (§2.3 DOORBELL/INT_CLEAR/IRQ_STATUS)
 */
typedef void (*shm_write_cb_t)(long, const void *, long);
typedef void (*shm_read_cb_t)(long, void *, long);
typedef void (*reg_write_cb_t)(long, long);
typedef int32_t (*reg_read_cb_t)(long);
static shm_write_cb_t g_shm_write_cb = NULL;
static shm_read_cb_t  g_shm_read_cb = NULL;
static reg_write_cb_t g_reg_write_cb = NULL;
static reg_read_cb_t  g_reg_read_cb = NULL;
void register_cim_hw_sim(shm_write_cb_t sw, shm_read_cb_t sr,
                         reg_write_cb_t rw, reg_read_cb_t rr) {
    g_shm_write_cb = sw; g_shm_read_cb = sr;
    g_reg_write_cb = rw; g_reg_read_cb = rr;
}
static inline void shm_write(long off, const void *s, long n) {
    if (g_shm_write_cb) g_shm_write_cb(off, s, n);
}
static inline void shm_read(long off, void *d, long n) {
    if (g_shm_read_cb) g_shm_read_cb(off, d, n);
}
static inline void reg_write(long reg, long val) {
    if (g_reg_write_cb) g_reg_write_cb(reg, val);
}
static inline uint32_t reg_read(long reg) {
    return g_reg_read_cb ? g_reg_read_cb(reg) : 0;
}
#define HW_READY (g_shm_write_cb != NULL)

/* ===== forward.bin idx 索引 (运行时加载, C3 产出, 按 idx 索引) ===== */
static uint8_t *g_fwd_buf = NULL;       /* 整个 forward.bin */
static uint8_t *g_fwd_base = NULL;      /* 段数据起点 */
static uint32_t g_fwd_nidx = 0;
static uint32_t *g_fwd_off = NULL;
static uint32_t *g_fwd_len = NULL;

void cim_load_forward(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("cim_load_forward fopen"); return; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    g_fwd_buf = (uint8_t *)malloc(sz);
    if (fread(g_fwd_buf, 1, sz, f) != (size_t)sz) { perror("cim_load_forward fread"); }
    fclose(f);
    if (memcmp(g_fwd_buf, "CIMF", 4) != 0) {
        fprintf(stderr, "[cim_stub] bad forward magic\n");
        free(g_fwd_buf); g_fwd_buf = NULL; return;
    }
    g_fwd_nidx = *(uint32_t *)(g_fwd_buf + 4);
    g_fwd_off = (uint32_t *)malloc(g_fwd_nidx * 4);
    g_fwd_len = (uint32_t *)malloc(g_fwd_nidx * 4);
    memcpy(g_fwd_off, g_fwd_buf + 8, g_fwd_nidx * 4);
    memcpy(g_fwd_len, g_fwd_buf + 8 + g_fwd_nidx * 4, g_fwd_nidx * 4);
    g_fwd_base = g_fwd_buf + 8 + 2 * g_fwd_nidx * 4;
}

/* ===== Preload init (一次性, 读 preload.bin 自包含, §4.2/4.6) ===== */
void cim_preload_init(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("cim_preload_init fopen"); return; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *buf = (uint8_t *)malloc(sz);
    if (fread(buf, 1, sz, f) != (size_t)sz) { perror("cim_preload_init fread"); }
    fclose(f);
    if (memcmp(buf, "CIMP", 4) != 0) {
        fprintf(stderr, "[cim_stub] bad preload magic\n"); free(buf); return;
    }
    uint32_t n_batch = *(uint32_t *)(buf + 4);
    uint32_t *b_off = (uint32_t *)(buf + 8);
    long body = 8 + (long)n_batch * 4;
    for (uint32_t b = 0; b < n_batch; b++) {
        uint8_t *p = buf + body + b_off[b];
        uint32_t n_tile = *(uint32_t *)p; p += 4;
        /* 写 tile 到覆盖区 (OVERWRITE_BASE + i*4)*PAGE, 每 tile 1024B = 4 PAGE */
        for (uint32_t i = 0; i < n_tile; i++)
            shm_write((long)(OVERWRITE_BASE + i * 4) * PAGE, p + i * 1024, 1024);
        p += (long)n_tile * 1024;
        /* 写 PROG_WGT 指令区 + SYNC_HALT */
        uint32_t instr_size = n_tile * 6 + 6;
        shm_write((long)INSTR_BASE * PAGE, p, instr_size);
        /* 门铃 -> 取指执行 (PROG_WGT 编程 Macro) */
        reg_write((long)DOORBELL_REG, (long)INSTR_BASE * PAGE);
        while (reg_read((long)IRQ_STATUS_REG) != IRQ_DONE) ;   /* poll IRQ */
        reg_write((long)INT_CLEAR_REG, 1);                     /* INT_CLEAR */
    }
    free(buf);
}

/* ===== fallback: CPU 算 matmul (无硬件无仿真时, 兼容旧路径) ===== */
static void unpack_2bit(const uint8_t *packed, int8_t *out, int64_t N, int64_t K4) {
    int64_t K = K4 * 4;
    for (int64_t n = 0; n < N; n++)
        for (int64_t i = 0; i < K4; i++) {
            uint8_t p = packed[n * K4 + i];
            for (int j = 0; j < 4; j++) {
                int code = (p >> (2 * j)) & 3;
                if (code >= 2) code -= 4;
                out[n * K + i * 4 + j] = (int8_t)code;
            }
        }
}

static Memref2D cim_launch_impl(const int8_t *x, const uint8_t *w,
                                int64_t M, int64_t K, int64_t N, int64_t K4) {
    int8_t *w_int = (int8_t *)malloc(N * K);
    unpack_2bit(w, w_int, N, K4);
    int32_t *result = (int32_t *)malloc(M * N * sizeof(int32_t));
    for (int64_t m = 0; m < M; m++)
        for (int64_t n = 0; n < N; n++) {
            int32_t acc = 0;
            const int8_t *xr = x + m * K, *wr = w_int + n * K;
            for (int64_t k = 0; k < K; k++) acc += (int32_t)xr[k] * (int32_t)wr[k];
            result[m * N + n] = acc;
        }
    free(w_int);
    Memref2D r = {result, result, 0, M, N, N, 1};
    return r;
}

/* ===== [A1] 单一 @cim_launch(idx, X, W): MMIO 驱动 Forward, idx 参数查 forward.bin 段 =====
 * 固定硬件驱动: .so 一次编译, 任意规模复用 (idx 由 L3 传常量, 规模变只改 forward.bin + IR)。
 * 单 token 指令流重复 M 次 (方案 C, seq_len 运行时吸收); 大段分块 (P2-6)。
 * A_PAGE/PSUM_PAGE 串行复用 (首 k_blk ACCUM=0 清旧 acc, 固化在指令里)。
 */
Memref2D cim_launch(int64_t idx,
                    void *xa, void *xaa, int64_t xoff, int64_t M, int64_t K,
                    int64_t xs0, int64_t xs1,
                    void *wa, void *waa, int64_t woff, int64_t N, int64_t K4,
                    int64_t ws0, int64_t ws1) {
    (void)xa; (void)xs0; (void)xs1; (void)wa; (void)ws0; (void)ws1;
    const int8_t *x = (const int8_t *)((const char *)xaa + xoff);
    const uint8_t *w = (const uint8_t *)((const char *)waa + woff);
    if (!HW_READY) return cim_launch_impl(x, w, M, K, N, K4);
    int64_t n_tiles = (N + TILE - 1) / TILE;
    int64_t k_tiles = (K + TILE - 1) / TILE;
    int32_t *result = (int32_t *)malloc((size_t)M * (size_t)N * sizeof(int32_t));
    /* [P2-6] 大段分块: 单段 > 681 MATMUL 时拆块 (指令区 4KB=682 条, 留 1 SYNC_HALT)
       每块 ≤ 681 MATMUL + 末尾 SYNC_HALT, M 循环内多块门铃, 跨块 PSUM_PAGE RMW 累加保留
       (doorbell 只清 busy_until/page_busy 时序, 不清 cache 数据; accum 字段固化保证 K维顺序) */
    const int64_t SEG_MAX = 681;
    int64_t n_mm = (int64_t)g_fwd_len[idx] / 6 - 1;  /* MATMUL 数 (减末尾 SYNC_HALT), idx 查 forward.bin */
    int64_t n_blk = (n_mm + SEG_MAX - 1) / SEG_MAX;  /* 块数 (1=不分块) */
    static const uint8_t HALT[6] = {0,0,0,0,0,0xE0};  /* SYNC_HALT word (0x7<<45 小端) */
    for (int64_t m = 0; m < M; m++) {
        /* 写 A_PAGE (按 k_blk 切, 每 64 int8) -- §4.7.3 */
        for (int64_t kb = 0; kb < k_tiles; kb++)
            shm_write((long)(A_PAGE_BASE + kb) * PAGE,
                      x + m * K + kb * TILE, TILE);
        /* 分块门铃: 每块 ≤ 681 MATMUL + 末尾 SYNC_HALT (跨块 PSUM_PAGE 累加) */
        int64_t mm_done = 0;
        for (int64_t b = 0; b < n_blk; b++) {
            int64_t blk_mm = (mm_done + SEG_MAX <= n_mm) ? SEG_MAX : (n_mm - mm_done);
            shm_write((long)INSTR_BASE * PAGE,
                      g_fwd_base + g_fwd_off[idx] + mm_done * 6, blk_mm * 6);
            shm_write((long)INSTR_BASE * PAGE + blk_mm * 6, (const void *)HALT, 6);
            reg_write((long)DOORBELL_REG, (long)INSTR_BASE * PAGE);
            while (reg_read((long)IRQ_STATUS_REG) != IRQ_DONE) ;  /* poll IRQ */
            reg_write((long)INT_CLEAR_REG, 1);
            mm_done += blk_mm;
        }
        /* 读 PSUM_PAGE acc[m] (按 n_blk 拼, 取 valid 行) -- §4.7.5 */
        for (int64_t nb = 0; nb < n_tiles; nb++) {
            int32_t acc_buf[TILE];
            shm_read((long)(PSUM_PAGE_BASE + nb) * PAGE, acc_buf, PAGE);
            int64_t s = nb * TILE, e = s + TILE; if (e > N) e = N;
            memcpy(result + m * N + s, acc_buf, (size_t)(e - s) * sizeof(int32_t));
        }
    }
    Memref2D r = {result, result, 0, M, N, N, 1}; return r;
}
