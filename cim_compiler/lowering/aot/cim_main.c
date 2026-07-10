/* cim_sim AOT 主入口: 完整文本生成 (IPC 仿真器)。
 *
 * 构造 50 个 unranked memref (49 buffer 复用 + idx 每步重建), 调 _mlir_ciface_main,
 * greedy 自回归生成 (argmax logits[0, last, :] + block_size 截断, 对齐 run_sim_text.py)。
 * w_packed 空壳 (cim_launch 不读 W, 用 idx 查 forward.bin + Macro 驻留权重)。
 *
 * 启动: 先 cim_sim_server.py, 再 ./cim_sim --prompt "ROMEO:" --n 60
 */
#include "cim_runtime.h"
#include "cim_ipc.h"
#include "tokenizer_data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* _mlir_ciface_main: refbackend C 入口, 50 个 UnrankedMemRefDescriptor* 参数。
 * 顺序 = input_specs (跳过 PARAMETER):
 *   [0..47] 6 层 × 8 (inv_freq, causal_mask, q/k/v/o_proj.w_packed, fc1.w_packed, fc2.w_packed)
 *   [48]    lm_head.w_packed
 *   [49]    idx (USER_INPUT, int64 [1, seq]) */
extern void _mlir_ciface_main(
    UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*,
    UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*,
    UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*,
    UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*,
    UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*,
    UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*,
    UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*,
    UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*,
    UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*,
    UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*,
    UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*,
    UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*,
    UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*,
    UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*,
    UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*,
    UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*,
    UnrankedMemRefDescriptor*, UnrankedMemRefDescriptor*);

extern void cim_load_forward(const char*);
extern void cim_preload_init(const char*);

#define D_MODEL 512
#define D_HEAD  64
#define FFN_DIM 1664
#define BLOCK   256
#define N_LAYER 6

/* 每层 6 个 w_packed 的 (N, K/4): q_proj, k_proj, v_proj, o_proj, fc1, fc2 */
static const int WP_N[6]  = {512, 256, 256, 512, 1664, 512};
static const int WP_K4[6] = {128, 128, 128, 128, 128,  416};

static void fill_inv_freq(float* buf) {            /* (32,) f32: 1/(10000^(2i/d_head)) */
    for (int i = 0; i < D_HEAD / 2; i++)
        buf[i] = 1.0f / powf(10000.0f, (float)(2 * i) / (float)D_HEAD);
}
static void fill_causal_mask(uint8_t* buf) {       /* (1,1,BLOCK,BLOCK) bool: True=mask 未来 */
    for (int i = 0; i < BLOCK; i++)
        for (int j = 0; j < BLOCK; j++)
            buf[i * BLOCK + j] = (j > i) ? 1 : 0;
}

/* 构造 49 buffer memref (desc[0..48]), 一次构造 generate 循环复用 */
static void build_buffers(UnrankedMemRefDescriptor* desc) {
    for (int L = 0; L < N_LAYER; L++) {
        int b = L * 8;
        float* inv = (float*)malloc(32 * sizeof(float)); fill_inv_freq(inv);
        { int64_t sz[] = {32}, st[] = {1}; make_unranked_memref(&desc[b + 0], inv, 1, sz, st); }
        uint8_t* cm = (uint8_t*)malloc((size_t)BLOCK * BLOCK); fill_causal_mask(cm);
        { int64_t sz[] = {1, 1, BLOCK, BLOCK}, st[] = {BLOCK*BLOCK, BLOCK*BLOCK, BLOCK, 1};
          make_unranked_memref(&desc[b + 1], cm, 4, sz, st); }
        for (int w = 0; w < 6; w++) {              /* w_packed 空壳 (零, shape 对; cim_launch 不读) */
            uint8_t* wp = (uint8_t*)calloc((size_t)WP_N[w] * WP_K4[w], 1);
            int64_t sz[] = {WP_N[w], WP_K4[w]}, st[] = {WP_K4[w], 1};
            make_unranked_memref(&desc[b + 2 + w], wp, 2, sz, st);
        }
    }
    uint8_t* lmh = (uint8_t*)calloc((size_t)VOCAB * 128, 1);   /* lm_head.w_packed 空壳 */
    { int64_t sz[] = {VOCAB, 128}, st[] = {128, 1}; make_unranked_memref(&desc[48], lmh, 2, sz, st); }
}

/* 构造 idx memref (desc[49]), 每步重建 (seq 变化) */
static void build_idx(UnrankedMemRefDescriptor* desc, int64_t* idx_data, int seq) {
    int64_t sz[] = {1, seq}, st[] = {seq, 1};
    make_unranked_memref(&desc[49], idx_data, 2, sz, st);
}

static int encode_prompt(const char* text, int* ids, int max) {
    int n = 0;
    for (const char* p = text; *p && n < max; p++) {
        int id = stoi[(unsigned char)*p];
        if (id < 0) { fprintf(stderr, "[err] vocab 外字符: '%c' (0x%02x)\n", *p, (unsigned char)*p); return -1; }
        ids[n++] = id;
    }
    return n;
}

int main(int argc, char** argv) {
    const char* prompt = "ROMEO:";
    const char* socket = "/tmp/cim_sim.sock";
    const char* fwd = "cim_compiler/cimres/checkpoints/forward.bin";
    const char* pre = "cim_compiler/cimres/checkpoints/preload.bin";
    int n = 60, block = BLOCK;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--socket") && i + 1 < argc) socket = argv[++i];
        else if (!strcmp(argv[i], "--forward") && i + 1 < argc) fwd = argv[++i];
        else if (!strcmp(argv[i], "--preload") && i + 1 < argc) pre = argv[++i];
        else if (!strcmp(argv[i], "--block") && i + 1 < argc) block = atoi(argv[++i]);
    }
    printf("[cim_sim] AOT 文本生成 (IPC 仿真器), prompt=%s, n=%d\n", prompt, n);
    if (cim_ipc_init(socket) < 0) {
        fprintf(stderr, "[cim_sim] IPC 连接失败, 先启动 cim_sim_server.py\n");
        return 1;
    }
    cim_load_forward(fwd);
    fprintf(stderr, "[cim_sim] Preload (经 IPC 驱动仿真器)...\n");
    cim_preload_init(pre);
    fprintf(stderr, "[cim_sim] Preload 完成\n");

    int idx[BLOCK + 128];
    int n_idx = encode_prompt(prompt, idx, BLOCK);
    if (n_idx < 0) return 1;
    if (n_idx == 0) { idx[0] = 0; n_idx = 1; }     /* BOS */
    fprintf(stderr, "[prompt] %d token:", n_idx);
    for (int i = 0; i < n_idx; i++) fprintf(stderr, " %d", idx[i]);
    fprintf(stderr, "\n");

    UnrankedMemRefDescriptor desc[50];
    build_buffers(desc);                            /* 49 buffer 一次构造 */

    int64_t* idx_buf = (int64_t*)malloc((size_t)BLOCK * sizeof(int64_t));
    for (int step = 0; step < n; step++) {
        int seq = n_idx < block ? n_idx : block;    /* block_size 截断 (取最后 seq 个) */
        for (int i = 0; i < seq; i++) idx_buf[i] = (int64_t)idx[n_idx - seq + i];
        build_idx(desc, idx_buf, seq);              /* 重建 desc[49] */
        _mlir_ciface_main(
            &desc[0], &desc[1], &desc[2], &desc[3], &desc[4], &desc[5], &desc[6], &desc[7], &desc[8], &desc[9],
            &desc[10], &desc[11], &desc[12], &desc[13], &desc[14], &desc[15], &desc[16], &desc[17], &desc[18], &desc[19],
            &desc[20], &desc[21], &desc[22], &desc[23], &desc[24], &desc[25], &desc[26], &desc[27], &desc[28], &desc[29],
            &desc[30], &desc[31], &desc[32], &desc[33], &desc[34], &desc[35], &desc[36], &desc[37], &desc[38], &desc[39],
            &desc[40], &desc[41], &desc[42], &desc[43], &desc[44], &desc[45], &desc[46], &desc[47], &desc[48], &desc[49]);
        float* last = g_logits + (int64_t)(seq - 1) * VOCAB;   /* logits[0, last_token, :] */
        int nxt = 0;
        for (int v = 1; v < VOCAB; v++) if (last[v] > last[nxt]) nxt = v;
        idx[n_idx++] = nxt;
        if ((step + 1) % 10 == 0) fprintf(stderr, "  ...已生成 %d/%d token\n", step + 1, n);
    }

    /* decode + 输出 */
    char out[BLOCK + 128];
    int o = 0;
    for (int i = 0; i < n_idx && o < (int)sizeof(out) - 1; i++) out[o++] = itos[idx[i]];
    out[o] = 0;
    printf("prompt  : %s\n", prompt);
    printf("输出    : %s\n", out);
    return 0;
}
