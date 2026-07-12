/* cim_sim AOT 主入口: 固定通用宿主, 完整文本生成 (IPC 仿真器)。
 *
 * 任意模型规模复用: 超参/buffer 形状/vocab/tokenizer 运行时从 model_config.bin 读
 * (gen_config.py 从 .pt2 提取)。换模型只换 .pt2/forward.bin/preload.bin + 重编译
 * forward.o, cim_main.c 不改一行。forward 入口 _mlir_ciface_main 参数个数 = n_buffer+1
 * 随 n_layer 变, 由 libffi 运行时变参调用 (ffi_call) 解决 -- 这是 cim_main.c 能固定的
 * 关键: "固定的程序处理变化的内容"。
 *
 * 构造 n_buffer+1 个 unranked memref (n_buffer buffer 复用 + idx 每步重建), ffi_call 调
 * _mlir_ciface_main, greedy 自回归生成 (argmax logits[0,last,:] + block_size 截断)。
 * w_packed 空壳 (cim_launch 不读 W, 用 idx 查 forward.bin + Macro 驻留权重)。
 *
 * 启动: 先 cim_sim_server.py, 再 ./cim_sim --prompt "ROMEO:" --n 60
 */
#include "cim_runtime.h"
#include "cim_ipc.h"
#include "model_config.h"
#include <ffi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern void _mlir_ciface_main(void);   /* 无原型, libffi 运行时变参调用 */

extern void cim_load_forward(const char*);
extern void cim_preload_init(const char*);

/* ===== model_config.bin 加载 (格式见 model_config.h) ===== */
static int load_config(ModelConfig* cfg, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror("load_config fopen"); return -1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t* buf = (uint8_t*)malloc(sz);
    if (fread(buf, 1, sz, f) != (size_t)sz) { perror("load_config fread"); free(buf); fclose(f); return -1; }
    fclose(f);
    const uint8_t* p = buf;
    cfg->magic = *(uint32_t*)p; p += 4;
    if (cfg->magic != MC_MAGIC) { fprintf(stderr, "[cim_sim] bad config magic\n"); free(buf); return -1; }
    cfg->n_buffer   = *(uint32_t*)p; p += 4;
    cfg->n_layer    = *(uint32_t*)p; p += 4;
    cfg->vocab      = *(uint32_t*)p; p += 4;
    cfg->block_size = *(uint32_t*)p; p += 4;
    cfg->n_kv       = *(uint32_t*)p; p += 4;
    cfg->head_dim   = *(uint32_t*)p; p += 4;
    cfg->buffers = (BufferDesc*)malloc(cfg->n_buffer * sizeof(BufferDesc));
    for (uint32_t i = 0; i < cfg->n_buffer; i++) {
        cfg->buffers[i].kind = *p++;
        cfg->buffers[i].rank = *p++;
        for (int j = 0; j < cfg->buffers[i].rank; j++) { cfg->buffers[i].shape[j] = *(int64_t*)p; p += 8; }
    }
    cfg->itos = (char*)malloc(cfg->vocab);
    memcpy(cfg->itos, p, cfg->vocab); p += cfg->vocab;
    for (int j = 0; j < 128; j++) { cfg->stoi[j] = *(int32_t*)p; p += 4; }
    int hd2 = (int)cfg->head_dim / 2;
    cfg->inv_freq_data = (float*)malloc((size_t)hd2 * sizeof(float));
    memcpy(cfg->inv_freq_data, p, (size_t)hd2 * sizeof(float));
    p += (size_t)hd2 * sizeof(float);
    free(buf);
    return 0;
}

/* row-major strides from shape */
static void row_major_strides(const int64_t* shape, int rank, int64_t* strides) {
    if (rank == 0) return;
    strides[rank - 1] = 1;
    for (int j = rank - 2; j >= 0; j--) strides[j] = strides[j + 1] * shape[j + 1];
}

/* 构造 n_buffer buffer memref, 一次构造 generate 循环复用 (按 config 描述表驱动) */
static void build_buffers(UnrankedMemRefDescriptor* desc, const ModelConfig* cfg) {
    for (uint32_t i = 0; i < cfg->n_buffer; i++) {
        BufferDesc* bd = &cfg->buffers[i];
        int64_t strides[MC_MAX_RANK];
        row_major_strides(bd->shape, bd->rank, strides);
        void* data = NULL;
        if (bd->kind == MC_KIND_INVFREQ) {            /* f32[d_head/2]: 从 config 读 (和 model 一致, 非 powf 运行时算) */
            int nh = (int)bd->shape[0];
            float* d = (float*)malloc(nh * sizeof(float));
            memcpy(d, cfg->inv_freq_data, nh * sizeof(float));
            data = d;
        } else if (bd->kind == MC_KIND_CAUSAL_MASK) { /* u8[B,B]: True=mask 未来 */
            int B = (int)cfg->block_size;
            uint8_t* d = (uint8_t*)malloc((size_t)B * B);
            for (int r = 0; r < B; r++) for (int c = 0; c < B; c++) d[r * B + c] = (c > r) ? 1 : 0;
            data = d;
        } else {                                       /* w_packed/lmhead 空壳 (零, cim_launch 不读) */
            size_t sz = 1;
            for (int j = 0; j < bd->rank; j++) sz *= (size_t)bd->shape[j];
            data = calloc(sz, 1);
        }
        make_unranked_memref(&desc[i], data, bd->rank, bd->shape, strides);
    }
}

/* 构造 idx memref (desc[idx_pos]), 每步重建 (seq 变化) */
static void build_idx(UnrankedMemRefDescriptor* desc, int idx_pos, int64_t* idx_data, int seq) {
    int64_t sz[] = {1, seq}, st[] = {seq, 1};
    make_unranked_memref(&desc[idx_pos], idx_data, 2, sz, st);
}

/* libffi 运行时变参调用 _mlir_ciface_main (参数个数 = n, 随 n_layer 变)。
 * arg_vals[i] = &desc[i] (UnrankedMemRefDescriptor*), 全指针 -> ffi_type_pointer。 */
static ffi_cif g_cif;
static ffi_type* g_arg_types[MC_MAX_BUFFER + 1];
static void* g_arg_vals[MC_MAX_BUFFER + 1];
static void* g_desc_ptrs[MC_MAX_BUFFER + 1];   /* 存 &desc[i] (指针值), arg_vals 指向它 */
static int g_n_args = 0;
static int g_cif_ready = 0;
static float g_temperature = 0.0f;   /* <=0: greedy argmax; >0: 采样 (softmax+top_k+multinomial) */
static int   g_top_k = 40;
static void call_forward(UnrankedMemRefDescriptor* desc, int n) {
    if (!g_cif_ready) {
        for (int i = 0; i < n; i++) g_arg_types[i] = &ffi_type_pointer;
        if (ffi_prep_cif(&g_cif, FFI_DEFAULT_ABI, n, &ffi_type_void, g_arg_types) != FFI_OK) {
            fprintf(stderr, "[cim_sim] ffi_prep_cif 失败\n"); exit(1);
        }
        g_n_args = n; g_cif_ready = 1;
    }
    /* ffi_type_pointer: arg_vals[i] 指向存放指针值的 8 字节, 即 &g_desc_ptrs[i];
     * ffi 读 *(void**)arg_vals[i] = g_desc_ptrs[i] = &desc[i] (UnrankedMemRefDescriptor*) */
    for (int i = 0; i < g_n_args; i++) {
        g_desc_ptrs[i] = &desc[i];
        g_arg_vals[i] = &g_desc_ptrs[i];
    }
    ffi_call(&g_cif, FFI_FN(_mlir_ciface_main), NULL, g_arg_vals);
}

/* ===== 增量 KV cache 模式 (--kv): prefill 逐 token + decode 单 token (M=1) ===== */
/* O(n²)->O(n): 全序列每步 M=T (ceil(T/64) M-tile), 增量 decode 每步 M=1 (1 M-tile)。
 * forward 返回 (logits, new_k, new_v) 经多输出 consume 写 g_logits/g_new_k/g_new_v。
 * cache 状态 (k_data/v_data/T) 每步从 g_new_k/g_new_v 更新, 下步作 k_caches/v_caches 输入。 */

/* 构造 cache memref [n_layer,1,T,n_kv,hd] (T 动态, data 复用 g_new_k/g_new_v 输出)。 */
static void build_cache_memref(UnrankedMemRefDescriptor* desc, float* data,
                               int n_layer, int64_t T, int n_kv, int hd) {
    int64_t shape[5] = {n_layer, 1, T, n_kv, hd};
    int64_t strides[5];
    row_major_strides(shape, 5, strides);
    make_unranked_memref(desc, data, 5, shape, strides);
}

/* 构造 cos/sin memref [1,1,1,hd2] (row-major, data=buf[hd2])。 */
static void build_vec_memref(UnrankedMemRefDescriptor* desc, float* buf, int hd2) {
    int64_t shape[4] = {1, 1, 1, hd2};
    int64_t strides[4];
    row_major_strides(shape, 4, strides);
    make_unranked_memref(desc, buf, 4, shape, strides);
}

/* 算 cos/sin: pos=T(cache_len), freq[d]=T*inv_freq[d] (对齐 JIT einsum("t,d->td"))。 */
static void compute_cos_sin(float* cos_buf, float* sin_buf, int64_t T,
                            const float* inv_freq, int hd2) {
    for (int d = 0; d < hd2; d++) {
        float freq = (float)T * inv_freq[d];
        cos_buf[d] = cosf(freq);
        sin_buf[d] = sinf(freq);
    }
}

/* 从 desc[inv_freq_pos] 提取 inv_freq data (第一个 INVFREQ buffer, 各层相同)。 */
static float* get_inv_freq(UnrankedMemRefDescriptor* desc, const ModelConfig* cfg) {
    for (uint32_t i = 0; i < cfg->n_buffer; i++) {
        if (cfg->buffers[i].kind == MC_KIND_INVFREQ) {
            return *(float**)((char*)desc[i].descriptor + 8);  /* aligned */
        }
    }
    return NULL;
}

/* token 选择: temperature<=0 greedy argmax; 否则 softmax(temp)+top_k 截断+multinomial 采样。
 * vocab 小 (65), O(vocab*top_k) 每步开销可忽略; softmax 减最大值数值稳定, 累积用 double。 */
static int sample_token(const float* logits, int vocab, float temp, int top_k) {
    if (temp <= 0.0f) {                              /* greedy */
        int nxt = 0;
        for (int v = 1; v < vocab; v++) if (logits[v] > logits[nxt]) nxt = v;
        return nxt;
    }
    float* l = (float*)malloc((size_t)vocab * sizeof(float));
    memcpy(l, logits, (size_t)vocab * sizeof(float));
    if (top_k > 0 && top_k < vocab) {                /* top_k: 保留 top_k 大, 其余 -INF */
        char* keep = (char*)calloc((size_t)vocab, 1);
        for (int r = 0; r < top_k; r++) {
            int mi = 0; float mv = -1e30f;
            for (int v = 0; v < vocab; v++) if (!keep[v] && l[v] > mv) { mv = l[v]; mi = v; }
            keep[mi] = 1;
        }
        for (int v = 0; v < vocab; v++) if (!keep[v]) l[v] = -1e30f;
        free(keep);
    }
    float mx = -1e30f;                               /* softmax(l/temp) 数值稳定 */
    for (int v = 0; v < vocab; v++) if (l[v] > mx) mx = l[v];
    float* p = (float*)malloc((size_t)vocab * sizeof(float));
    double sum = 0.0;
    for (int v = 0; v < vocab; v++) { p[v] = expf((l[v] - mx) / temp); sum += p[v]; }
    float r = (float)rand() / (float)RAND_MAX;       /* [0,1) */
    double acc = 0.0;
    int nxt = vocab - 1;                             /* 兜底: 浮点误差 r>所有 acc */
    for (int v = 0; v < vocab; v++) { acc += p[v] / sum; if (r <= acc) { nxt = v; break; } }
    free(l); free(p);
    return nxt;
}

/* 增量 KV 生成: prefill prompt 逐 token (建 cache) + decode n token (M=1)。
 * args 顺序 = BUFFER[0..n_buffer-1] + USER_INPUT(idx, k_caches, v_caches, cos, sin)。 */
static int run_kv(const ModelConfig* cfg, int* idx, int n_idx, int n,
                  UnrankedMemRefDescriptor* desc, int n_args) {
    int n_layer = (int)cfg->n_layer;
    int n_kv = (int)cfg->n_kv;
    int hd = (int)cfg->head_dim;
    int hd2 = hd / 2;
    int vocab = (int)cfg->vocab;
    int idx_pos = (int)cfg->n_buffer;
    int k_pos = idx_pos + 1, v_pos = idx_pos + 2;
    int cos_pos = idx_pos + 3, sin_pos = idx_pos + 4;

    float* inv_freq = get_inv_freq(desc, cfg);
    if (!inv_freq) { fprintf(stderr, "[cim_sim] inv_freq buffer 未找到\n"); return -1; }

    float* cos_buf = (float*)malloc((size_t)hd2 * sizeof(float));
    float* sin_buf = (float*)malloc((size_t)hd2 * sizeof(float));
    int64_t* idx_buf = (int64_t*)malloc(sizeof(int64_t));

    /* cache 状态: 首步 T=0 (空 cache, data=NULL) */
    float* k_data = NULL;
    float* v_data = NULL;
    int64_t T = 0;

    /* prefill prompt 逐 token (建 cache, 每步 T 增长) */
    for (int i = 0; i < n_idx; i++) {
        idx_buf[0] = (int64_t)idx[i];
        compute_cos_sin(cos_buf, sin_buf, T, inv_freq, hd2);
        build_idx(desc, idx_pos, idx_buf, 1);
        build_cache_memref(&desc[k_pos], k_data, n_layer, T, n_kv, hd);
        build_cache_memref(&desc[v_pos], v_data, n_layer, T, n_kv, hd);
        build_vec_memref(&desc[cos_pos], cos_buf, hd2);
        build_vec_memref(&desc[sin_pos], sin_buf, hd2);
        call_forward(desc, n_args);                 /* consume 写 g_logits/g_new_k/g_new_v */
        k_data = g_new_k; v_data = g_new_v;
        T = g_new_k_shape[2];                       /* [n_layer,1,T,n_kv,hd] -> T */
    }

    /* decode n token (M=1: argmax 上步 logits -> next, step(next) 用 cache) */
    for (int s = 0; s < n; s++) {
        int nxt = sample_token(g_logits, vocab, g_temperature, g_top_k);
        idx[n_idx++] = nxt;
        idx_buf[0] = (int64_t)nxt;
        compute_cos_sin(cos_buf, sin_buf, T, inv_freq, hd2);
        build_idx(desc, idx_pos, idx_buf, 1);
        build_cache_memref(&desc[k_pos], k_data, n_layer, T, n_kv, hd);
        build_cache_memref(&desc[v_pos], v_data, n_layer, T, n_kv, hd);
        build_vec_memref(&desc[cos_pos], cos_buf, hd2);
        build_vec_memref(&desc[sin_pos], sin_buf, hd2);
        call_forward(desc, n_args);
        k_data = g_new_k; v_data = g_new_v;
        T = g_new_k_shape[2];
        if ((s + 1) % 10 == 0) fprintf(stderr, "  ...已生成 %d/%d token\n", s + 1, n);
    }

    free(cos_buf); free(sin_buf); free(idx_buf);
    return n_idx;
}

static int encode_prompt(const ModelConfig* cfg, const char* text, int* ids, int max) {
    int n = 0;
    for (const char* p = text; *p && n < max; p++) {
        int id = cfg->stoi[(unsigned char)*p];
        if (id < 0) { fprintf(stderr, "[err] vocab 外字符: '%c' (0x%02x)\n", *p, (unsigned char)*p); return -1; }
        ids[n++] = id;
    }
    return n;
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);   /* 无缓冲: 避免大输出 printf 卡 (重定向时全缓冲, kill 后才 flush) */
    const char* prompt = "ROMEO:";
    const char* socket = "/tmp/cim_sim.sock";
    const char* fwd = NULL;
    const char* pre = NULL;
    const char* cfg_path = NULL;
    int n = 60;
    int kv = 0;
    float temperature = 0.0f;
    int top_k = 40;
    unsigned int seed = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--socket") && i + 1 < argc) socket = argv[++i];
        else if (!strcmp(argv[i], "--forward") && i + 1 < argc) fwd = argv[++i];
        else if (!strcmp(argv[i], "--preload") && i + 1 < argc) pre = argv[++i];
        else if (!strcmp(argv[i], "--config") && i + 1 < argc) cfg_path = argv[++i];
        else if (!strcmp(argv[i], "--kv")) kv = 1;
        else if (!strcmp(argv[i], "--temperature") && i + 1 < argc) temperature = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--top_k") && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (unsigned)atoi(argv[++i]);
    }
    g_temperature = temperature;
    g_top_k = top_k;
    srand(seed + 1);   /* +1: glibc srand(0)==srand(1), 偏移使 seed 0/1 不同 */
    if (!fwd)      fwd      = kv ? "cim_compiler/cimres/checkpoints/forward_kv.bin"
                                 : "cim_compiler/cimres/checkpoints/forward.bin";
    if (!pre)      pre      = kv ? "cim_compiler/cimres/checkpoints/preload_kv.bin"
                                 : "cim_compiler/cimres/checkpoints/preload.bin";
    if (!cfg_path) cfg_path = kv ? "cim_compiler/cimres/checkpoints/model_config_kv.bin"
                                 : "cim_compiler/cimres/checkpoints/model_config.bin";

    ModelConfig cfg;
    if (load_config(&cfg, cfg_path) < 0) return 1;
    int block = (int)cfg.block_size;
    int vocab = (int)cfg.vocab;
    int n_args = kv ? ((int)cfg.n_buffer + 5) : ((int)cfg.n_buffer + 1);
    printf("[cim_sim] AOT 文本生成 (IPC 仿真器 + libffi), prompt=%s, n=%d%s\n",
           prompt, n, kv ? " [KV cache 增量]" : "");
    printf("[cim_sim] model: n_layer=%u n_buffer=%u vocab=%u block=%u n_kv=%u head=%u (n_args=%d)\n",
           cfg.n_layer, cfg.n_buffer, cfg.vocab, cfg.block_size, cfg.n_kv, cfg.head_dim, n_args);

    if (cim_ipc_init(socket) < 0) {
        fprintf(stderr, "[cim_sim] IPC 连接失败, 先启动 cim_sim_server.py\n");
        return 1;
    }
    cim_load_forward(fwd);
    fprintf(stderr, "[cim_sim] Preload (经 IPC 驱动仿真器)...\n");
    cim_preload_init(pre);
    fprintf(stderr, "[cim_sim] Preload 完成\n");

    int* idx = (int*)malloc((size_t)(block + 128) * sizeof(int));
    int n_idx = encode_prompt(&cfg, prompt, idx, block);
    if (n_idx < 0) return 1;
    if (n_idx == 0) { idx[0] = 0; n_idx = 1; }     /* BOS */
    fprintf(stderr, "[prompt] %d token:", n_idx);
    for (int i = 0; i < n_idx; i++) fprintf(stderr, " %d", idx[i]);
    fprintf(stderr, "\n");

    UnrankedMemRefDescriptor* desc = (UnrankedMemRefDescriptor*)calloc(n_args, sizeof(UnrankedMemRefDescriptor));
    build_buffers(desc, &cfg);                          /* n_buffer buffer 一次构造 */

    int64_t* idx_buf = (int64_t*)malloc((size_t)block * sizeof(int64_t));
    if (kv) {
        n_idx = run_kv(&cfg, idx, n_idx, n, desc, n_args);
        if (n_idx < 0) return 1;
    } else {
        int idx_pos = (int)cfg.n_buffer;                /* idx 在最后 */
        for (int step = 0; step < n; step++) {
            int seq = n_idx < block ? n_idx : block;    /* block_size 截断 (取最后 seq 个) */
            for (int i = 0; i < seq; i++) idx_buf[i] = (int64_t)idx[n_idx - seq + i];
            build_idx(desc, idx_pos, idx_buf, seq);     /* 重建 desc[idx_pos] */
            call_forward(desc, n_args);                 /* libffi 变参调 _mlir_ciface_main */
            float* last = g_logits + (int64_t)(seq - 1) * vocab;   /* logits[0, last_token, :] */
            int nxt = sample_token(last, vocab, g_temperature, g_top_k);
            idx[n_idx++] = nxt;
            if ((step + 1) % 10 == 0) fprintf(stderr, "  ...已生成 %d/%d token\n", step + 1, n);
        }
    }

    /* decode + 输出 */
    char* out = (char*)malloc((size_t)(block + 128));
    int o = 0;
    for (int i = 0; i < n_idx && o < block + 127; i++) out[o++] = cfg.itos[idx[i]];
    out[o] = 0;
    printf("prompt  : %s\n", prompt);
    printf("输出    : %s\n", out);
    return 0;
}
