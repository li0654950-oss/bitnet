/* C 版 torch-mlir refbackend runtime: consume 回调 + unranked memref 构造。
 *
 * AOT 链接时提供 refbackend_consume_func_return_* (to_object.py dump .o 要求 symbol
 * resolved, 作为 shared_lib 提供), 替代 JIT 模式的 Python ctypes 回调。
 *
 * 两类 consume (forward 入口经 refback-munge-calling-conventions 改写为 void + consume 回调):
 *   - refbackend_consume_func_return_mrf32                 : 单输出 (全序列 forward, 仅 logits)
 *   - refbackend_consume_func_return_mrf32_mrf32_mrf32     : 三输出 (增量 KV forward, logits+new_k+new_v)
 *
 * memref descriptor ABI 对齐 torch_mlir.runtime.make_nd_memref_descriptor:
 *   ranked: allocated(8) | aligned(8) | offset(8) | shape[rank](8*rank) | strides[rank](8*rank)
 *   unranked consume 参数: {i64 rank; void* descriptor}*  (descriptor -> ranked buffer)
 *   多输出 ABI = N 个独立 {i64,ptr}* 参数 (对齐 JIT CFUNCTYPE(None, P(UMD), ..., P(UMD)),
 *   refbackend.py get_ctype_func; register_runtime 不改参数布局, C 函数直接匹配)。
 */
#include "cim_runtime.h"
#include <stdlib.h>

/* ===== 单输出 (全序列 forward) ===== */
float* g_logits = NULL;
int64_t g_logits_rank = 0;
int64_t g_logits_shape[8] = {0, 0, 0, 0, 0, 0, 0, 0};

/* ===== 多输出 (增量 KV forward): new_k/new_v_caches [n_layer,1,T,n_kv,hd] f32 ===== */
float* g_new_k = NULL; int64_t g_new_k_rank = 0; int64_t g_new_k_shape[8] = {0,0,0,0,0,0,0,0};
float* g_new_v = NULL; int64_t g_new_v_rank = 0; int64_t g_new_v_shape[8] = {0,0,0,0,0,0,0,0};

/* 从 unranked consume 参数 sp ({i64 rank; void* descriptor}*) 提取 aligned data + shape。
 * sp -> {i64 rank; void* descriptor}; descriptor -> ranked buffer
 *   {allocated, aligned, offset, shape[rank], strides[rank]}。 */
static void extract_memref(void* sp, float** data, int64_t* rank, int64_t* shape) {
    if (!sp) { *data = NULL; *rank = 0; return; }
    int64_t r = *(int64_t*)sp;
    void* descriptor = *(void**)((char*)sp + 8);   /* ranked buffer ptr */
    *rank = r;
    if (!descriptor) { *data = NULL; return; }
    char* p = (char*)descriptor;
    *data = *(float**)(p + 8);                     /* aligned */
    int64_t* sh = (int64_t*)(p + 24);
    for (int i = 0; i < r && i < 8; i++) shape[i] = sh[i];
}

/* 单输出: forward 返回 logits (unranked memref f32)。 */
void _mlir_ciface_refbackend_consume_func_return_mrf32(void* sp) {
    extract_memref(sp, &g_logits, &g_logits_rank, g_logits_shape);
}
/* ExecutionEngine dump 编译 emit_c_interface wrapper 时查 _mlir_<name> symbol, 提供 alias。 */
extern void _mlir_refbackend_consume_func_return_mrf32(void* sp)
    __attribute__((alias("_mlir_ciface_refbackend_consume_func_return_mrf32")));

/* 三输出: (logits, new_k_caches, new_v_caches), 每个独立 {i64 rank; void* descriptor}* 参数。
 * 顺序 = _KVCacheModel.forward 返回 (logits, stack(new_ks), stack(new_vs))。
 * cim_main 增量循环: 每步 call_forward 后读 g_logits(argmax) + g_new_k/g_new_v(更新 cache)。 */
void _mlir_ciface_refbackend_consume_func_return_mrf32_mrf32_mrf32(
        void* a, void* b, void* c) {
    extract_memref(a, &g_logits, &g_logits_rank, g_logits_shape);
    extract_memref(b, &g_new_k,  &g_new_k_rank,  g_new_k_shape);
    extract_memref(c, &g_new_v,  &g_new_v_rank,  g_new_v_shape);
}
extern void _mlir_refbackend_consume_func_return_mrf32_mrf32_mrf32(void* a, void* b, void* c)
    __attribute__((alias("_mlir_ciface_refbackend_consume_func_return_mrf32_mrf32_mrf32")));

void make_unranked_memref(UnrankedMemRefDescriptor* desc, void* data,
                          int64_t rank, const int64_t* sizes, const int64_t* strides) {
    size_t sz = 24 + (size_t)16 * (size_t)rank;   /* 3*8 + rank*8 (shape) + rank*8 (strides) */
    char* buf = (char*)malloc(sz);
    *(void**)(buf + 0)  = data;                    /* allocated */
    *(void**)(buf + 8)  = data;                    /* aligned */
    *(int64_t*)(buf + 16) = 0;                     /* offset */
    int64_t* shape = (int64_t*)(buf + 24);
    int64_t* strd  = (int64_t*)(buf + 24 + 8 * (size_t)rank);
    for (int i = 0; i < rank; i++) { shape[i] = sizes[i]; strd[i] = strides[i]; }
    desc->rank = rank;
    desc->descriptor = buf;
}
