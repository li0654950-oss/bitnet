/* C 版 torch-mlir refbackend runtime: consume 回调 + unranked memref 构造。
 *
 * AOT 链接时提供 refbackend_consume_func_return_mrf32 (to_object.py dump .o 要求 symbol
 * resolved, 作为 shared_lib 提供), 替代 JIT 模式的 Python ctypes 回调。
 *
 * memref descriptor ABI 对齐 torch_mlir.runtime.make_nd_memref_descriptor:
 *   ranked: allocated(8) | aligned(8) | offset(8) | shape[rank](8*rank) | strides[rank](8*rank)
 */
#include "cim_runtime.h"
#include <stdlib.h>

float* g_logits = NULL;
int64_t g_logits_rank = 0;
int64_t g_logits_shape[8] = {0, 0, 0, 0, 0, 0, 0, 0};

void _mlir_ciface_refbackend_consume_func_return_mrf32(void* sp) {
    /* sp -> {i64 rank; void* descriptor} (wrapper alloca+store, mod.ll line 184-191) */
    if (!sp) { g_logits = NULL; g_logits_rank = 0; return; }
    int64_t rank = *(int64_t*)sp;
    void* descriptor = *(void**)((char*)sp + 8);   /* ranked buffer ptr */
    g_logits_rank = rank;
    if (!descriptor) { g_logits = NULL; return; }
    char* p = (char*)descriptor;
    /* ranked: allocated(8) | aligned(8) | offset(8) | shape[rank] | strides[rank] */
    g_logits = *(float**)(p + 8);                  /* aligned */
    int64_t* shape = (int64_t*)(p + 24);
    for (int i = 0; i < rank && i < 8; i++) g_logits_shape[i] = shape[i];
}
/* ExecutionEngine dump 编译 emit_c_interface wrapper 时查 _mlir_<name> symbol, 提供 alias。 */
extern void _mlir_refbackend_consume_func_return_mrf32(void* sp)
    __attribute__((alias("_mlir_ciface_refbackend_consume_func_return_mrf32")));

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
