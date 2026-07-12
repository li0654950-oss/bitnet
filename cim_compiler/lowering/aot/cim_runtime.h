#ifndef CIM_RUNTIME_H
#define CIM_RUNTIME_H
#include <stdint.h>

/* torch-mlir refbackend unranked memref descriptor ABI (对齐 torch_mlir.runtime)
 * ranked  : { void* allocated; void* aligned; int64_t offset;
 *             int64_t shape[rank]; int64_t strides[rank]; }   // 24 + 16*rank 字节
 * unranked: { int64_t rank; void* descriptor; }               // descriptor -> ranked buffer
 *
 * refbackend main (forward_entry) 经 refback-munge-calling-conventions:
 *   参数 = UnrankedMemRefDescriptor** (invoke 用 pointer(pointer(descriptor)))
 *   return = void, 经 refbackend_consume_func_return_<type> 回调交出结果
 *   多输出 = <type>_<type>... (下划线分隔), consume 一次收 N 个 {i64,ptr}* 独立参数
 */

typedef struct {
    int64_t rank;
    void* descriptor;  /* 指向 ranked buffer (动态 malloc, 24+16*rank 字节) */
} UnrankedMemRefDescriptor;

/* C 版 consume (ciface): 收 forward 返回值 (unranked memref)。
 * 调用链: main -> @refbackend_consume_func_return_<types>(rank, descriptor...) [wrapper, .o 内部]
 *       -> _mlir_ciface_refbackend_consume_func_return_<types>(sp...) [宿主提供, 本函数]
 * wrapper 把每个 (rank, descriptor) 打包成 {i64, ptr} struct 存栈, 传 struct 指针 sp。
 * sp -> {i64 rank; void* descriptor}; descriptor -> ranked buffer
 *   {allocated, aligned, offset, shape[rank], strides[rank]}。
 * 符号 = _mlir_ciface_ + name (register_runtime 的 raw_register_runtime 约定)。 */

/* 单输出: logits (全序列 forward)。 */
void _mlir_ciface_refbackend_consume_func_return_mrf32(void* sp);
/* 三输出: (logits, new_k_caches, new_v_caches) (增量 KV forward)。 */
void _mlir_ciface_refbackend_consume_func_return_mrf32_mrf32_mrf32(void* a, void* b, void* c);

/* 全局 logits (consume 写, main 读) */
extern float* g_logits;
extern int64_t g_logits_rank;
extern int64_t g_logits_shape[8];

/* 全局 new_k/new_v caches (多输出 consume 写, main 增量循环读 + 下步作输入) */
extern float* g_new_k;
extern int64_t g_new_k_rank;
extern int64_t g_new_k_shape[8];
extern float* g_new_v;
extern int64_t g_new_v_rank;
extern int64_t g_new_v_shape[8];

/* 构造 unranked memref: data + rank + sizes + strides -> desc
 * desc->descriptor 由本函数 malloc (24+16*rank 字节, ranked buffer), 调用方持有 desc 即可。 */
void make_unranked_memref(UnrankedMemRefDescriptor* desc, void* data,
                          int64_t rank, const int64_t* sizes, const int64_t* strides);

#endif /* CIM_RUNTIME_H */
