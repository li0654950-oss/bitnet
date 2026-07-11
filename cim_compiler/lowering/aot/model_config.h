#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H
#include <stdint.h>

/* model_config.bin: 运行时模型配置 (gen_config.py 从 .pt2 提取, cim_main.c 读)。
 *
 * 目的: 让 cim_main.c 成为固定通用宿主, 任意模型规模复用 -- 换模型只换 .pt2/
 * forward.bin/preload.bin + 重编译 forward.o, 不改 cim_main.c 一行。forward 入口
 * 参数个数 = n_buffer+1 随 n_layer 变, 由 libffi 运行时变参调用解决 (cim_main.c)。
 *
 * 磁盘格式 (小端, gen_config.py struct.pack 对应):
 *   "CIMC" magic(4) | n_buffer(u32) | n_layer(u32) | vocab(u32) | block_size(u32)
 *   buffer 描述表 (n_buffer 项, 每项紧凑, 不对齐填充):
 *     kind(u8) | rank(u8) | shape[rank](i64*rank)
 *   tokenizer (char-level):
 *     itos[vocab](char*vocab) | stoi[128](i32*128, ascii->id, -1=未知)
 *
 * kind: 0=inv_freq(f32[d_head/2]) 1=causal_mask(u8[block,block])
 *       2=w_packed(u8[N,K4] 空壳) 3=lm_head.w_packed(u8[V,K4] 空壳)
 * buffer 顺序 = .pt2 input_specs 顺序 (PARAMETER 跳过, idx=USER_INPUT 最后, 不入表)。
 *   即每层 8 个 (inv_freq, causal_mask, q/k/v/o_proj.w_packed, fc1.w_packed, fc2.w_packed)
 *   + lm_head.w_packed; n_buffer = n_layer*8 + 1。
 */

#define MC_MAGIC 0x434D4943u   /* "CIMC" 小端 (字节 43 49 4D 43) */

#define MC_KIND_INVFREQ         0
#define MC_KIND_CAUSAL_MASK     1
#define MC_KIND_W_PACKED        2
#define MC_KIND_LMHEAD_W_PACKED 3

#define MC_MAX_RANK   4
#define MC_MAX_BUFFER 512   /* n_layer*8+1 <= 512 -> n_layer <= 63 */

typedef struct {
    uint8_t kind;
    uint8_t rank;
    int64_t shape[MC_MAX_RANK];
} BufferDesc;

typedef struct {
    uint32_t magic;
    uint32_t n_buffer;    /* 不含 idx */
    uint32_t n_layer;
    uint32_t vocab;
    uint32_t block_size;
    BufferDesc* buffers;  /* [n_buffer], malloc */
    char*     itos;       /* [vocab], malloc */
    int32_t   stoi[128];  /* ascii -> id, -1 未知 */
} ModelConfig;

#endif /* MODEL_CONFIG_H */
