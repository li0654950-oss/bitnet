/* L5: @cim_launch CPU 仿真 stub (C, .so, 返回 struct by value)。
 *
 * @cim_launch_<idx> LLVM calling convention (L4 产出):
 *   (X 2D memref, W 2D memref) -> result 2D memref
 *   每个 2D memref 7 参数: (allocated, aligned, offset, size0, size1, stride0, stride1)
 *   X: si8 [M, K]  per-token int8 激活
 *   W: ui8 [N, K/4]  2bit 补码打包三值权重
 *   -> result: si32 [M, N]  累加输出
 *
 * 语义同 cim_op.cim_matmul: unpack W (ternary) + int matmul -> int32。
 * 37 个 cim_launch_<idx> 逻辑相同 (sizes 从 memref descriptor 读), 用 macro 生成。
 *
 * 编译: cc -O2 -shared -fPIC cim_stub.c -o cim_stub.so
 * 用法: ExecutionEngine(mod, shared_libs=["cim_stub.so"])
 */
#include <stdlib.h>
#include <stdint.h>

typedef struct {
    void *allocated;
    void *aligned;
    int64_t offset;
    int64_t size0, size1;
    int64_t stride0, stride1;
} Memref2D;

/* uint8[N, K4] (2bit 补码) -> int8[N, K] {-1,0,1}。同 cim_op.unpack_2bit_aten。 */
static void unpack_2bit(const uint8_t *packed, int8_t *out, int64_t N, int64_t K4) {
    int64_t K = K4 * 4;
    for (int64_t n = 0; n < N; n++) {
        for (int64_t i = 0; i < K4; i++) {
            uint8_t p = packed[n * K4 + i];
            for (int j = 0; j < 4; j++) {
                int code = (p >> (2 * j)) & 3;
                if (code >= 2) code -= 4;
                out[n * K + i * 4 + j] = (int8_t)code;
            }
        }
    }
}

static Memref2D cim_launch_impl(const int8_t *x, const uint8_t *w,
                                int64_t M, int64_t K, int64_t N, int64_t K4) {
    int8_t *w_int = (int8_t *)malloc(N * K);
    unpack_2bit(w, w_int, N, K4);
    int32_t *result = (int32_t *)malloc(M * N * sizeof(int32_t));
    for (int64_t m = 0; m < M; m++) {
        for (int64_t n = 0; n < N; n++) {
            int32_t acc = 0;
            const int8_t *xr = x + m * K;
            const int8_t *wr = w_int + n * K;
            for (int64_t k = 0; k < K; k++) {
                acc += (int32_t)xr[k] * (int32_t)wr[k];
            }
            result[m * N + n] = acc;
        }
    }
    free(w_int);
    Memref2D r = {result, result, 0, M, N, N, 1};
    return r;
}

/* 37 个 @cim_launch_<idx> wrapper (逻辑相同, calling convention 一致)。 */
#define DEF_LAUNCH(IDX)                                                        \
    Memref2D cim_launch_##IDX(                                                 \
        void *xa, void *xaa, int64_t xoff, int64_t M, int64_t K,               \
        int64_t xs0, int64_t xs1,                                              \
        void *wa, void *waa, int64_t woff, int64_t N, int64_t K4,              \
        int64_t ws0, int64_t ws1) {                                            \
        (void)xa; (void)xs0; (void)xs1; (void)wa; (void)ws0; (void)ws1;        \
        const int8_t *x = (const int8_t *)((const char *)xaa + xoff);          \
        const uint8_t *w = (const uint8_t *)((const char *)waa + woff);        \
        return cim_launch_impl(x, w, M, K, N, K4);                            \
    }

DEF_LAUNCH(0)
DEF_LAUNCH(1)
DEF_LAUNCH(2)
DEF_LAUNCH(3)
DEF_LAUNCH(4)
DEF_LAUNCH(5)
DEF_LAUNCH(6)
DEF_LAUNCH(7)
DEF_LAUNCH(8)
DEF_LAUNCH(9)
DEF_LAUNCH(10)
DEF_LAUNCH(11)
DEF_LAUNCH(12)
DEF_LAUNCH(13)
DEF_LAUNCH(14)
DEF_LAUNCH(15)
DEF_LAUNCH(16)
DEF_LAUNCH(17)
DEF_LAUNCH(18)
DEF_LAUNCH(19)
DEF_LAUNCH(20)
DEF_LAUNCH(21)
DEF_LAUNCH(22)
DEF_LAUNCH(23)
DEF_LAUNCH(24)
DEF_LAUNCH(25)
DEF_LAUNCH(26)
DEF_LAUNCH(27)
DEF_LAUNCH(28)
DEF_LAUNCH(29)
DEF_LAUNCH(30)
DEF_LAUNCH(31)
DEF_LAUNCH(32)
DEF_LAUNCH(33)
DEF_LAUNCH(34)
DEF_LAUNCH(35)
DEF_LAUNCH(36)
