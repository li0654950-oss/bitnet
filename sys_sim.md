把 cpu 侧（LLVM JIT）和 cim 侧（hw_simulator）串起来，看运行时一次前向推理（invoker.invoke("main", *inputs)）实际怎么走。

总览

[预备·一次性]  preload_phase (PROG_WGT 预载 3664 Macro) + register_cim_sim_callback + build_inputs(50 个 memref)
      │
[invoke main]  ctypes 50 input -> ee.invoke("main")  ── main = BitNet forward 拓扑 (LLVM JIT 执行)
      │
[main 执行]    embed [CPU] ──► 6×layer ──► ln_f [CPU] ──► lm_head [CIM] ──► return logits
                  └ 每层: q/k/v_proj[CIM闭环] → attention[CPU] → o_proj[CIM闭环] → 残差[CPU]
                          fc1[CIM闭环] → act[CPU] → fc2[CIM闭环] → 残差[CPU]
      │
[return]       refbackend_consume_func_return callback → logits numpy
      │
[argmax]       next = argmax(logits[0,-1])  (Python, 自回归 cat 进 idx)

CPU 域 = ExecutionEngine JIT 跑的 LLVM IR；CIM 域 = Python hw_simulator（经 cim_stub.so 回调）。桥接点 = 37 个 func.call @cim_launch_<idx>。

0. 预备（invoke 之前，一次性）

- build_llvm_mod：in-memory L2+L3+L4 产出 LLVM mod（不落盘）。
- CIMInvoker：ExecutionEngine(mod, shared_libs=[cim_stub.so]) + 注册 refbackend_consume_func_return_* return callback。
- 方案 A --sim：
  - HwCimSimulator(placed.mlir, weights.bin, partition.json) -> _load 建 idx2name/tile_2bit/instr_map（不预载 Macro）。
  - preload_phase()：分批（681）写 tile 到覆盖区 + PROG_WGT 指令 + 门铃 -> Controller 取指 -> 读覆盖区 2bit 解包 -> Macro[dest]。3664 Macro 权重永久驻留，之后所有 Forward 复用。
  - register_cim_sim_callback(so, sim)：ctypes.CDLL 加载 cim_stub.so，注册 py_cim_sim 回调到 g_cim_sim_cb。sim_lib 必须保存（持回调引用防 GC 段错误）。
- build_inputs：50 个 numpy（每层 inv_freq/causal_mask/qkvofc1fc2 w_packed ×6 + lm_head.w_packed + idx）。w_packed uint8->view int8 匹配 MLIR i8。

1. invoke main 启动

- invoker.invoke("main", *inputs)：每个 input 经 get_unranked_memref_descriptor + ctypes.pointer(ctypes.pointer(...)) 转 FFI 指针。
- main 用 refback-munge calling convention，return 不走常规返回，走 refbackend_consume_func_return_* callback（L4 保留）。
- main 开始按 BitNet forward 拓扑执行：CPU op 直接 LLVM 跑，遇到 func.call @cim_launch_<idx> 跳进 cim_stub.so。

2. 逐层执行（CPU/CIM 交替）

2.1 embedding [CPU]

embed_tokens(idx) -> x[1, T, 512] f32。纯 LLVM（查表 + cast），无 CIM。

2.2 每个 layer（6 层，结构相同）

attention 子层：
- q_proj BitLinear —— CIM 闭环（见 §3 详细）
- k_proj、v_proj BitLinear —— CIM 闭环（同 §3）
- attention [CPU]：split Q/K/V -> RoPE（用 inv_freq/causal_mask input）-> Q@K^T -> softmax -> Attn@V -> GQA。全降级为 tm_tensor + linalg.generic，LLVM 跑。
- o_proj BitLinear —— CIM 闭环
- 残差 x = x + o_proj_out [CPU]

mlp 子层：
- fc1 BitLinear —— CIM 闭环
- act [CPU]：SwiGLU/GLU 激活（linalg.generic）
- fc2 BitLinear —— CIM 闭环
- 残差 [CPU]

2.3 ln_f [CPU] + lm_head [CIM]

- ln_f [CPU]
- lm_head BitLinear —— CIM 闭环（第 37 个，idx=36）
- return logits [1, T, 65]

3. 单个 BitLinear 的 CPU→CIM→CPU 闭环（核心，37 次中的每次）

这是 BitLinearInference.forward 降级后的实际执行。M = seq_len（动态），CPU 一次传 M 个 token，CIM 仿真器内部循环 M 次。

[CPU·LLVM]  x[M,K] f32  (上一层输出)
    │  norm (SubLayerNorm, linalg.generic)
    ▼
[CPU·LLVM]  x_norm[M,K]
    │  scale_x = 127/max|x_norm|  (per-token, [M,1])      §4.7.2 量化
    │  x_int8 = round(x_norm * scale_x).clamp(-128,127)   [M,K] int8
    ▼
[CPU·LLVM]  func.call @cim_launch_<idx>(x_int8, w_packed) : (memref<MxKxi8>, memref<NxK/4xi8>) -> memref<MxNxi32>
    ═══════════════ 桥接 (LLVM -> cim_stub.so -> Python) ═══════════════
[cim_stub.c]  cim_launch_<idx> 从 memref descriptor 拿 xaa+xoff / waa+woff 指针
              malloc result[M*N] int32
              g_cim_sim_cb(IDX, x_ptr, M, K, w_ptr, N, K4, result_ptr)   ← IDX = 该 BitLinear 的全局序号
    │
[py_cim_sim]  np.ctypeslib.as_array 从指针读 x[M,K] int8 + w[N,K4] uint8
              sim.simulate(idx, x, w)    ← w 实际未用 (Macro 已预载权重)
    │
[hw_simulator.simulate]  name = idx2name[idx]   (idx → BitLinear 名 → 该 BitLinear 的指令流 + 预载 Macro)
    │  for m in range(M):              ← 动态 M 循环 (seq_len 运行时吸收, 方案 C)
    │      acc[m] = forward_bitlinear(name, x[m])
    │
[hw_simulator.forward_bitlinear]  (单 token, 重复 M 次)
    │  ① 写 x[m] -> A_PAGE (0x010+k_blk, 覆盖区)          §4.7.3 CPU 搬运激活到共享缓存
    │  ② 写 MATMUL 指令区 (0xBF0) + SYNC_HALT
    │  ③ controller.doorbell(INSTR_BASE*PAGE)             §2.3 CPU 敲门铃, 唤醒取指
    │  ④ Controller._run 取指执行:
    │       MATMUL: read A_PAGE int8[64] × Macro[dest] 三值[64,64] -> int32[64]
    │       UpstreamArbiter.writeback(psum_page, y, accum)  §4.7.4 RMW (首 k_blk ACCUM=0 覆盖, 后续=1 累加)
    │       ... K 维 tile 串行累加 (外层 k 串行避 PSUM_PAGE 冲突)
    │       SYNC_HALT -> break
    │  ⑤ irq_status=DONE, irq_flag=True                   §4.7 IRQ_CIM 置位
    │  ⑥ controller.wait_irq() + int_clear()               CPU 等 IRQ
    │  ⑦ 读 PSUM_PAGE acc[m] (0xC00+n_blk, 取 valid 行去 pad)  §4.7.5 CPU 读累加结果
    ▼
[py_cim_sim]  out[:] = acc    写回 C malloc 的 result buffer
    │
[cim_stub.c]  包装 result 为 Memref2D {result, result, 0, M, N, N, 1} 返回   (绕过 ctypes 不能返回 struct by value)
    ═══════════════ 桥接回 (Python -> cim_stub.so -> LLVM) ═══════════════
[CPU·LLVM]  acc[M,N] int32  (func.call 返回值)
    │  acc.to(f32) / (scale_x * scale_w)                  §4.7.5 CPU rescale (FP32 留 CPU, 不写回共享缓存)
    ▼
[CPU·LLVM]  out[M,N] f32  -> 给下一层 / attention

关键点：
- idx 链路一致：L3 按图遍历顺序给 func.call @cim_launch_<idx> 编 idx（0..36）== partition.json cim_blocks[idx] 顺序 == hw_simulator idx2name[idx]。所以 cim_stub 传的 IDX 直接路由到对应 BitLinear 的指令流 + 预载 Macro。
- w_packed 传而不用：每次 func.call 都传 w_packed memref（main input），但 hw_simulator 忽略它——Preload 后权重已驻留 Macro，Forward 只需激活。这符合真实硬件（Preload 一次，Forward 流式激活）。
- rescale 在 CPU：acc/(scale_x*scale_w) 是 cim.matmul 之后的 CPU op，降级为 linalg.generic 留在 LLVM。FP32 不写回共享缓存（§4.7.5）。
- M 循环在 CIM 侧：CPU 一次传 M 个 token 的 x_int8，仿真器 simulate 内部 for m in range(M) 循环 forward_bitlinear。单 token 指令流静态，seq_len 动态由循环吸收（方案 C）。

4. return + argmax

- main 执行完，logits 经 refbackend_consume_func_return_* callback -> unranked_memref_to_numpy -> numpy [1, T, 65]。
- generate.py 的 jit_generate：next = argmax(logits[0, -1])，idx = cat(idx, next)，循环 N 次出 token。

数据域切换总结

┌────────────────┬────────────────┬────────────────────────────────────────┐
│       段       │       域       │                  数据                  │
├────────────────┼────────────────┼────────────────────────────────────────┤
│ norm + 量化    │ CPU·LLVM       │ x f32 -> x_int8                        │
├────────────────┼────────────────┼────────────────────────────────────────┤
│ func.call 桥接 │ LLVM→C→Python  │ x_int8 memref -> 指针 -> numpy         │
├────────────────┼────────────────┼────────────────────────────────────────┤
│ 写 A_PAGE      │ CIM·hw_sim     │ x_int8 -> 覆盖区                       │
├────────────────┼────────────────┼────────────────────────────────────────┤
│ MATMUL 取指    │ CIM·Controller │ A_PAGE × Macro -> PSUM_PAGE RMW        │
├────────────────┼────────────────┼────────────────────────────────────────┤
│ 读 acc         │ CIM·hw_sim     │ PSUM_PAGE -> acc int32                 │
├────────────────┼────────────────┼────────────────────────────────────────┤
│ 桥接回         │ Python→C→LLVM  │ acc numpy -> result buffer -> Memref2D │
├────────────────┼────────────────┼────────────────────────────────────────┤
│ rescale        │ CPU·LLVM       │ acc int32 -> out f32                   │
└────────────────┴────────────────┴────────────────────────────────────────┘

一次前向 = 37 次 CPU→CIM→CPU 闭环（6 层 × 6 BitLinear/qkvofc1fc2 + lm_head），中间穿插 CPU attention/act/残差/norm。CIM 侧 Preload 一次（3664 Macro 驻留），Forward 阶段每次闭环走"写 A_PAGE → 门铃 → 取指 MATMUL → IRQ → 读 acc"协议。

要我把某一段再展开吗？比如门铃/IRQ 的时序契约、或 37 次 func.call 与 6 层拓扑的对应关系、或动态 M 循环与单 token 指令流的衔接。