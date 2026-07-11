# 系统级仿真运行时（JIT 模式 + AOT 模式）

两种模式（共用 `cim_stub.c` 不改 + `HwCimSimulator` 不改，差别在 CPU 侧执行方式 + 仿真器连接）：
- **JIT 模式**（§0-5）：CPU 侧 LLVM JIT 进程内执行 + CIM 侧 ctypes 同进程回调。`cim_jit.py` / `run_sim_text.py`。
- **AOT 模式**（§6）：CPU 侧独立可执行文件 `cim_sim` + CIM 侧 IPC（unix socket）跨进程。`cim_compiler/lowering/aot/`。

---

## JIT 模式

把 cpu 侧（LLVM JIT）和 cim 侧（hw_simulator）串起来，看运行时一次前向推理（`invoker.invoke("main", *inputs)`）实际怎么走。

**当前架构**：CPU 侧（cim_stub.c）通过标准 **MMIO 读写 + 门铃/IRQ polling** 驱动 CIM 侧（hw_simulator.py 纯硬件仿真器），桥接点是 4 个 MMIO 回调（不是旧的 py_cim_sim 单回调）。CIM 侧为 **cycle 级时序 + 物理建模**（门铃异步取指 + Macro 并行统计），对齐 cim_mlp.md §2.2/2.3/3/4.6/4.7。

## 总览

```
[预备·一次性]  cim_preload_init(preload.bin)  MMIO 驱动 Preload (3664 Macro 权重驻留)
               + register_cim_hw_sim(4 MMIO 回调) + cim_load_forward(forward.bin) + build_inputs
      │
[invoke main]  ctypes 50 input -> ee.invoke("main")  ── main = BitNet forward 拓扑 (LLVM JIT)
      │
[main 执行]    embed [CPU] ──► 6×layer ──► ln_f [CPU] ──► lm_head [CIM] ──► return logits
                  └ 每层: q/k/v_proj[CIM闭环] -> attention[CPU] -> o_proj[CIM闭环] -> 残差[CPU]
                          fc1[CIM闭环] -> act[CPU] -> fc2[CIM闭环] -> 残差[CPU]
      │
[return]       refbackend_consume_func_return callback -> logits numpy
      │
[argmax]       next = argmax(logits[0,-1])  (Python, 自回归 cat 进 idx)
```

CPU 域 = ExecutionEngine JIT 跑的 LLVM IR；CIM 域 = Python hw_simulator（经 cim_stub.so 的 MMIO 回调）。桥接点 = 37 个 `func.call @cim_launch_<idx>` -> cim_stub.c 的 MMIO 驱动 -> hw_simulator 纯硬件。

## 0. 预备（invoke 之前，一次性）

- **build_llvm_mod**：in-memory L2+L3+L4 产出 LLVM mod（不落盘）。
- **CIMInvoker**：`ExecutionEngine(mod, shared_libs=[cim_stub.so])` + 注册 `refbackend_consume_func_return_*` return callback。
- **方案 A --sim**（cim_jit.py:204-213 / generate.py:113-116）：
  - `sim = HwCimSimulator()` —— **纯硬件，无参数**（不读 IR/weights）。内部建 SharedCache + MacroArray + UpstreamArbiter + BusDispatcher + Controller。`sys.setswitchinterval(0.001)` 设 1ms 线程切换（门铃异步 poll 响应）。
  - `sim_lib = register_cim_hw_sim(so, sim)` —— ctypes.CDLL 加载 cim_stub.so，注册 **4 个 MMIO 回调**（shm_write / shm_read / reg_write / reg_read）到 cim_stub 的 `register_cim_hw_sim`。`sim_lib` 必须保存（持回调引用防 GC 段错误）。同时绑定 `cim_load_forward` / `cim_preload_init` 的 argtypes。
  - `sim_lib.cim_load_forward(forward.bin)` —— C 侧读 forward.bin（CIMF magic）到 `g_fwd_buf`，解析 `g_fwd_off[37]`/`g_fwd_len[37]`，供 `cim_launch_<idx>` 按 idx 查段。
  - `sim_lib.cim_preload_init(preload.bin)` —— **MMIO 驱动 Preload**：C 侧读 preload.bin（CIMP magic），分批（681 tile/批）写 tile 到覆盖区 + 写 PROG_WGT 指令区 + 门铃 + poll IRQ + INT_CLEAR。硬件取指执行 PROG_WGT，把 2bit tile 解包编程到 `Macro[dest].weight`。**3664 Macro 权重永久驻留**，之后所有 Forward 复用（doorbell 不清 weight）。
- **build_inputs**：50 个 numpy（每层 inv_freq/causal_mask/qkvofc1fc2 w_packed ×6 + lm_head.w_packed + idx）。w_packed uint8->view int8 匹配 MLIR i8。

## 1. invoke main 启动

- `invoker.invoke("main", *inputs)`：每个 input 经 `get_unranked_memref_descriptor` + `ctypes.pointer(ctypes.pointer(...))` 转 FFI 指针。
- main 用 refback-munge calling convention，return 走 `refbackend_consume_func_return_*` callback（L4 保留）。
- main 按 BitNet forward 拓扑执行：CPU op 直接 LLVM 跑，遇到 `func.call @cim_launch_<idx>` 跳进 cim_stub.so。

## 2. 逐层执行（CPU/CIM 交替）

### 2.1 embedding [CPU]
embed_tokens(idx) -> x[1, T, 512] f32。纯 LLVM（查表 + cast），无 CIM。

### 2.2 每个 layer（6 层，结构相同）
attention 子层：
- q_proj / k_proj / v_proj BitLinear -- CIM 闭环（见 §3）
- attention [CPU]：split Q/K/V -> RoPE（用 inv_freq/causal_mask input）-> Q@K^T -> softmax -> Attn@V -> GQA。全降级为 tm_tensor + linalg.generic，LLVM 跑。
- o_proj BitLinear -- CIM 闭环
- 残差 x = x + o_proj_out [CPU]

mlp 子层：
- fc1 BitLinear -- CIM 闭环
- act [CPU]：SwiGLU/GLU 激活（linalg.generic）
- fc2 BitLinear -- CIM 闭环
- 残差 [CPU]

### 2.3 ln_f [CPU] + lm_head [CIM]
- ln_f [CPU]
- lm_head BitLinear -- CIM 闭环（第 37 个，idx=36）
- return logits [1, T, 65]

## 3. 单个 BitLinear 的 CPU->CIM->CPU 闭环（核心，37 次中的每次）

M = seq_len（动态）。CPU 一次传 M 个 token 的 x_int8，**C 侧 cim_launch 循环 M 次**（方案 C：单 token 指令流编译期静态，seq_len 运行时吸收）。

```
[CPU·LLVM]  x[M,K] f32  (上一层输出)
    │  norm (SubLayerNorm, linalg.generic)
    ▼
[CPU·LLVM]  x_norm[M,K]
    │  scale_x = 127/max|x_norm|  (per-token, [M,1])      §4.7.2 量化
    │  x_int8 = round(x_norm * scale_x).clamp(-128,127)   [M,K] int8
    ▼
[CPU·LLVM]  func.call @cim_launch_<idx>(x_int8, w_packed) : (memref<MxKxi8>, memref<NxK/4xi8>) -> memref<MxNxi32>
    ═══════════════ 桥接 (LLVM -> cim_stub.so) ═══════════════
[cim_stub.c · cim_launch_<idx>]  (DEF_LAUNCH 宏, cim_stub.c:184-218)
    从 memref descriptor 拿 xaa+xoff / waa+woff 指针; malloc result[M*N] int32
    if (!HW_READY) return cim_launch_impl(...);   ← 无仿真时 CPU 算 matmul (fallback, 用 w)
    n_tiles=ceil(N/64), k_tiles=ceil(K/64)
    ① 写指令区一次:  shm_write(INSTR_BASE*PAGE, g_fwd_base+g_fwd_off[IDX], g_fwd_len[IDX])
                       ↑ idx 查 forward.bin 的第 IDX 段 (MATMUL...+SYNC_HALT, 编译期静态)
    ② for m in 0..M:                          ← M 循环 (seq_len 运行时吸收)
         for kb in 0..k_tiles:  shm_write((A_PAGE_BASE+kb)*PAGE, x+m*K+kb*64, 64)   §4.7.3 写激活
         reg_write(DOORBELL_REG, INSTR_BASE*PAGE)            §2.3 敲门铃, 唤醒取指
         while (reg_read(IRQ_STATUS_REG) != IRQ_DONE) ;      §2.3/4.7 poll IRQ (2=done)
         for nb in 0..n_tiles:  shm_read((PSUM_PAGE_BASE+nb)*PAGE, acc_buf, 256)    §4.7.5 读累加
              s=nb*64; e=min(s+64,N);  memcpy(result+m*N+s, acc_buf, (e-s)*4)        ← 非对齐取 valid
         reg_write(INT_CLEAR_REG, 1)                         §2.3 清中断
    包装 result 为 Memref2D {result, result, 0, M, N, N, 1} 返回
    ═══════════════ MMIO 回调 (cim_stub.so -> Python hw_simulator) ═══════════════
[hw_simulator · MMIO 回调]  (cim_jit.py 4 个 CFUNCTYPE -> sim.mmio_*)
    shm_write(addr,ptr,n)  -> sim.mmio_shm_write:  cache.data[addr:addr+n]=arr,  mmio_cycle+=T_SHM
    shm_read (addr,ptr,n)  -> sim.mmio_shm_read :  arr[:]=cache.data[addr:addr+n], mmio_cycle+=T_SHM
    reg_write(DOORBELL,val) -> sim.mmio_reg_write: controller.doorbell(val)  ← 异步启动 _run 线程
    reg_write(INT_CLEAR,1)  -> sim.mmio_reg_write: controller.int_clear()
    reg_read (IRQ_STATUS)   -> sim.mmio_reg_read :  return controller.irq_status (0/1/2/3)
    ═══════════════ CIM 侧硬件执行 (门铃异步 + cycle 取指) ═══════════════
[hw_simulator · Controller.doorbell]  (hw_simulator.py:202-210)
    arbiter.reset()  (清 page_busy, 新指令段 PSUM_PAGE 独立)
    for m in macros: m.busy_until=0  (新指令段 Macro idle, 清前序残留; 不清 m.weight §4.6 两阶段)
    irq_status = BUSY
    Thread(_run, addr).start()  ← 异步, CPU 端 poll IRQ_STATUS
[hw_simulator · Controller._run]  (cycle 级取指, hw_simulator.py:211-241)
    cycle = 0
    while True:
        w = cache.read_instr(addr); addr+=6; cycle += T_FETCH           §3 取指
        op,dest,p1,p2,accum = 解码 48-bit  (op=<<45|dest<<33|p1<<21|p2<<9|accum<<8)
        if op==PROG_WGT:  dispatcher.dispatch_prog_wgt(dest,p1,cycle)   非阻塞 (更新 busy_until)
        if op==MATMUL:    dispatcher.dispatch_matmul(dest,p1,p2,accum,cycle)  非阻塞
              └ dispatch_matmul: 读 A_PAGE int8[64] × Macro[dest].weight 三值[64,64] -> int32[64] (数值同步)
                 arbiter.writeback(psum_page, y, accum):  start=max(finish,page_busy[psum_page])
                 cache.rmw_int32(psum_page, y, accum)  §4.7.4 RMW (accum=0 覆盖, =1 累加)
                 page_busy[psum_page] = start + T_WB   (同 PSUM_PAGE 串行, 不同并行)
        if op==SYNC_HALT: cycle = max([cycle]+all_busy+all_page); break  §3.4/4.7.7 join
    self.cycle = cycle;  irq_status = DONE
    ═══════════════ 桥接回 (Python -> cim_stub.so -> LLVM) ═══════════════
[CPU·LLVM]  acc[M,N] int32  (func.call 返回值)
    │  acc.to(f32) / (scale_x * scale_w)                  §4.7.5 CPU rescale (FP32 留 CPU, 不写回共享缓存)
    ▼
[CPU·LLVM]  out[M,N] f32  -> 给下一层 / attention
```

## 4. CIM 侧硬件协议（MMIO + 门铃/IRQ + cycle 时序）

### 4.1 地址布局（对齐 cim_mlp.md §2.3/4.6，cim_stub.c:46-57）
```
寄存器 (§2.3, 相对 REG_BASE=0x20000000):
  DOORBELL_REG   = 0x20000000  写: 指令区起始 byte addr, 唤醒取指
  INT_CLEAR_REG  = 0x20000004  写: 清中断
  IRQ_STATUS_REG = 0x20000008  读: 0=idle 1=busy 2=done 3=error
共享缓存 (§4.6, 1MB, PAGE=256B, byte offset = page*PAGE):
  覆盖区 0x000~0xBEF (3056 PAGE)  Preload 暂存权重 / Forward int8 输入 (两阶段复用)
  指令区 0xBF0~0xBFF (16 PAGE, 4KB)  48-bit 指令 (~682 条)
  累加区 0xC00~0xFFF (1024 PAGE)  int32 部分和 RMW
Forward PAGE 绑定 (place.py): A_PAGE=0x010+k_blk, PSUM_PAGE=0xC00+n_blk
Preload PAGE 绑定 (place.py): b_page_start=(dest_id%681)*4  (每 tile 4 PAGE=1024B, 批内复用)
```

### 4.2 MMIO 回调桥接（4 回调，cim_jit.py:126-160）
cim_stub.c 的 `shm_write/shm_read/reg_write/reg_read` 经 `register_cim_hw_sim` 注册的 4 个 CFUNCTYPE 回调转发到 `sim.mmio_*`。CPU 侧完全不感知 Python，只做标准 MMIO 读写 + 门铃/IRQ polling，与真实硬件驱动骨架一致。

### 4.3 门铃异步 + IRQ 协议（§2.3/4.7）
- **门铃异步**：`mmio_reg_write(DOORBELL)` -> `controller.doorbell(val)` 启动 `_run` 守护线程，立即返回（CPU 不阻塞）。
- **CPU poll IRQ**：cim_stub `while (reg_read(IRQ_STATUS) != IRQ_DONE)` 自旋读寄存器；Python 端 `wait_irq` 用 `time.sleep(0)` 让 GIL 给 `_run` 线程。`sys.setswitchinterval(0.001)` 保证 1ms 内线程切换，C-poll 与 Python _run 线程 GIL 交替正常。
- **状态机**：doorbell 设 BUSY -> _run 完成设 DONE（或异常设 ERROR）-> CPU 读到 DONE -> INT_CLEAR 回 IDLE。

### 4.4 cycle 级时序 + Macro 并行建模（§3/4.7.7）
**数值同步 + 时序统计解耦**：dispatch 同步算 matmul + RMW（保证数值正确，max_diff=0），`busy_until`/`page_busy` 统计并行时序（不破坏数值）。

cycle 参数（§3 无规定，估算）：`T_FETCH=1, T_DISPATCH=2, T_PROG_WGT=10, T_MATMUL=64, T_WB=4, T_SHM=2, T_REG=1`

并行保证：
- **Macro.busy_until**：`start = max(cycle, m.busy_until)`，同 Macro 串行（§4.7.7），不同 Macro 独立并行。
- **Arbiter.page_busy**：`start = max(finish, page_busy[psum_page])`，同 PSUM_PAGE 串行 RMW（K 维累加顺序），不同 PAGE 并行。
- **Controller._run**：`cycle += T_FETCH` per 指令，dispatch **非阻塞**（不 `cycle = max(cycle, finish)`），SYNC_HALT `cycle = max(all_busy, all_page)` join。
- **doorbell 清 busy_until/page_busy**：新指令段 Macro/Arbiter idle（不清 weight，§4.6 两阶段）。

段内调度（forward.bin 单段，外层 kb 串行 / 内层 nb 并行）：同 kb 不同 nb（不同 Macro + 不同 PSUM_PAGE）并行；不同 kb 同 nb（同 PSUM_PAGE）RMW 串行累加。

验证（q.proj 8×8 tile，单 token）：cim_cycle=134（串行 4480，并行度 33.4x）；整体并行度 40.92x；max_diff=0。

### 4.5 编译期产物（emit_instr.py）
- **forward.bin**（CIMF）：`magic + n_idx=37 + offsets[37] + lengths[37] + 37 段`。每段 = `n_tiles×k_tiles` 条 MATMUL（`dest_id=nb*k_tiles+kb` 全局连续唯一 [0,3663]，`a_page=0x010+kb` 广播、`psum_page=0xC00+nb` RMW、`accum=(kb>0)`）+ 1 条 SYNC_HALT。`cim_launch_<idx>` 用 idx 查段。
- **preload.bin**（CIMP）：`magic + n_batch + batch_offsets + 批`。每批 = `n_tile | tile_data(1024B/tile, 真实 2bit 权重) | PROG_WGT 指令 | SYNC_HALT`，681 tile/批（指令区 4KB/6B=682 条约束）。`cim_preload_init` 读此文件 MMIO 驱动 Preload。

## 5. return + argmax
- main 执行完，logits 经 `refbackend_consume_func_return_*` callback -> `unranked_memref_to_numpy` -> numpy [1, T, 65]。
- generate.py 的 `jit_generate`：`next = argmax(logits[0, -1])`，`idx = cat(idx, next)`，循环 N 次出 token。

## 数据域切换总结

| 段 | 域 | 数据 |
|---|---|---|
| norm + 量化 | CPU·LLVM | x f32 -> x_int8 |
| func.call 桥接 | LLVM->cim_stub | x_int8 memref -> 指针 |
| 写指令区 + A_PAGE | cim_stub·MMIO | forward.bin 段 + x_int8 -> 覆盖区 (shm_write) |
| 门铃 | cim_stub->hw_sim | reg_write(DOORBELL) -> 异步 _run |
| MATMUL 取指 | CIM·Controller | A_PAGE × Macro.weight -> PSUM_PAGE RMW (cycle 级) |
| poll IRQ | cim_stub->hw_sim | reg_read(IRQ_STATUS) until DONE |
| 读 acc | cim_stub·MMIO | PSUM_PAGE -> acc int32 (shm_read, 取 valid 行) |
| INT_CLEAR | cim_stub->hw_sim | reg_write(INT_CLEAR) -> IDLE |
| 桥接回 | cim_stub->LLVM | acc -> result buffer -> Memref2D |
| rescale | CPU·LLVM | acc int32 -> out f32 |

## 关键点

- **MMIO 协议驱动**：CPU 侧（cim_stub.c）只做标准 MMIO 读写 + 门铃/IRQ polling，不感知 Python；4 个 MMIO 回调桥接到 hw_simulator。与真实硬件驱动骨架一致（地址布局对齐 §2.3/4.6）。
- **门铃异步 + cycle 取指**：doorbell 启动 _run 守护线程，CPU poll IRQ_STATUS；_run cycle 级取指（T_FETCH），dispatch 非阻塞，SYNC_HALT join。数值同步 + 时序统计解耦（max_diff=0 + 并行度 33x）。
- **Macro 并行时序**：busy_until（同 Macro 串行/不同并行）+ page_busy（同 PSUM_PAGE 串行 RMW）+ SYNC_HALT join。段内外层 kb 串行 / 内层 nb 并行。
- **idx 链路一致**：L3 按图遍历顺序给 `func.call @cim_launch_<idx>` 编 idx（0..36）== forward.bin 段序号 == emit 时 func 顺序。cim_stub 传的 IDX 直接查 forward.bin 第 IDX 段。
- **w_packed 传而不用（--sim 路径）**：每次 func.call 都传 w_packed memref（main input），但 --sim 路径 HW_READY=1 不 fallback，w 不用——Preload 后权重已驻留 Macro，Forward 只需激活。符合真实硬件（Preload 一次，Forward 流式激活）。无仿真时 fallback `cim_launch_impl` 用 w CPU 算。
- **M 循环在 C 侧**：cim_launch 写指令区一次（accum 固化，首 kb 清旧 acc），M 次门铃+IRQ 复用。单 token 指令流静态，seq_len 动态由 C 循环吸收（方案 C）。
- **rescale 在 CPU**：`acc/(scale_x*scale_w)` 是 cim.matmul 之后的 CPU op，降级为 linalg.generic 留 LLVM。FP32 不写回共享缓存（§4.7.5）。
- **Preload 一次性**：`cim_preload_init` 在 invoke main 之前调用一次，3664 Macro 权重永久驻留；doorbell 只清 busy_until/page_busy 不清 weight（§4.6 两阶段）。

一次前向 = 37 次 CPU->CIM->CPU 闭环（6 层 × 6 BitLinear/qkvofc1fc2 + lm_head），中间穿插 CPU attention/act/残差/norm。CIM 侧 Preload 一次（3664 Macro 驻留），Forward 每次闭环走"写指令区 + A_PAGE -> 门铃 -> 取指 MATMUL(cycle 级) -> poll IRQ -> 读 acc -> INT_CLEAR"MMIO 协议。

---

## 6. AOT 模式（CPU 可执行文件 + IPC 仿真器）

JIT 模式的演进：CPU 侧从 LLVM JIT 进程内执行 -> **AOT 独立可执行文件**，CIM 侧从 ctypes 同进程回调 -> **IPC（unix socket）跨进程**。模拟真实 CPU<->硬件分离（CPU 主动驱动，硬件被动响应，经 MMIO 协议）。

### 6.1 架构（`cim_compiler/lowering/aot/`）

```
cim_sim (ELF 可执行文件) = 链接:
  模型 .o  (to_object.py dump + objcopy --redefine-sym main=forward_entry)
    ├ _mlir_ciface_main  (50 UnrankedMemRefDescriptor*, refbackend C 入口, emit_c_interface)
    ├ forward_entry      (内部 main, 100 参数 {rank,descriptor}×50, objcopy 改名避冲突)
    └ @cim_launch / @refbackend_consume_func_return_* external declare
  cim_stub.o   (cim_launch + register_cim_hw_sim + cim_load_forward + cim_preload_init, 不改)
  cim_runtime.o (C 版 _mlir_ciface_consume + unranked memref 构造工具, 对齐 torch_mlir runtime ABI)
  cim_ipc.o    (IPC client: shm_* memcpy 共享内存 + reg_* socket, cim_ipc_init register)
  cim_shm.o    (POSIX 共享内存抽象: shm_open/ftruncate/mmap, C/Python 共享 1MB)
  cim_main.o   (main 入口: ipc_init + load_forward + preload_init + generate 循环 + tokenizer)

cim_sim_server.py (Python, 独立进程, 循环 accept 持久服务):
  HwCimSimulator (cache.data backed by SharedMemory) + SharedMemory("cim_cache") + unix socket
  reg_* handler -> sim.mmio_reg_write/read (shm_* 不经 socket, C 直接写共享内存)
```

### 6.2 启动
```
终端1: python cim_compiler/lowering/aot/cim_sim_server.py        # 仿真器 server (先启动)
终端2: ./cim_compiler/lowering/aot/cim_sim --prompt "ROMEO:" --n 60  # AOT 可执行文件
一键:  ./cim_compiler/lowering/aot/run_aot.sh --prompt "ROMEO:" --n 60  # nohup server + cim_sim + kill
构建:  make -C cim_compiler/lowering/aot                            # 全量构建 cim_sim
```

### 6.3 数据流
```
cim_main: cim_ipc_init (连 server + register_cim_hw_sim IPC 回调 -> HW_READY=1)
        -> cim_load_forward (CPU 本地读 forward.bin, 建 idx->段表)
        -> cim_preload_init (经 IPC 驱动 server Preload, 3664 Macro 权重驻留)
        -> generate 循环 (greedy, 对齐 run_sim_text):
             构造 50 memref (49 buffer 复用 + idx 每步重建) -> _mlir_ciface_main
             -> forward_entry -> cim_launch(idx, X, W) [W 空壳, cim_launch 不读, 用 idx 查 forward.bin]
             -> cim_stub MMIO 回调 (shm_write A_PAGE/指令 + shm_read PSUM 走共享内存零往返; reg_write 门铃 + reg_read poll 走 socket)
             -> 共享内存 (shm_*) + socket (reg_*) -> server HwCimSimulator (MATMUL cycle 级)
             -> _mlir_ciface_consume 收 logits (rank=3 [1,seq,65]) -> argmax(logits[0,last,:]) -> append
        -> decode (itos) 输出文本
```

### 6.4 IPC 协议（共享内存 shm 数据 + unix socket reg 控制）

shm_*（数据传输，大块高频）：C/Python 共享 1MB POSIX 共享内存（`/dev/shm/cim_cache`，承载 CIM 共享缓存 `SharedCache.data`），C 直接 `memcpy` 读写（零拷贝零往返，替代 socket）。
reg_*（控制信号，少量）：走 unix socket（作同步点，保证可见性）：
```
reg_write: req [op=3|reg(8)|val(8)] -> resp [ack(1)]
reg_read:  req [op=4|reg(8)]        -> resp [val(4)]
```
同步语义：C 写 shm -> `reg_write(DOORBELL)` socket -> Python 读（socket 往返保证 C 写可见）；Python 写 PSUM -> `irq=DONE` socket -> C `shm_read`（DONE 同步保证 PSUM 可见）。

共享内存：C `shm_open("/cim_cache")` == Python `SharedMemory("cim_cache")`。server 创建（`create=True`），cim_sim 打开。`SharedCache.data = np.frombuffer(shm.buf)` backed by 共享内存，所有 `read_bytes`/`rmw_int32` 方法不变。

**cim_stub.c 不改**：`register_cim_hw_sim` 注册 4 回调（shm 走共享内存，reg 走 socket），机制天然支持换回调源。

### 6.5 关键技术点
- **AOT 编译**：`to_object.py` dump `.o`（`shared_libs=[cim_stub.so, cim_runtime.so]` 提供 consume 符号 resolved）+ `objcopy --redefine-sym main=forward_entry`（避免与 cim_main 的 main 冲突）。
- **consume C ABI**：main 调 `@refbackend_consume_func_return_mrf32(rank, descriptor)` [wrapper, .o 内部] -> `_mlir_ciface_refbackend_consume_func_return_mrf32(sp)` [宿主提供, cim_runtime.o]。`sp` 指向 `{i64 rank, void* descriptor}`，descriptor -> ranked buffer `{allocated, aligned, offset, shape[rank], strides[rank]}`。
- **C 入口**：`_mlir_ciface_main(50 UnrankedMemRefDescriptor*)`，每个参数指向 `{rank, descriptor}` struct（ciface body load + extractvalue 拆开传内部 main）。C main 传 `&desc[i]`。
- **w_packed 空壳**：cim_launch(idx,X,W) 不读 W 数据（Preload 已驻留 Macro，Forward 用 idx 查 forward.bin + Macro 权重），W memref 仅 shape 对（N,K/4）即可，数据零。省去 C 侧加载 37 个权重。
- **无 KV cache**：`_LogitsOnly.forward(idx)` 无状态（embed->layers->ln_f->lm_head），每步重算（O(n²)），C main 无状态机。对齐 run_sim_text。
- **CharTokenizer C 化**：`tokenizer_data.h`（itos[65] + stoi[128]）从 `bitnet/data_char.py` meta 导出，C main encode/decode。

### 6.6 与 JIT 模式对比
| | JIT 模式 (§0-5) | AOT 模式 (§6) |
|---|---|---|
| CPU 侧 | ExecutionEngine JIT 进程内 | 独立 ELF 可执行文件 |
| 仿真器连接 | ctypes 同进程回调 | IPC unix socket 跨进程 |
| consume | Python ctypes 回调 | C 版 `_mlir_ciface_consume` |
| 输入构造 | Python `build_inputs` | C main 构造 memref |
| 生成循环 | Python `generate` | C main `generate` |
| cim_stub.c | 不改 | 不改 |
| HwCimSimulator | 不改 | 不改 |

### 6.7 验证
- **单步 forward**：cim_sim logits 与 Python 参考模型 `model(idx)` max_diff=0（`[4.0113, 1.6103, -6.1351, -8.0173, -7.4788]` 完全一致）。
- **文本生成**：cim_sim `--prompt R --n 3` 输出 `RD I` 与 `run_sim_text.py` (JIT) 完全一致；`--prompt ROMEO: --n 8` 输出 `ROMEO:\n` 一致。
- **server 统计**：n=3 共 12010 MMIO，cim_cycle=100。

### 6.8 性能
- **socket 模式**（早期）：每条 MMIO（shm+reg）一次 socket 往返，n=3 ~1.5s / 12010 MMIO。
- **共享内存模式**（当前）：shm_* 走共享内存零往返，仅 reg_* 走 socket。n=3 ~0.93s / 3307 reg-socket（消除 ~8700 shm 往返，**快 ~40%**）。
- 剩余开销：reg_* socket 往返（`reg_read` poll IRQ 高频）。阶段 2 优化方向：irq_status 放共享内存控制区 + doorbell 用 eventfd（poll 零往返）。
- generate O(n²) 累积，大批量仍慢。

2. IR 抽象层次单一——cimres dialect 只有 3 种操作（preload_weight/macro_matmul/sync_halt），从 tile 级直接到指令级。成熟 DSL（Triton 的 block→program、TensorIR 的多级 schedule）有多层 IR 逐步 lowering。这里 place.py 之后基本是直接编码，缺少中间调度 IR。
3. 无自动调度搜索——place.py 页分配是确定性算法，emit_instr.py 顺序编码。没有调度空间搜索（对比 Ansor 的多阶段搜索、Triton 的 autotune）。对固定 tile 硬件这是合理取舍，但限制了复杂拓扑的优化空间。
4. 优化 pass 少且硬编码——tile 64×64 固定、分块策略固定。缺算子融合、循环变换、内存复用分析等通用 pass。cim_stub.c 的 A_PAGE/PSUM_PAGE 串行复用是手工设计的，非编译器推导。