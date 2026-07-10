#!/usr/bin/env python3
"""完整硬件级 CIM 仿真器 - cycle 级时序 + 物理建模 (对应 cim_mlp.md §2-4)。

纯硬件 + MMIO + cycle 级时序: 完整模拟真实硬件处理流程 (§2.2 9 步):
  CPU MMIO(AXI/APB) -> 门铃 -> 控制器取指 -> BusDispatcher 广播 -> Macro 并行执行
  -> 上行仲裁写回 -> SYNC_HALT join -> IRQ -> CPU 读结果

时序建模 (§4.7.7 宏间并行):
  - 同一 Macro 串行 (busy_until), 不同 Macro 并行 (独立 busy_until)
  - 同 PSUM_PAGE 串行 RMW (page_busy), 不同 PAGE 并行
  - SYNC_HALT join (等所有 Macro + Arbiter 完成, §3.4)
  - 门铃异步 (线程): CPU 写门铃立即返回, CIM 后台执行, CPU poll IRQ_STATUS (§2.3)

数值同步 + 时序统计解耦: dispatch 同步算 matmul+RMW (数值正确, max_diff=0),
  busy_until 统计并行时序 (不破坏数值)。同 PSUM_PAGE 的 RMW 顺序靠指令顺序
  (C1 外层 k 串行) + page_busy 双重保证。

组件 (§2.2):
  SharedCache     - 1MB 共享缓存, PAGE 寻址, 三区 (§4.6)
  MacroArray      - 4096 Macro (64×64 2bit), busy_until 并行时序 (§4.7.7)
  Controller      - 门铃异步 + cycle 取指 + IRQ 状态机 (§2.3/3.4)
  BusDispatcher   - Dest_ID 广播总线 (T_DISPATCH, §2.2)
  UpstreamArbiter - 上行仲裁 (同 PAGE 串行 RMW, T_WB, §2.1/2.2)

MMIO (§2.3): shm (共享缓存, T_SHM) + reg (寄存器, T_REG) + cycle 统计。

运行 (self-test): nanogpt-gpu python cim_compiler/cimres/hw_simulator.py
"""
import os
import sys
import math
import time
import struct
import ctypes
import threading
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CIM_COMPILER = os.path.dirname(HERE)
REPO = os.path.dirname(CIM_COMPILER)
EXPORT_DIR = os.path.join(CIM_COMPILER, "export")
for _p in (REPO, EXPORT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cim_compiler.export.weight_blob import read_weight_blob

TILE = 64
SHARED_SIZE = 1 << 20
PAGE = 256
OVERWRITE_BASE = 0x000
INSTR_BASE = 0xBF0
ACCUM_BASE = 0xC00
A_PAGE_BASE = 0x010
PSUM_PAGE_BASE = 0xC00
INSTR_CAPACITY = 16 * PAGE // 6
PRELOAD_BATCH = INSTR_CAPACITY - 1
REG_BASE_DEFAULT = 0x20000000
DOORBELL_OFF = 0x00
INT_CLEAR_OFF = 0x04
IRQ_STATUS_OFF = 0x08
IDLE, BUSY, DONE, ERROR = 0, 1, 2, 3
OP_PROG_WGT = 0x1
OP_MATMUL = 0x2
OP_SYNC_HALT = 0x7
FORWARD_MAGIC = b"CIMF"
PRELOAD_MAGIC = b"CIMP"

# ===== cycle 开销 (§3 无规定, 估算默认值, 真实硬件可调) =====
T_FETCH    = 1    # 取指 (1 条 48-bit, §3.1)
T_DISPATCH = 2    # 广播总线 Dest_ID 12b 路由 + 负载 (§2.2)
T_PROG_WGT = 10   # 64×64 2bit (1024B) 解包 + load (§3.2)
T_MATMUL   = 64   # 64×64 int8×ternary MvM -> int32 (§3.3, 64 输出每 cycle 一列)
T_WB       = 4    # 上行仲裁 + int32 ALU RMW (§2.1/2.2)
T_SHM      = 2    # 共享缓存 MMIO (AXI, per 64B, §2.2)
T_REG      = 1    # 寄存器 MMIO (§2.3)


def unpack_2bit_np(packed):
    """uint8[..., K//4] (2bit 补码) -> int8[..., K] {-1,0,1}。"""
    p = packed.astype(np.int32)
    c0 = p % 4
    c1 = (p // 4) % 4
    c2 = (p // 16) % 4
    c3 = (p // 64) % 4
    code = np.stack([c0, c1, c2, c3], axis=-1).reshape(packed.shape[0], -1)
    return np.where(code >= 2, code - 4, code).astype(np.int8)


def _norm(name):
    return name.replace("_", ".")


# ===================== 硬件组件 (§2.2) =====================
class SharedCache:
    """1MB 共享缓存, PAGE 寻址, 三区 (§2.1/§4.6)。"""
    def __init__(self):
        self.data = np.zeros(SHARED_SIZE, dtype=np.uint8)
        self.data32 = self.data.view(np.int32)

    def read_bytes(self, byte_addr, n):
        return bytes(self.data[byte_addr:byte_addr + n])

    def read_int8_vec(self, page, n=TILE):
        addr = page * PAGE
        return self.data[addr:addr + n].astype(np.int8).astype(np.int32)

    def rmw_int32(self, page, y_int32, accum):
        a32 = page * PAGE // 4
        n = len(y_int32)
        if not accum:
            self.data32[a32:a32 + n] = y_int32
        else:
            self.data32[a32:a32 + n] = self.data32[a32:a32 + n] + y_int32

    def read_int32_vec(self, page, n=TILE):
        a32 = page * PAGE // 4
        return self.data32[a32:a32 + n].copy()

    def read_instr(self, byte_addr):
        b = self.data[byte_addr:byte_addr + 6]
        return int.from_bytes(bytes(b) + b"\x00\x00", "little")


class Macro:
    """单个 64×64 Macro: 三值权重 + busy_until (同 Macro 串行, §4.7.7)。"""
    __slots__ = ("weight", "busy_until")
    def __init__(self):
        self.weight = None       # [64,64] int8
        self.busy_until = 0      # cycle, 同 Macro 串行


class MacroArray:
    """4096 个 64×64 三值寄存器 + 并行时序 (§2.2/§4.7.7)。
    同 Macro 串行 (busy_until), 不同 Macro 并行 (独立 busy_until)。"""
    def __init__(self):
        self.macro = {}   # dest_id -> Macro

    def load(self, dest_id, tile_2bit_1024B, cycle):
        """PROG_WGT: 解包 2bit -> Macro[dest]. 返回完成 cycle (T_DISPATCH+T_PROG_WGT)。"""
        m = self.macro.get(dest_id)
        if m is None:
            m = Macro(); self.macro[dest_id] = m
        start = max(cycle, m.busy_until)                       # 同 Macro 串行
        packed = np.frombuffer(tile_2bit_1024B, dtype=np.uint8).reshape(TILE, TILE // 4)
        m.weight = unpack_2bit_np(packed)                      # 数值同步
        m.busy_until = start + T_DISPATCH + T_PROG_WGT
        return m.busy_until

    def matmul(self, dest_id, x_int8_64, cycle):
        """MATMUL: int8[64] × 三值[64,64] -> int32[64]. 返回 (y, 完成 cycle)。"""
        m = self.macro[dest_id]
        start = max(cycle, m.busy_until)                       # 同 Macro 串行
        y = m.weight.astype(np.int32) @ x_int8_64.astype(np.int32)  # 数值同步
        m.busy_until = start + T_DISPATCH + T_MATMUL
        return y, m.busy_until


class UpstreamArbiter:
    """上行总线仲裁器: int32 写回累加区 RMW (§2.1/2.2)。
    同 PSUM_PAGE 串行 (page_busy, K维 RMW 顺序), 不同 PAGE 并行。"""
    def __init__(self, cache):
        self.cache = cache
        self.page_busy = {}   # psum_page -> busy_until

    def writeback(self, psum_page, y_int32, accum, finish_cycle):
        start = max(finish_cycle, self.page_busy.get(psum_page, 0))  # 同 PAGE 串行 RMW
        self.cache.rmw_int32(psum_page, y_int32, accum)             # 数值同步 (int32 ALU)
        self.page_busy[psum_page] = start + T_WB
        return self.page_busy[psum_page]

    def reset(self):
        self.page_busy.clear()


class BusDispatcher:
    """多宏分发与总线驱动: Dest_ID 广播路由 (§2.2/§4.7.7)。T_DISPATCH 计入 load/matmul。"""
    def __init__(self, macros, cache, arbiter):
        self.macros = macros
        self.cache = cache
        self.arbiter = arbiter

    def dispatch_prog_wgt(self, dest_id, b_page_start, cycle):
        tile = self.cache.read_bytes(b_page_start * PAGE, 1024)
        return self.macros.load(dest_id, tile, cycle)         # 含 T_DISPATCH+T_PROG_WGT

    def dispatch_matmul(self, dest_id, a_page, psum_page, accum, cycle):
        x = self.cache.read_int8_vec(a_page, TILE)
        y, finish = self.macros.matmul(dest_id, x, cycle)     # 含 T_DISPATCH+T_MATMUL
        return self.arbiter.writeback(psum_page, y, accum, finish)  # T_WB


class Controller:
    """控制器: 门铃异步 + cycle 级取指 + IRQ 状态机 (§2.2/2.3/3.4/4.7.7)。
    门铃异步 (线程): doorbell 启动 _run 线程, CPU poll IRQ_STATUS。
    cycle 级取指: T_FETCH per 指令, dispatch 非阻塞 (busy_until), SYNC_HALT join。"""
    def __init__(self, cache, dispatcher):
        self.cache = cache
        self.dispatcher = dispatcher
        self.irq_status = IDLE
        self.cycle = 0          # 本次 doorbell 累计 cycle
        self._thread = None

    def doorbell(self, instr_byte_addr):
        """CPU 写门铃: 设 BUSY, 启动 _run 线程 (异步, §2.3)。新指令段, Macro/Arbiter idle。"""
        self.dispatcher.arbiter.reset()   # 新指令段, PSUM_PAGE 独立 (清 page_busy)
        for m in self.dispatcher.macros.macro.values():
            m.busy_until = 0              # 新指令段, Macro idle (清 preload/前序残留, §4.6 两阶段)
        self.irq_status = BUSY
        self._thread = threading.Thread(target=self._run, args=(instr_byte_addr,), daemon=True)
        self._thread.start()

    def _run(self, addr):
        """cycle 级取指执行至 SYNC_HALT (§2.3/3.4)。数值同步 + busy_until 时序统计。"""
        cycle = 0
        macros = self.dispatcher.macros
        arbiter = self.dispatcher.arbiter
        try:
            while True:
                w = self.cache.read_instr(addr); addr += 6; cycle += T_FETCH
                op = (w >> 45) & 0x7
                dest = (w >> 33) & 0xFFF
                p1 = (w >> 21) & 0xFFF
                p2 = (w >> 9) & 0xFFF
                accum = (w >> 8) & 1
                if op == OP_PROG_WGT:
                    self.dispatcher.dispatch_prog_wgt(dest, p1, cycle)            # 非阻塞 (更新 busy_until)
                elif op == OP_MATMUL:
                    self.dispatcher.dispatch_matmul(dest, p1, p2, accum, cycle)   # 非阻塞 (更新 busy_until/page_busy)
                elif op == OP_SYNC_HALT:
                    # join: 等所有 Macro + Arbiter 完成 (§3.4/4.7.7)
                    all_busy = [m.busy_until for m in macros.macro.values()]
                    all_page = list(arbiter.page_busy.values())
                    cycle = max([cycle] + all_busy + all_page)
                    break
                else:
                    self.irq_status = ERROR
                    return
            self.cycle = cycle
            self.irq_status = DONE
        except Exception:
            self.irq_status = ERROR

    def int_clear(self):
        self.irq_status = IDLE


# ===================== HwCimSimulator (纯硬件 + MMIO + cycle 统计) =====================
class HwCimSimulator:
    """纯硬件 CIM 仿真器: MMIO 接口 + cycle 级时序统计。
    CPU (cim_stub.c) 经 MMIO 读写操作, 硬件门铃异步取指执行。"""

    def __init__(self, reg_base=REG_BASE_DEFAULT):
        sys.setswitchinterval(0.001)   # 1ms 线程切换 (门铃异步 poll 响应)
        self.cache = SharedCache()
        self.macros = MacroArray()
        self.arbiter = UpstreamArbiter(self.cache)
        self.dispatcher = BusDispatcher(self.macros, self.cache, self.arbiter)
        self.controller = Controller(self.cache, self.dispatcher)
        self.REG_BASE = reg_base
        self.DOORBELL_REG = reg_base + DOORBELL_OFF
        self.INT_CLEAR_REG = reg_base + INT_CLEAR_OFF
        self.IRQ_STATUS_REG = reg_base + IRQ_STATUS_OFF
        self.mmio_cycle = 0     # MMIO cycle 累计 (CPU 侧)

    # ---- 内部 helper (Python 端 driver 用) ----
    def _shm_write(self, byte_addr, arr):
        v = np.frombuffer(arr, dtype=np.uint8) if isinstance(arr, (bytes, bytearray)) else np.asarray(arr, dtype=np.uint8)
        self.cache.data[byte_addr:byte_addr + len(v)] = v
        self.mmio_cycle += T_SHM

    def _shm_read(self, byte_addr, n):
        self.mmio_cycle += T_SHM
        return self.cache.data[byte_addr:byte_addr + n].copy()

    # ---- MMIO 回调 (C 经 ctypes 调用) ----
    def mmio_shm_write(self, byte_addr, ptr, n):
        arr = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8)), (n,))
        self.cache.data[byte_addr:byte_addr + n] = arr
        self.mmio_cycle += T_SHM

    def mmio_shm_read(self, byte_addr, ptr, n):
        arr = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8)), (n,))
        arr[:] = self.cache.data[byte_addr:byte_addr + n]
        self.mmio_cycle += T_SHM

    def mmio_reg_write(self, reg, val):
        self.mmio_cycle += T_REG
        if reg == self.DOORBELL_REG:
            self.controller.doorbell(val)      # 异步启动 _run
        elif reg == self.INT_CLEAR_REG:
            self.controller.int_clear()

    def mmio_reg_read(self, reg):
        self.mmio_cycle += T_REG
        if reg == self.IRQ_STATUS_REG:
            return int(self.controller.irq_status)
        return 0

    def wait_irq(self):
        """CPU poll IRQ_STATUS until DONE (§2.3/4.7)。Python 端 driver 用。"""
        while self.controller.irq_status != DONE:
            if self.controller.irq_status == ERROR:
                raise RuntimeError("CIM ERROR (未知 opcode 或 _run 异常)")
            time.sleep(0)   # 让 GIL 给 _run 线程

    def stats_snapshot(self):
        """返回时序统计: CIM cycle / MMIO cycle / Macro 数。"""
        return {
            "cim_cycle": self.controller.cycle,
            "mmio_cycle": self.mmio_cycle,
            "n_macro": len(self.macros.macro),
        }


# ===================== Python 端 driver (模拟 cim_stub.c, self-test 用) =====================
def driver_preload(sim, preload_path):
    """模拟 cim_preload_init: 读 preload.bin (自包含), 分批 MMIO 驱动 Preload。"""
    data = open(preload_path, "rb").read()
    assert data[:4] == PRELOAD_MAGIC, f"bad preload magic: {data[:4]}"
    n_batch = struct.unpack("<I", data[4:8])[0]
    b_off = struct.unpack(f"<{n_batch}I", data[8:8 + n_batch * 4])
    body = 8 + n_batch * 4
    for off in b_off:
        p = body + off
        n_tile = struct.unpack("<I", data[p:p + 4])[0]
        p += 4
        tile_data = data[p:p + n_tile * 1024]
        p += n_tile * 1024
        instr_size = n_tile * 6 + 6
        instrs = data[p:p + instr_size]
        for i in range(n_tile):
            sim._shm_write((OVERWRITE_BASE + i * 4) * PAGE,
                           np.frombuffer(tile_data[i * 1024:(i + 1) * 1024], dtype=np.uint8))
        sim._shm_write(INSTR_BASE * PAGE, np.frombuffer(instrs, dtype=np.uint8))
        sim.mmio_reg_write(sim.DOORBELL_REG, INSTR_BASE * PAGE)   # 门铃 (异步)
        sim.wait_irq()                                            # poll IRQ
        sim.mmio_reg_write(sim.INT_CLEAR_REG, 1)


def driver_forward_seg(forward_path, idx):
    """读 forward.bin 的 idx 段 -> bytes (MATMUL...+SYNC_HALT)。"""
    data = open(forward_path, "rb").read()
    assert data[:4] == FORWARD_MAGIC, f"bad forward magic: {data[:4]}"
    n_idx = struct.unpack("<I", data[4:8])[0]
    assert idx < n_idx, f"idx {idx} >= n_idx {n_idx}"
    off = struct.unpack(f"<{n_idx}I", data[8:8 + n_idx * 4])[idx]
    length = struct.unpack(f"<{n_idx}I", data[8 + n_idx * 4:8 + 2 * n_idx * 4])[idx]
    base = 8 + 2 * n_idx * 4
    return data[base + off:base + off + length]


def driver_launch(sim, forward_path, idx, x_int8_1d, N, K):
    """模拟 cim_launch_<idx> 单 token: MMIO 驱动 Forward, 返回 (acc, cim_cycle)。"""
    k_tiles = math.ceil(K / TILE)
    n_tiles = math.ceil(N / TILE)
    Kp = k_tiles * TILE
    xpad = np.zeros(Kp, dtype=np.int8)
    xpad[:K] = np.asarray(x_int8_1d, dtype=np.int8)
    seg = driver_forward_seg(forward_path, idx)
    sim._shm_write(INSTR_BASE * PAGE, np.frombuffer(seg, dtype=np.uint8))
    for kb in range(k_tiles):
        sim._shm_write((A_PAGE_BASE + kb) * PAGE,
                       xpad[kb * TILE:(kb + 1) * TILE].astype(np.uint8))
    sim.mmio_reg_write(sim.DOORBELL_REG, INSTR_BASE * PAGE)   # 门铃 (异步, _run 清 page_busy)
    sim.wait_irq()                                          # poll IRQ
    cim_cycle = sim.controller.cycle                        # 本次 Forward cycle
    acc = np.zeros(N, dtype=np.int32)
    for nb in range(n_tiles):
        vec = sim._shm_read((PSUM_PAGE_BASE + nb) * PAGE, PAGE).view(np.int32)
        s = nb * TILE
        e = min(s + TILE, N)
        acc[s:e] = vec[:e - s]
    sim.mmio_reg_write(sim.INT_CLEAR_REG, 1)
    return acc, cim_cycle


def _self_test():
    """Python driver (MMIO) 驱动纯硬件: Preload + Forward, 数值 + 时序统计。"""
    import json
    partition = "checkpoints/bitnet_ternary_partition.json"
    weights = "checkpoints/bitnet_ternary_weights.bin"
    preload = "cim_compiler/cimres/checkpoints/preload.bin"
    forward = "cim_compiler/cimres/checkpoints/forward.bin"

    sim = HwCimSimulator()
    driver_preload(sim, preload)
    pre_stats = sim.stats_snapshot()
    print(f"[Hw] preload (MMIO driver): {len(sim.macros.macro)} Macro, "
          f"cim_cycle={pre_stats['cim_cycle']}, mmio_cycle={pre_stats['mmio_cycle']}", file=sys.stderr)

    part = json.load(open(partition))
    idx2name = {blk["idx"]: blk["bitlinear_name"] for blk in part["cim_blocks"]}
    weights_list = read_weight_blob(weights)
    wmap = {_norm(w.name): w for w in weights_list}
    w_ternary = {n: unpack_2bit_np(np.frombuffer(wmap[n].packed, dtype=np.uint8).reshape(wmap[n].N, wmap[n].K // 4))
                 for n in idx2name.values()}

    rng = np.random.default_rng(0)
    max_diff = 0
    n = 0
    total_cim_cycle = 0
    serial_cycle = 0   # 串行估算 (k_tiles*n_tiles*(T_DISPATCH+T_MATMUL+T_WB))
    for idx in sorted(idx2name):
        name = idx2name[idx]
        we = wmap[name]
        N, K = we.N, we.K
        n_tiles = math.ceil(N / TILE)
        k_tiles = math.ceil(K / TILE)
        for M in (1, 3):
            x = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
            acc_sim = np.zeros((M, N), dtype=np.int32)
            cyc = 0
            for m in range(M):                       # 动态 M 循环
                a, c = driver_launch(sim, forward, idx, x[m], N, K)
                acc_sim[m] = a
                cyc += c
            acc_ref = (x.astype(np.int32) @ w_ternary[name].astype(np.int32).T).astype(np.int32)
            diff = int(np.max(np.abs(acc_sim.astype(np.int64) - acc_ref.astype(np.int64))))
            max_diff = max(max_diff, diff)
            n += 1
            serial = k_tiles * n_tiles * (T_DISPATCH + T_MATMUL + T_WB)   # 串行 (无并行)
            serial_cycle += serial * M
            total_cim_cycle += cyc
            if idx < 3 or diff != 0:
                print(f"  [idx={idx:2d}] {name} N={N} K={K}: M=1,3 diff={diff}, "
                      f"单token cycle={cyc // M}, 串行={serial}, 并行度={serial / (cyc / M):.2f}x "
                      f"{'OK' if diff == 0 else 'FAIL'}", file=sys.stderr)
    speedup = serial_cycle / total_cim_cycle if total_cim_cycle else 0
    print(f"[Hw] {n} 次 driver_launch (37 BitLinear × M=1,3), max_diff={max_diff}", file=sys.stderr)
    print(f"[Hw] 时序: 总 cim_cycle={total_cim_cycle}, 串行估算={serial_cycle}, "
          f"整体并行度={speedup:.2f}x {'PASS ✓' if max_diff == 0 else 'FAIL ✗'}", file=sys.stderr)
    return max_diff


if __name__ == "__main__":
    sys.exit(0 if _self_test() == 0 else 1)
