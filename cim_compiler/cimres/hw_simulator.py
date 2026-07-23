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
import sys
import time
import ctypes
import threading
import numpy as np

# CIM ASIC 硬件参数集中定义 (cim_compiler/cimres/hw_config.py, C/Python 镜像)
from cim_compiler.cimres.hw_config import *   # noqa: F401  TILE/PAGE/三区/INSTR_CAPACITY/PRELOAD_BATCH/REG_BASE_DEFAULT/...
from cim_compiler.cimres.ppa_config import PPAConfig, ActivityTracker, PPAEstimator  # 架构级 PPA 估算

# ===== cycle 开销 (§3 无规定, 估算默认值, 真实硬件可调) =====
T_FETCH    = 1    # 取指 (1 条 48-bit, §3.1)
T_DISPATCH = 2    # 广播总线 Dest_ID 路由 + 负载 (§2.2; 位宽见 §3.1)
T_PROG_WGT = 10   # 2bit tile 解包 + load (§3.2)
# T_MATMUL 来自 hw_config (=TILE, ADC 串行列扫每 cycle 1 列; §3.3), from import * 带入
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


# ===================== 硬件组件 (§2.2) =====================
class SharedCache:
    """1MB 共享缓存, PAGE 寻址, 三区 (§2.1/§4.6)。
    shm_buf 非空时 data backed by POSIX 共享内存 (C/Python 共享, IPC 优化, AOT 模式用)。"""
    def __init__(self, shm_buf=None):
        if shm_buf is not None:
            self.data = np.frombuffer(shm_buf, dtype=np.uint8, count=SHARED_SIZE)
        else:
            self.data = np.zeros(SHARED_SIZE, dtype=np.uint8)
        self.data32 = self.data.view(np.int32)

    def read_bytes(self, byte_addr, n):
        return bytes(self.data[byte_addr:byte_addr + n])

    def read_int8_vec(self, page, n=TILE):
        addr = page * PAGE
        return self.data[addr:addr + n].astype(np.int8).astype(np.int32)

    def rmw_int32(self, page, y_int32, accum):
        # 跨页 RMW: y_int32 可能跨多 PAGE (TILE>64, PSUM_PAGES_PER_NBLK>1; @TILE=64 单页)
        per_page = PAGE // I32_BYTES
        remaining = y_int32
        cur_page = page
        while len(remaining) > 0:
            a32 = cur_page * PAGE // I32_BYTES
            chunk = min(len(remaining), per_page)
            if not accum:
                self.data32[a32:a32 + chunk] = remaining[:chunk]
            else:
                self.data32[a32:a32 + chunk] = self.data32[a32:a32 + chunk] + remaining[:chunk]
            remaining = remaining[chunk:]
            cur_page += 1

    def read_int32_vec(self, page, n=TILE):
        # 跨页读 (TILE>64 跨多 PAGE; @TILE=64 单页)
        per_page = PAGE // I32_BYTES
        parts = []
        cur_page = page
        remaining = n
        while remaining > 0:
            a32 = cur_page * PAGE // I32_BYTES
            chunk = min(remaining, per_page)
            parts.append(self.data32[a32:a32 + chunk].copy())
            remaining -= chunk
            cur_page += 1
        return np.concatenate(parts)[:n]

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

    def load(self, dest_id, tile_2bit_packed, cycle):
        """PROG_WGT: 解包 2bit -> Macro[dest]. 返回完成 cycle (T_DISPATCH+T_PROG_WGT)。"""
        m = self.macro.get(dest_id)
        if m is None:
            m = Macro(); self.macro[dest_id] = m
        start = max(cycle, m.busy_until)                       # 同 Macro 串行
        packed = np.frombuffer(tile_2bit_packed, dtype=np.uint8).reshape(TILE, TILE // CODES_PER_BYTE)
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
    def __init__(self, macros, cache, arbiter, tracker=None):
        self.macros = macros
        self.cache = cache
        self.arbiter = arbiter
        self.tracker = tracker   # ActivityTracker (PPA 活动统计, 可选)

    def dispatch_prog_wgt(self, dest_id, b_page_start, cycle):
        tile = self.cache.read_bytes(b_page_start * PAGE, TILE_BYTES)
        if self.tracker: self.tracker.record_prog_wgt()
        return self.macros.load(dest_id, tile, cycle)         # 含 T_DISPATCH+T_PROG_WGT

    def dispatch_matmul(self, dest_id, a_page, psum_page, accum, cycle):
        x = self.cache.read_int8_vec(a_page, TILE)
        y, finish = self.macros.matmul(dest_id, x, cycle)     # 含 T_DISPATCH+T_MATMUL
        if self.tracker: self.tracker.record_matmul()
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
        self.total_cycle = 0    # 跨 doorbell 累计 (PPA 总 CIM 耗时)
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
                dest = ((w >> 33) & 0xFFF) | ((w & 0xF) << 12)   # Dest_ID 16b 非连续: [44:33]低12 + [3:0]高4
                p1 = (w >> 19) & PAGE_MASK   # page1 14 bit [32:19] (PAGE 派生后 PAGE 数可达 16384)
                p2 = (w >> 5) & PAGE_MASK    # page2 14 bit [18:5]
                accum = (w >> 4) & 1
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
            self.total_cycle += cycle   # 累计跨 doorbell (PPA 总 CIM 耗时)
            self.irq_status = DONE
        except Exception:
            self.irq_status = ERROR

    def int_clear(self):
        self.irq_status = IDLE


# ===================== HwCimSimulator (纯硬件 + MMIO + cycle 统计) =====================
class HwCimSimulator:
    """纯硬件 CIM 仿真器: MMIO 接口 + cycle 级时序统计。
    CPU (cim_stub.c) 经 MMIO 读写操作, 硬件门铃异步取指执行。"""

    def __init__(self, reg_base=REG_BASE_DEFAULT, shm_buf=None, ppa_cfg=None):
        sys.setswitchinterval(0.001)   # 1ms 线程切换 (门铃异步 poll 响应)
        self.cache = SharedCache(shm_buf)
        self.macros = MacroArray()
        self.arbiter = UpstreamArbiter(self.cache)
        self.tracker = ActivityTracker()                      # PPA 活动因子统计
        self.ppa_cfg = ppa_cfg or PPAConfig()
        self.dispatcher = BusDispatcher(self.macros, self.cache, self.arbiter, self.tracker)
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
        self.tracker.record_bus(len(v))

    def _shm_read(self, byte_addr, n):
        self.mmio_cycle += T_SHM
        self.tracker.record_bus(n)
        return self.cache.data[byte_addr:byte_addr + n].copy()

    # ---- MMIO 回调 (C 经 ctypes 调用) ----
    def mmio_shm_write(self, byte_addr, ptr, n):
        arr = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8)), (n,))
        self.cache.data[byte_addr:byte_addr + n] = arr
        self.mmio_cycle += T_SHM
        self.tracker.record_bus(n)

    def mmio_shm_read(self, byte_addr, ptr, n):
        arr = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8)), (n,))
        arr[:] = self.cache.data[byte_addr:byte_addr + n]
        self.mmio_cycle += T_SHM
        self.tracker.record_bus(n)

    def mmio_reg_write(self, reg, val):
        self.mmio_cycle += T_REG
        self.tracker.record_reg()
        if reg == self.DOORBELL_REG:
            self.controller.doorbell(val)      # 异步启动 _run
        elif reg == self.INT_CLEAR_REG:
            self.controller.int_clear()

    def mmio_reg_read(self, reg):
        self.mmio_cycle += T_REG
        self.tracker.record_reg()
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
        """返回时序统计 + PPA 估算 (CIM/MMIO cycle / Macro 数 / 功耗时能效面积)。"""
        ppa = PPAEstimator(self.ppa_cfg, self.tracker, self.controller.total_cycle,
                           self.mmio_cycle, len(self.macros.macro)).estimate()
        return {
            "cim_cycle": self.controller.total_cycle,
            "mmio_cycle": self.mmio_cycle,
            "n_macro": len(self.macros.macro),
            "ppa": ppa,
        }
