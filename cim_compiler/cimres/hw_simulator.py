#!/usr/bin/env python3
"""完整硬件级 CIM 指令仿真器 (对应 cim_mlp.md §2-4 硬件架构)。

组件 (§2.2/2.3):
  SharedCache   - 1MB 共享缓存, PAGE 寻址, 三区 (覆盖/指令/累加), 两阶段 (preload/forward)
  MacroArray    - 4096 个 64×64 三值寄存器, load(2bit)/matmul
  Controller    - 门铃寄存器 + 取指 + 解码 + 分发 + IRQ_STATUS/IRQ_CIM
  BusDispatcher - 多宏分发与总线驱动 (Dest_ID 路由)
  UpstreamArbiter - 上行总线仲裁器 (int32 写回累加区 RMW)

指令取指 (§3, 闭合 PROG_WGT gap):
  PROG_WGT  - 从 b_page_start 读 4 PAGE 2bit -> 解包 -> Macro[dest] (权重经覆盖区暂存)
  MATMUL    - 从 a_page 读 64 int8 × Macro[dest] -> int32, 按 accum 写 psum_page (RMW)
  SYNC_HALT - 等 Macro完成 + IRQ

两阶段 (§4.1/4.6):
  Preload Phase - CPU 写 tile 到覆盖区 + PROG_WGT 指令 + 门铃 (分批 + 批间 IRQ)
  Forward Phase - CPU 写 x_int8 到 A_PAGE + MATMUL 指令 + 门铃 + IRQ + 读 PSUM_PAGE acc

系统级 (保持架构一致, 方案 A):
  simulate(idx, x, w) 动态 M 循环 forward_bitlinear; cim_stub 回调对接 L6 JIT

替换 sys_simulator.py + simulator.py。
运行 (自测): nanogpt-gpu python cim_compiler/cimres/hw_simulator.py
"""
import os
import sys
import json
import math
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CIM_COMPILER = os.path.dirname(HERE)
REPO = os.path.dirname(CIM_COMPILER)
EXPORT_DIR = os.path.join(CIM_COMPILER, "export")
for _p in (REPO, EXPORT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from torch_mlir import ir
from cim_compiler.cimres.dialect import register_cimres, attr_i32, attr_bool, attr_str
from cim_compiler.cimres.place import A_PAGE_BASE, PSUM_PAGE_BASE
from cim_compiler.cimres.emit_instr import encode, OP_PROG_WGT, OP_MATMUL, OP_SYNC_HALT
from cim_compiler.export.weight_blob import read_weight_blob

TILE = 64
SHARED_SIZE = 1 << 20       # 1MB
PAGE = 256
# 共享缓存三区 (§4.6)
OVERWRITE_BASE = 0x000      # 覆盖区 0x000~0xBEF (3056 PAGE)
INSTR_BASE = 0xBF0          # 指令区 0xBF0~0xBFF (16 PAGE, 4KB)
ACCUM_BASE = 0xC00          # 累加区 0xC00~0xFFF (1024 PAGE, int32)
INSTR_CAPACITY = 16 * PAGE // 6   # 指令区 48-bit 条数 (~682)
PRELOAD_BATCH = INSTR_CAPACITY - 1   # 批大小受指令区约束 (留 1 条 SYNC_HALT)


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


# ===================== 硬件组件 (§2.2/2.3) =====================
class SharedCache:
    """1MB 共享缓存, PAGE 寻址, 三区, 两阶段 (§2.1/§4.6)。"""
    def __init__(self):
        self.data = np.zeros(SHARED_SIZE, dtype=np.uint8)
        self.data32 = self.data.view(np.int32)
        self.phase = "forward"   # "preload" / "forward" (语义切换, 物理同缓存)

    def write_bytes(self, byte_addr, b):
        v = np.frombuffer(b, dtype=np.uint8) if isinstance(b, (bytes, bytearray)) else b.astype(np.uint8)
        self.data[byte_addr:byte_addr + len(v)] = v

    def read_bytes(self, byte_addr, n):
        return bytes(self.data[byte_addr:byte_addr + n])

    # int8 特征 (A_PAGE, 覆盖区)
    def write_int8_vec(self, page, vec):
        addr = page * PAGE
        self.data[addr:addr + len(vec)] = np.asarray(vec, dtype=np.int8).astype(np.uint8)

    def read_int8_vec(self, page, n=TILE):
        addr = page * PAGE
        return self.data[addr:addr + n].astype(np.int8).astype(np.int32)

    # int32 部分和 (PSUM_PAGE, 累加区)
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

    # 指令区 (48-bit 字, 6 字节, §4.6 指令区)
    def write_instr(self, byte_offset, words):
        addr = INSTR_BASE * PAGE + byte_offset
        buf = bytearray()
        for w in words:
            buf += int(w).to_bytes(6, "little")
        self.data[addr:addr + len(buf)] = np.frombuffer(bytes(buf), dtype=np.uint8)

    def read_instr(self, byte_addr):
        b = self.data[byte_addr:byte_addr + 6]
        return int.from_bytes(bytes(b) + b"\x00\x00", "little")


class MacroArray:
    """4096 个 64×64 三值寄存器 (§2.2)。"""
    def __init__(self):
        self.macro = {}   # dest_id -> [64,64] int8

    def load(self, dest_id, tile_2bit_1024B):
        """从 1024B 2bit (4 PAGE) 解包 64×64 三值 -> Macro[dest] (§3.2)。"""
        packed = np.frombuffer(tile_2bit_1024B, dtype=np.uint8).reshape(TILE, TILE // 4)
        self.macro[dest_id] = unpack_2bit_np(packed)

    def matmul(self, dest_id, x_int8_64):
        """矩阵向量乘 int8[64] × 三值[64,64] -> int32[64] (§3.3)。"""
        return self.macro[dest_id].astype(np.int32) @ x_int8_64.astype(np.int32)


class UpstreamArbiter:
    """上行总线仲裁器: int32 写回累加区 RMW (§2.2)。"""
    def __init__(self, cache):
        self.cache = cache

    def writeback(self, psum_page, y_int32, accum):
        self.cache.rmw_int32(psum_page, y_int32, accum)


class BusDispatcher:
    """多宏分发与总线驱动: Dest_ID 路由 (§2.2/§4.7.7)。"""
    def __init__(self, macros, cache, arbiter):
        self.macros = macros
        self.cache = cache
        self.arbiter = arbiter

    def dispatch_prog_wgt(self, dest_id, b_page_start):
        tile = self.cache.read_bytes(b_page_start * PAGE, 1024)
        self.macros.load(dest_id, tile)

    def dispatch_matmul(self, dest_id, a_page, psum_page, accum):
        x = self.cache.read_int8_vec(a_page, TILE)
        y = self.macros.matmul(dest_id, x)
        self.arbiter.writeback(psum_page, y, accum)


class Controller:
    """控制器: 门铃 + 取指 + 解码 + 分发 + IRQ (§2.2/§2.3)。"""
    IDLE, BUSY, DONE, ERROR = 0, 1, 2, 3

    def __init__(self, cache, dispatcher):
        self.cache = cache
        self.dispatcher = dispatcher
        self.doorbell_reg = 0
        self.irq_status = self.IDLE
        self.irq_flag = False

    def doorbell(self, instr_byte_addr):
        """CPU 写门铃: 设指令区起始地址, 唤醒分发器取指 (§2.3)。"""
        self.doorbell_reg = instr_byte_addr
        self.irq_status = self.BUSY
        self.irq_flag = False
        self._run()

    def _run(self):
        """顺序取指执行至 SYNC_HALT (§2.3 doorbell 语义)。"""
        addr = self.doorbell_reg
        while True:
            w = self.cache.read_instr(addr)
            addr += 6
            op = (w >> 45) & 0x7
            dest = (w >> 33) & 0xFFF
            p1 = (w >> 21) & 0xFFF
            p2 = (w >> 9) & 0xFFF
            accum = (w >> 8) & 1
            if op == OP_PROG_WGT:
                self.dispatcher.dispatch_prog_wgt(dest, p1)
            elif op == OP_MATMUL:
                self.dispatcher.dispatch_matmul(dest, p1, p2, accum)
            elif op == OP_SYNC_HALT:
                break
            else:
                self.irq_status = self.ERROR
                return
        self.irq_status = self.DONE
        self.irq_flag = True   # IRQ_CIM 置位

    def wait_irq(self):
        """CPU 等待 IRQ (§4.7)。"""
        assert self.irq_status == self.DONE, f"IRQ 未完成: status={self.irq_status}"
        self.irq_flag = False

    def int_clear(self):
        self.irq_status = self.IDLE
        self.irq_flag = False


# ===================== HwCimSimulator (整合 + CPU 驱动) =====================
class HwCimSimulator:
    """完整硬件级 CIM 仿真器: 系统级 CPU-CIM 协同 (§2-4)。"""

    def __init__(self, placed_ir_path, weights_path, partition_path):
        self.cache = SharedCache()
        self.macros = MacroArray()
        self.arbiter = UpstreamArbiter(self.cache)
        self.dispatcher = BusDispatcher(self.macros, self.cache, self.arbiter)
        self.controller = Controller(self.cache, self.dispatcher)
        self.idx2name = {}
        self.meta = {}          # name -> (N, K, n_tiles, k_tiles, dests)
        self.instr_map = {}     # name -> [48-bit MATMUL 指令字] (forward)
        self.tile_2bit = {}     # dest_id -> 1024B 2bit (preload 用)
        self.w_ternary_map = {}  # name -> [N,K] int8 (验证用)
        self._loaded = False
        self._load(placed_ir_path, weights_path, partition_path)

    def _load(self, placed_ir_path, weights_path, partition_path):
        """建映射 + 存 tile 2bit + forward 指令 (不预载 Macro, preload_phase 走 PROG_WGT)。"""
        part = json.load(open(partition_path))
        for blk in part["cim_blocks"]:
            self.idx2name[blk["idx"]] = blk["bitlinear_name"]
        weights = read_weight_blob(weights_path)
        wmap = {_norm(w.name): w for w in weights}
        ctx = ir.Context()
        ctx.load_all_available_dialects()
        register_cimres(ctx)
        with ctx:
            mod = ir.Module.parse(open(placed_ir_path).read(), ctx)
        for op in list(mod.body):
            if op.operation.name != "func.func":
                continue
            blk = op.regions[0].blocks[0]
            mats = [i for i in list(blk.operations) if i.operation.name == "cimres.macro_matmul"]
            if not mats:
                continue
            name = attr_str(mats[0], "bitlinear_name")
            we = wmap[name]
            N, K = we.N, we.K
            n_tiles = math.ceil(N / TILE)
            k_tiles = math.ceil(K / TILE)
            Np, Kp = n_tiles * TILE, k_tiles * TILE
            packed = np.frombuffer(we.packed, dtype=np.uint8).reshape(N, K // 4)
            self.w_ternary_map[name] = unpack_2bit_np(packed)
            # tile 2bit (pad 到 Np×Kp/4, 切 [64,16]=1024B per tile)
            packed_pad = np.zeros((Np, Kp // 4), dtype=np.uint8)
            packed_pad[:N, :K // 4] = packed
            instrs = []
            dests = []
            for m in mats:
                d = attr_i32(m, "dest_id")
                nb = attr_i32(m, "n_blk")
                kb = attr_i32(m, "k_blk")
                tile_packed = packed_pad[nb * TILE:(nb + 1) * TILE, kb * (TILE // 4):(kb + 1) * (TILE // 4)]
                self.tile_2bit[d] = tile_packed.tobytes()   # 1024B 2bit
                a = attr_i32(m, "a_page")
                p = attr_i32(m, "psum_page")
                acc = attr_bool(m, "accum")
                instrs.append(encode(OP_MATMUL, dest_id=d, page1=a, page2=p, accum=1 if acc else 0))
                dests.append(d)
            instrs.append(encode(OP_SYNC_HALT))
            self.instr_map[name] = instrs
            self.meta[name] = (N, K, n_tiles, k_tiles, dests)

    def preload_phase(self):
        """Preload Phase: CPU 写 tile 到覆盖区 + PROG_WGT 指令 + 门铃 (分批, §4.2/4.6)。
        权重经覆盖区暂存, 控制器从 b_page_start 取指读 2bit 解包 -> Macro (闭合 PROG_WGT gap)。"""
        self.cache.phase = "preload"
        dests_sorted = sorted(self.tile_2bit.keys())
        for batch_start in range(0, len(dests_sorted), PRELOAD_BATCH):
            batch = dests_sorted[batch_start:batch_start + PRELOAD_BATCH]
            instrs = []
            for i, d in enumerate(batch):
                b_page = OVERWRITE_BASE + i * 4          # 覆盖区, 每 tile 4 PAGE (1024B)
                self.cache.write_bytes(b_page * PAGE, self.tile_2bit[d])
                instrs.append(encode(OP_PROG_WGT, dest_id=d, page1=b_page))
            instrs.append(encode(OP_SYNC_HALT))
            self.cache.write_instr(0, instrs)
            self.controller.doorbell(INSTR_BASE * PAGE)   # 门铃 -> 控制器取指
            self.controller.wait_irq()
            self.controller.int_clear()
        self._loaded = True

    def forward_bitlinear(self, name, x_int8_1d):
        """Forward Phase (单 token): 写 A_PAGE + MATMUL 指令 + 门铃 + IRQ + 读 acc (§4.3/4.7)。"""
        assert self._loaded, "须先 preload_phase()"
        N, K, n_tiles, k_tiles, _ = self.meta[name]
        Kp = k_tiles * TILE
        xpad = np.zeros(Kp, dtype=np.int8)
        xpad[:K] = np.asarray(x_int8_1d, dtype=np.int8)
        self.cache.phase = "forward"
        # CPU 写 A_PAGE (搬运 x_int8 -> 共享缓存, §4.7.2/3)
        for kb in range(k_tiles):
            self.cache.write_int8_vec(A_PAGE_BASE + kb, xpad[kb * TILE:(kb + 1) * TILE])
        # CPU 写 MATMUL 指令区 + SYNC_HALT, 门铃 -> 控制器取指
        self.cache.write_instr(0, self.instr_map[name])
        self.controller.doorbell(INSTR_BASE * PAGE)
        self.controller.wait_irq()
        self.controller.int_clear()
        # CPU 读 PSUM_PAGE acc (CIM->CPU, 取 valid 行去 pad)
        acc = np.zeros(N, dtype=np.int32)
        for nb in range(n_tiles):
            vec = self.cache.read_int32_vec(PSUM_PAGE_BASE + nb, TILE)
            s = nb * TILE
            e = min(s + TILE, N)
            acc[s:e] = vec[:e - s]
        return acc

    def simulate(self, idx, x_int8, w_packed=None):
        """@cim_launch_<idx> 系统级入口: 动态 M 循环 forward_bitlinear (方案 C, §4.7)。"""
        name = self.idx2name[idx]
        N = self.meta[name][0]
        x = np.ascontiguousarray(np.asarray(x_int8), dtype=np.int8)
        M, K = x.shape
        acc = np.zeros((M, N), dtype=np.int32)
        for m in range(M):               # 动态 M 循环 (seq_len 运行时吸收)
            acc[m] = self.forward_bitlinear(name, x[m])
        return acc


def _self_test():
    """H0-H2 自测: preload_phase (PROG_WGT 取指) + simulate (MATMUL 取指) acc 对齐参考。"""
    sim = HwCimSimulator(
        "cim_compiler/cimres/checkpoints/bitnet_ternary_cimres_placed.mlir",
        "checkpoints/bitnet_ternary_weights.bin",
        "checkpoints/bitnet_ternary_partition.json",
    )
    print(f"[Hw] {len(sim.idx2name)} BitLinear, {len(sim.tile_2bit)} tile 2bit", file=sys.stderr)
    sim.preload_phase()
    print(f"[Hw] preload_phase 完成 (PROG_WGT 取指, {len(sim.macros.macro)} Macro 预载)",
          file=sys.stderr)

    rng = np.random.default_rng(0)
    max_diff = 0
    n = 0
    for idx in sorted(sim.idx2name):
        name = sim.idx2name[idx]
        N, K = sim.meta[name][:2]
        for M in (1, 3):
            x = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
            acc_sim = sim.simulate(idx, x)
            acc_ref = (x.astype(np.int32) @ sim.w_ternary_map[name].astype(np.int32).T).astype(np.int32)
            diff = int(np.max(np.abs(acc_sim.astype(np.int64) - acc_ref.astype(np.int64))))
            max_diff = max(max_diff, diff)
            n += 1
        if idx < 3 or diff != 0:
            print(f"  [idx={idx:2d}] {name} N={N} K={K}: M=1,3 diff={diff} "
                  f"{'OK' if diff == 0 else 'FAIL'}", file=sys.stderr)
    print(f"[Hw] {n} 次 simulate (37 BitLinear × M=1,3), max_diff={max_diff} "
          f"{'PASS ✓' if max_diff == 0 else 'FAIL ✗'}", file=sys.stderr)
    return max_diff


if __name__ == "__main__":
    sys.exit(0 if _self_test() == 0 else 1)
