#!/usr/bin/env python3
"""CIM 架构级 PPA 估算 (28nm @ 1GHz) - 计算耗时 / 功耗能效 / 面积。

三件套:
  PPAConfig       - 工艺/频率/能耗/面积参数表 (28nm@1GHz 估算, 可配置)
  ActivityTracker - 仿真过程活动因子统计 (嵌入 hw_simulator, dispatch/mmio 接入)
  PPAEstimator    - 仿真结束算 PPA (cycle->ns / 动态+静态功耗 / 能效 / 面积)

定位: 架构级估算 (±30~50%), 非 RTL/后仿级。计算耗时复用 hw_simulator 的 busy_until
并行模型 (给定 T_MATMUL 单 macro GEMV 耗时, cim_cycle 即真实并行耗时), 只加频率转 ns。
功耗需活动因子 (ActivityTracker), 面积纯资源数 × 工艺参数。
"""
import os
import sys
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
CIM_COMPILER = os.path.dirname(HERE)
if CIM_COMPILER not in sys.path:
    sys.path.insert(0, CIM_COMPILER)

from cim_compiler.cimres.hw_config import SHARED_SIZE, TILE   # 1MB 共享缓存 + Macro 64×64 (单一事实源)

# Macro 一次 matmul = int8[TILE] × ternary[TILE,TILE] -> int32[TILE] = 4096 MAC
MAC_PER_MATMUL = TILE * TILE
A_PAGE_BYTES = TILE        # int8[TILE] 激活输入 (K 维 tile)
PSUM_BYTES = TILE * 4      # int32[TILE] 累加输出 (N 维 tile)
TILE_BYTES = TILE * TILE // 4   # 2bit packed TILE×TILE 权重 tile (PROG_WGT)


@dataclass
class PPAConfig:
    """28nm @ 1GHz, RRAM 存算一体 Macro + 4bit ADC (CIM 物理层估算; 文献量级, 非 PDK 实测)。"""
    # 工艺 + 频率
    tech_node_nm: int = 28
    clock_freq_ghz: float = 1.0       # 1 GHz -> cycle = 1 ns
    # CIM 模拟 MAC 能耗 (per MAC): DAC(输入电压) + RRAM 存储单元电流(KCL 求和) + 4bit ADC(量化)
    # 三值权重省乘法器 + 4bit ADC 低位宽, 比数字乘法器省 (文献 ~0.3-1 pJ)
    e_mac_pj: float = 0.5
    # RRAM 非易失写入 (编程电导, per cell, Preload 一次性)
    # 文献 10 pJ-10 nJ/cell, 取中值 100 pJ (可配置; 高能 RRAM 可到 nJ, 占比升)
    e_prog_pj_cell: float = 100.0
    # SharedCache 是 SRAM (数字缓存, 非 RRAM)
    e_sram_read_pj_byte: float = 2.0
    e_sram_write_pj_byte: float = 2.0
    e_bus_pj_byte: float = 1.5        # 总线/互连 (CPU<->CIM MMIO 传输)
    e_reg_pj: float = 0.5             # 寄存器 MMIO 访问
    # 漏电 (mW): RRAM 非易失近 0 + 外围 ADC/DAC 漏电
    leakage_per_macro_mw: float = 0.05
    leakage_per_mb_sram_mw: float = 50.0
    # 面积 (um^2, 28nm): RRAM 阵列密度高(小) + ADC/DAC 占面积(大)
    area_macro_um2: float = 2000      # 单 64×64 RRAM Macro + ADC/DAC 外围
    area_sram_um2_byte: float = 0.6   # SRAM 密度 (SharedCache)
    area_logic_gate: int = 50000      # 控制器/总线/仲裁门数估算
    area_gate_um2: float = 0.4        # 标准单元 (NAND2 equiv)
    util: float = 0.7                 # 利用率


class ActivityTracker:
    """仿真活动因子统计 (dispatch/mmio 接入, 零语义改动, 只加计数)。"""
    def __init__(self):
        self.n_mac = 0                # 总 MAC 数
        self.sram_read_bytes = 0      # SRAM 读 (tile + A_PAGE)
        self.sram_write_bytes = 0     # SRAM 写 (PSUM)
        self.bus_bytes = 0            # 总线传输 (CPU<->CIM MMIO shm)
        self.reg_accesses = 0         # 寄存器访问
        self.n_prog = 0               # PROG_WGT 次数
        self.n_prog_cell = 0          # RRAM 非易失写入 cell 数 (PROG_WGT × 4096)
        self.n_matmul = 0             # MATMUL 次数

    def record_prog_wgt(self):
        """PROG_WGT: ① 读 1024B tile 从 SharedCache (SRAM 读) + ② 编程 RRAM 4096 cell (非易失写入)。"""
        self.sram_read_bytes += TILE_BYTES
        self.n_prog += 1
        self.n_prog_cell += MAC_PER_MATMUL   # 64×64 RRAM 阵列 = 4096 cell/tile

    def record_matmul(self):
        """MATMUL: 读 A_PAGE int8[64] + 4096 MAC + 写 PSUM int32[64]。"""
        self.n_mac += MAC_PER_MATMUL
        self.sram_read_bytes += A_PAGE_BYTES
        self.sram_write_bytes += PSUM_BYTES
        self.n_matmul += 1

    def record_bus(self, n):
        """shm 传输 n 字节 (CPU<->CIM)。"""
        self.bus_bytes += n

    def record_reg(self):
        """寄存器 MMIO 访问。"""
        self.reg_accesses += 1


class PPAEstimator:
    """PPA 计算 (耗时/功耗/能效/面积)。仿真结束 stats_snapshot 调。"""
    def __init__(self, cfg, tracker, cim_cycle, mmio_cycle, n_macro):
        self.cfg = cfg
        self.tr = tracker
        self.cim_cycle = cim_cycle
        self.mmio_cycle = mmio_cycle
        self.n_macro = n_macro

    def estimate(self):
        cfg = self.cfg
        tr = self.tr
        period_ns = 1.0 / cfg.clock_freq_ghz            # 1GHz -> 1ns/cycle (周期=1/f)
        total_cycles = self.cim_cycle + self.mmio_cycle
        time_ns = total_cycles * period_ns
        time_us = time_ns / 1000.0

        # 动态能耗 (pJ) -> 功率 (pJ/ns = mW)
        # AOT 共享内存: CPU<->CIM 传输不经 server 回调 (bus_bytes=0), 从 CIM 缓存访问推算
        # (同一数据双向: CPU 写 A_PAGE/tile=CIM 读, CPU 读 PSUM=CIM 写)
        bus_bytes_eff = tr.bus_bytes if tr.bus_bytes > 0 else (tr.sram_read_bytes + tr.sram_write_bytes)
        e_prog_pj = tr.n_prog_cell * cfg.e_prog_pj_cell        # Preload: RRAM 非易失写入 (一次性)
        e_dyn_pj = (tr.n_mac * cfg.e_mac_pj
                    + e_prog_pj
                    + tr.sram_read_bytes * cfg.e_sram_read_pj_byte
                    + tr.sram_write_bytes * cfg.e_sram_write_pj_byte
                    + bus_bytes_eff * cfg.e_bus_pj_byte
                    + tr.reg_accesses * cfg.e_reg_pj)
        e_fwd_pj = e_dyn_pj - e_prog_pj                        # Forward: MAC+sram+bus+reg (稳态, 不含 Preload)
        p_dyn_mw = e_fwd_pj / time_ns if time_ns > 0 else 0.0  # Forward 稳态功率 (Preload 单独计, 不混入)

        # 静态功耗 (mW): Macro 漏电 + SRAM 漏电
        sram_mb = SHARED_SIZE / (1 << 20)
        p_leak_mw = (self.n_macro * cfg.leakage_per_macro_mw
                     + sram_mb * cfg.leakage_per_mb_sram_mw)

        # 总功耗 (Forward 稳态) + 能效
        p_total_mw = p_dyn_mw + p_leak_mw
        leak_j = p_leak_mw * 1e-3 * time_ns * 1e-9
        energy_fwd_j = e_fwd_pj * 1e-12 + leak_j               # 稳态 (不含 Preload)
        energy_amort_j = e_dyn_pj * 1e-12 + leak_j             # amortized (含 Preload 一次性分摊到本次)
        gmacs_w = (tr.n_mac / energy_fwd_j / 1e9) if energy_fwd_j > 0 else 0.0    # 稳态能效
        gmacs_w_amort = (tr.n_mac / energy_amort_j / 1e9) if energy_amort_j > 0 else 0.0
        tops_w = 2 * gmacs_w
        tops_w_amort = 2 * gmacs_w_amort

        # 面积 (um^2 -> mm^2)
        a_macro = self.n_macro * cfg.area_macro_um2
        a_sram = SHARED_SIZE * cfg.area_sram_um2_byte
        a_logic = cfg.area_logic_gate * cfg.area_gate_um2
        area_mm2 = (a_macro + a_sram + a_logic) / cfg.util / 1e6

        return {
            "tech_nm": cfg.tech_node_nm, "freq_ghz": cfg.clock_freq_ghz,
            "total_cycles": total_cycles, "cim_cycle": self.cim_cycle, "mmio_cycle": self.mmio_cycle,
            "time_ns": time_ns, "time_us": time_us,
            "p_dyn_mw": p_dyn_mw, "p_leak_mw": p_leak_mw, "p_total_mw": p_total_mw,
            "e_dyn_pj": e_dyn_pj, "e_prog_pj": e_prog_pj, "e_fwd_pj": e_fwd_pj,
            "gmacs_w": gmacs_w, "tops_w": tops_w, "gmacs_w_amort": gmacs_w_amort, "tops_w_amort": tops_w_amort,
            "area_mm2": area_mm2, "a_macro_mm2": a_macro / 1e6,
            "a_sram_mm2": a_sram / 1e6, "a_logic_mm2": a_logic / 1e6,
            "util": cfg.util, "n_macro": self.n_macro,
            "n_mac": tr.n_mac, "n_matmul": tr.n_matmul, "n_prog": tr.n_prog, "n_prog_cell": tr.n_prog_cell,
            "sram_read_kb": tr.sram_read_bytes / 1024,
            "sram_write_kb": tr.sram_write_bytes / 1024,
            "bus_kb": tr.bus_bytes / 1024,
            "bus_kb_eff": bus_bytes_eff / 1024,
            "reg_accesses": tr.reg_accesses,
        }


def format_ppa_report(r):
    """格式化 PPA 报告 (cim_sim 跑完打印)。"""
    return (
        f"[PPA] 工艺 {r['tech_nm']}nm @ {r['freq_ghz']:.1f} GHz (架构级估算, ±30~50%)\n"
        f"  计算耗时: {r['total_cycles']} cycle = {r['time_us']:.3f} us "
        f"(cim={r['cim_cycle']} + mmio={r['mmio_cycle']})\n"
        f"  功耗:     Forward 稳态 {r['p_dyn_mw']:.2f} mW + 静态 {r['p_leak_mw']:.2f} mW "
        f"= {r['p_total_mw']:.2f} mW (Preload 一次性 {r['e_prog_pj']/1e6:.2f} μJ 单独计)\n"
        f"  能效:     稳态 {r['gmacs_w']:.2f} GMACs/W ({r['tops_w']:.2f} TOPS/W) | "
        f"含Preload {r['gmacs_w_amort']:.2f} GMACs/W ({r['tops_w_amort']:.2f} TOPS/W)\n"
        f"  面积:     Macro {r['a_macro_mm2']:.2f} + SRAM {r['a_sram_mm2']:.2f} "
        f"+ 逻辑 {r['a_logic_mm2']:.3f} = {r['area_mm2']:.2f} mm² (util {r['util']})\n"
        f"  活动:     MAC={r['n_mac']/1e3:.1f}k (matmul={r['n_matmul']}), "
        f"RRAM写={r['n_prog_cell']/1e6:.2f}M cell (prog={r['n_prog']}, {r['e_prog_pj']/1e6:.2f} μJ), "
        f"SRAM读={r['sram_read_kb']:.1f}KB, 写={r['sram_write_kb']:.1f}KB, "
        f"总线={r['bus_kb_eff']:.1f}KB(推算), reg={r['reg_accesses']}, Macro={r['n_macro']}"
    )
