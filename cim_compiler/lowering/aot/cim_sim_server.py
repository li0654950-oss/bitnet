#!/usr/bin/env python3
"""CIM 仿真器 IPC server: HwCimSimulator + 共享内存 (shm 数据) + unix socket (reg 控制)。

cim_sim (AOT) 经共享内存直接读写 CIM 共享缓存 (shm_*, 零拷贝零往返, 替代 socket),
经 socket 传 reg_* 控制信号 (门铃/IRQ/INT_CLEAR, socket 作同步点保证可见性)。

启动: python cim_sim_server.py
然后: ./cim_sim --prompt "ROMEO:" --n 60 ...
"""
import os
import sys
import socket
import struct
import argparse
from multiprocessing.shared_memory import SharedMemory

HERE = os.path.dirname(os.path.abspath(__file__))    # .../cim_compiler/lowering/aot
LOWERING = os.path.dirname(HERE)                     # .../cim_compiler/lowering
CIM_COMPILER = os.path.dirname(LOWERING)             # .../cim_compiler
REPO = os.path.dirname(CIM_COMPILER)                 # repo root
EXPORT_DIR = os.path.join(CIM_COMPILER, "export")    # cim_op 所在 (inference_model import)
for _p in (REPO, EXPORT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from cim_compiler.cimres.hw_simulator import HwCimSimulator, SHARED_SIZE
from cim_compiler.cimres.ppa_config import format_ppa_report   # PPA 报告格式化
from cim_compiler.cimres import hw_config      # S4+: LAYOUT_MAP + T_ROUT_PER_HOP (layout_config.json 加载)

DEFAULT_SOCKET = "/tmp/cim_sim.sock"
SHM_NAME = "cim_cache"   # 对齐 cim_shm.h CIM_SHM_NAME ("/cim_cache", Python SharedMemory 内部加 /)


def recvn(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--socket", default=DEFAULT_SOCKET)
    args = p.parse_args()

    # S4+: 加载 autotuner 序列化的 layout_config.json (LAYOUT_MAP + T_ROUT_PER_HOP)
    # 使实际仿真 cycle 反映 layout 优化 (hotspot vs linear); 不存在则 spec 广播总线基线 (T_ROUT=0)
    import json as _json
    _layout_cfg = os.path.join(CIM_COMPILER, "cimres", "checkpoints", "layout_config.json")
    if os.path.exists(_layout_cfg):
        with open(_layout_cfg) as _f:
            _cfg = _json.load(_f)
        hw_config.LAYOUT_MAP = {int(k): tuple(v) for k, v in _cfg["layout_map"].items()}
        hw_config.T_ROUT_PER_HOP = _cfg["t_rout_per_hop"]
        print(f"[server] 加载 layout_config.json: strategy={_cfg['strategy']} "
              f"T_ROUT={_cfg['t_rout_per_hop']} ({len(hw_config.LAYOUT_MAP)} Macro)", file=sys.stderr)
    else:
        print(f"[server] 无 layout_config.json, 用 spec 广播总线基线 (T_ROUT=0)", file=sys.stderr)

    # 创建共享内存 (承载 CIM 共享缓存, C/Python 共享; 残留则打开已有)
    try:
        shm = SharedMemory(name=SHM_NAME, create=True, size=SHARED_SIZE)
        print(f"[server] 共享内存 '{SHM_NAME}' ({SHARED_SIZE}B) 已创建", file=sys.stderr)
    except FileExistsError:
        shm = SharedMemory(name=SHM_NAME, create=False)
        print(f"[server] 共享内存 '{SHM_NAME}' 已存在 (复用)", file=sys.stderr)

    if os.path.exists(args.socket):
        os.unlink(args.socket)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.socket)
    srv.listen(1)
    print(f"[server] 监听 {args.socket}, 等待 cim_sim (Ctrl-C 退出)...", file=sys.stderr)
    try:
        while True:
            conn, _ = srv.accept()
            sim = HwCimSimulator(shm_buf=shm.buf)   # cache.data backed by 共享内存
            sim.cache.data[:] = 0                    # 清零 (前次连接残留)
            print(f"[server] cim_sim 已连接 (shm 数据走共享内存, reg 走 socket)", file=sys.stderr)
            n_ops = 0
            while True:
                hdr = recvn(conn, 1)
                if not hdr:
                    break
                op = hdr[0]
                if op == 3:                                  # reg_write(reg, val) [shm_* 不经 socket]
                    reg, val = struct.unpack("<qq", recvn(conn, 16))
                    sim.mmio_reg_write(reg, val)
                    conn.sendall(b"\x01")
                elif op == 4:                                # reg_read(reg) -> val(i32)
                    reg = struct.unpack("<q", recvn(conn, 8))[0]
                    val = sim.mmio_reg_read(reg)
                    conn.sendall(struct.pack("<i", val))
                else:
                    print(f"[server] 未知 op={op} (仅 reg_*=3/4; shm_* 走共享内存), 断开",
                          file=sys.stderr)
                    break
                n_ops += 1
            conn.close()
            st = sim.stats_snapshot()
            print(f"[server] 连接结束, {n_ops} reg-socket 往返 (shm 走共享内存零往返), "
                  f"cim_cycle={st['cim_cycle']}, mmio_cycle={st['mmio_cycle']}", file=sys.stderr)
            print(format_ppa_report(st["ppa"]), file=sys.stderr)
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()
        if os.path.exists(args.socket):
            os.unlink(args.socket)
        try:
            shm.unlink()   # 删除共享内存名 (mmap 进程退出自动释放; numpy view 持有不 close)
        except Exception as e:
            print(f"[server] shm unlink: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
