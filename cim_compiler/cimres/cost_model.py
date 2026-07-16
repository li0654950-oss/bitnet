#!/usr/bin/env python3
"""cimres 编译时 makespan 评估模型 (S1)。

复用 hw_simulator Controller._run 的 busy_until 时序逻辑 (T_FETCH/T_DISPATCH/
T_MATMUL/T_WB + macro_busy + page_busy + SYNC_HALT join), 但不执行数值只算 cycle。
遍历 placed IR 的 macro_matmul 序列模拟取指执行, 输出 per-func makespan + 总 forward makespan。

用途:
  - 编译时预估 forward makespan (不跑 hw_simulator)
  - 对照 hw_simulator 验证准确性 (S1 验收)
  - S2/S3 调度器选优的快速评估工具

单 BitLinear 调度空间分析 (8n×8k=64 tile, T_MATMUL=64):
  - k外n内 (当前 C1): ~126 cycle, 8n 并行起步早 (fetch 0-7), 接近下界
  - n外k内: ~154 cycle, n7 起步晚 (fetch 56), 更差
  - 下界: ~126 (8n 全并行 + 同 n 8k 串行 page RMW, k 间隔 ≥ 8)
  => 单 BitLinear 当前已近最优, makespan 优化在跨 BitLinear (S2 异步门铃 / S3 融合)

用法:
  python cim_compiler/cimres/cost_model.py --in <placed.mlir>
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
CIM_COMPILER = os.path.dirname(HERE)
REPO = os.path.dirname(CIM_COMPILER)
EXPORT_DIR = os.path.join(CIM_COMPILER, "export")
for _p in (REPO, EXPORT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cim_compiler.cimres.hw_simulator import T_FETCH, T_DISPATCH, T_MATMUL, T_WB
from cim_compiler.cimres.passes.common import func_blocks, matmuls_in_func


def estimate_func(ms) -> int:
    """预估单 func (BitLinear) 的 makespan (cycle)。ms = matmuls_in_func 列表。

    模拟 Controller._run: 按序取指 (T_FETCH), dispatch 非阻塞 (macro_busy/page_busy),
    SYNC_HALT join (max all busy)。无数值, 只 cycle。
    """
    macro_busy = {}   # dest_id -> busy_until (同 Macro 串行)
    page_busy = {}    # psum_page -> busy_until (同 PAGE RMW 串行, K维累加)
    cycle = 0
    for m in ms:
        cycle += T_FETCH
        start = max(cycle, macro_busy.get(m["dest_id"], 0))
        finish = start + T_DISPATCH + T_MATMUL
        macro_busy[m["dest_id"]] = finish
        wb_start = max(finish, page_busy.get(m["psum_page"], 0))
        page_busy[m["psum_page"]] = wb_start + T_WB
    # SYNC_HALT join: 等所有 Macro + Arbiter 完成
    return max([cycle] + list(macro_busy.values()) + list(page_busy.values()))


def estimate(mod) -> dict:
    """预估 forward makespan (所有 func 串行和, 不含 Preload)。

    每 func = 一个 doorbell (macro_busy/page_busy 重置, 对齐 Controller.doorbell)。
    返回 {per_func: [{name, makespan, n_tile, n_parallel}], total, n_func}。
    """
    per_func = []
    total = 0
    for func_op, _ in func_blocks(mod):
        ms = matmuls_in_func(func_op)
        if not ms:
            continue
        mk = estimate_func(ms)
        # 并行度 = 同一时刻最多多少 Macro 在算 (峰值并行 tile 数)
        n_macro = len({m["dest_id"] for m in ms})
        per_func.append({
            "name": ms[0]["bitlinear_name"],
            "makespan": mk,
            "n_tile": len(ms),
            "n_macro": n_macro,
        })
        total += mk
    return {"per_func": per_func, "total": total, "n_func": len(per_func)}


def estimate_kv(mod, n_tokens, block_size=256):
    """S3: KV cache vs 全序列重算的 CIM cycle 对比 (cim_stub M 行循环)。

    cim_stub.c cim_launch 对 M 行循环 (line 231/259 for m<M, "单 token 指令流
    重复 M 次"): 每行执行 forward.bin 一次 (1 doorbell)。base = forward makespan @ M=1 (单行)。

    全序列重算 (无 KV cache): 生成 n token, 每步 t 输入 T=min(t+1, block_size)
        (滑动窗口 crop), 每步 makespan = base × T。总 = sum_t。
    KV cache: prefill(prompt) + decode(单 token), 全程 M=1 (decode 每步只算
        新 token 的 K/V/q/o/fc proj, CIM matmul M: T->1)。总 = base × n。

    O(n²)->O(n): speedup = sum(T)/n ≈ (n+1)/2 (大 n)。
        实测 (AOT cim_sim_kv): n=80 全序列 cim=22819256 vs 增量 cim=542784, speedup 42x (≈ n/2)。
        旧 cost_model 用 ceil(T/64) (M-tile) 估算 1.20x 严重低估 -- cim_stub 实际 M 行循环。
    """
    r = estimate(mod)
    base = r["total"]                                  # forward makespan @ M=1 (单行)
    full = 0
    for t in range(n_tokens):
        T = min(t + 1, block_size)                     # 滑动窗口 crop
        full += base * T                               # M 行循环: makespan ∝ M (行数, cim_stub for m<M)
    kv = base * n_tokens                               # prefill + decode 全程 M=1
    return {
        "n": n_tokens, "base": base,
        "full": full, "kv": kv,
        "speedup": full / max(kv, 1),
        "full_per_tok": full / n_tokens, "kv_per_tok": kv / n_tokens,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default="cim_compiler/cimres/checkpoints/bitnet_ternary_cimres_placed.mlir")
    p.add_argument("--top", type=int, default=5, help="打印 makespan 最大的 N 个 func")
    p.add_argument("--kv", type=int, default=None, nargs="?", const=256,
                   help="S3: 量化 KV cache vs 全序列重算 CIM cycle (生成 N token; 不传 N 则打印 32/64/128/256 表)")
    args = p.parse_args()

    from cim_compiler.cimres.passes.common import load_cimres
    mod, _ = load_cimres(args.inp)

    if args.kv is not None:
        if args.kv == 256 and not sys.argv[-1].lstrip("-").isdigit():
            ns = (32, 64, 128, 256)            # --kv 无参数 -> 打印对比表
        else:
            ns = (args.kv,)
        base = estimate(mod)["total"]
        print(f"[cost_model] S3 KV cache vs 全序列重算 (CIM cycle, cim_stub M 行循环, "
              f"base={base}/step @ M=1):", file=sys.stderr)
        print(f"  {'n':>5} | {'全序列重算':>14} | {'KV cache':>12} | {'speedup':>8} | "
              f"{'每token 全/kv':>14}", file=sys.stderr)
        print("-" * 62, file=sys.stderr)
        for n in ns:
            r = estimate_kv(mod, n)
            print(f"  {r['n']:>5} | {r['full']:>12d} c | {r['kv']:>10d} c | "
                  f"{r['speedup']:>7.2f}x | {r['full_per_tok']:>6.0f}/{r['kv_per_tok']:.0f}",
                  file=sys.stderr)
        print("-" * 62, file=sys.stderr)
        print("[cost_model] makespan ∝ M 行 (cim_stub for m<M); speedup ≈ (n+1)/2 (O(n²)->O(n), 实测 n=80 42x)",
              file=sys.stderr)
        return

    r = estimate(mod)

    print(f"[cost_model] {r['n_func']} func, 总 forward makespan = {r['total']} cycle", file=sys.stderr)
    print(f"[cost_model] 平均/func = {r['total'] // max(r['n_func'], 1)} cycle", file=sys.stderr)
    print(f"[cost_model] makespan 最大的 {args.top} 个 func:", file=sys.stderr)
    top = sorted(r["per_func"], key=lambda x: -x["makespan"])[:args.top]
    for f in top:
        print(f"  {f['makespan']:6d} cycle  {f['n_tile']:4d} tile  {f['n_macro']:4d} Macro  {f['name']}",
              file=sys.stderr)
    # 单 BitLinear 最优性参考 (8n×8k, 512维)
    q = next((f for f in r["per_func"] if "q.proj" in f["name"]), None)
    if q and q["n_tile"] == 64:
        print(f"[cost_model] q_proj (8n×8k=64 tile) makespan={q['makespan']} cycle, "
              f"下界 ~126 (8n 并行 + 同 n 8k 串行)", file=sys.stderr)


if __name__ == "__main__":
    main()
