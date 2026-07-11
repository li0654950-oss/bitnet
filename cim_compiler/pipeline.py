#!/usr/bin/env python3
"""[P3-8] 一键流水线: 任意规模 BitNet -> 全流程编译 -> JIT --sim 数值验证。

在硬件约束内 (<=4096 Macro), 给定 ternary.pt + 模型配置, 一条命令跑通:
  export_fx -> L0 to_mlir -> partition -> C1 lower_to_cimres -> C2 place -> C3 emit_instr
  -> L1 cim_lowering -> gen_cim_stub -> cim_jit --sim (max_diff=0 验证)

任意规模 (n_layer/d_model/n_head/n_kv_head/ffn_dim) 自动适配, 无需手改代码。

用法:
  python cim_compiler/pipeline.py                                     # 默认 shakespeare (6层512维)
  python cim_compiler/pipeline.py --n_layer 2 --d_model 256           # 不同规模 (需对应 ternary.pt)
  python c_compiler/pipeline.py --ternary my.pt --n_layer 12 --ffn_dim 3072
"""
import os
import sys
import subprocess
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))  # cim_compiler/
REPO = os.path.dirname(HERE)  # bitnet/ (pipeline.py 在 cim_compiler/, 比 lowering/ 少一层)
DEFAULT_PY = "/home/li/anaconda3/envs/nanogpt-gpu/bin/python"


def run(step, cmd, py):
    full = [py] + cmd
    print(f"\n[pipeline] === {step} ===", file=sys.stderr)
    print(f"[pipeline] $ {' '.join(full)}", file=sys.stderr)
    subprocess.run(full, check=True, cwd=REPO)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ternary", default="checkpoints/bitnet_shakespeare_char_ternary.pt")
    p.add_argument("--vocab_size", type=int, default=65)
    p.add_argument("--python", default=DEFAULT_PY, help="python 解释器 (conda env)")
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_kv_head", type=int, default=4)
    p.add_argument("--ffn_dim", type=int, default=1664)
    p.add_argument("--T", type=int, default=8, help="JIT 验证 seq len")
    p.add_argument("--no-sim", action="store_true", help="跳过 --sim (用 cim_stub CPU fallback)")
    p.add_argument("--start-step", type=int, default=1, help="从第几步开始 (1-12, 调试用)")
    args = p.parse_args()
    py = args.python

    # 文件路径 (相对 REPO, 各脚本默认路径一致)
    pt2 = "checkpoints/bitnet_ternary.pt2"
    weights = "checkpoints/bitnet_ternary_weights.bin"
    partition = "checkpoints/bitnet_ternary_partition.json"
    cimres = "cim_compiler/cimres/checkpoints/bitnet_ternary_cimres.mlir"
    placed = "cim_compiler/cimres/checkpoints/bitnet_ternary_cimres_placed.mlir"
    mlir0 = "checkpoints/bitnet_ternary.mlir"
    placeholder = "checkpoints/bitnet_ternary_placeholder.mlir"
    so = "cim_compiler/lowering/cim_stub.so"

    # [A1] cim_stub.so 固定驱动 (一次编译, 任意规模复用); 不存在则编译
    if not os.path.exists(os.path.join(REPO, so)):
        subprocess.run(["gcc", "-shared", "-fPIC", "-O2", "-o", so,
                        "cim_compiler/lowering/cim_stub.c"], check=True, cwd=REPO)
        print("[pipeline] 编译 cim_stub.so (固定驱动, 任意规模复用)", file=sys.stderr)

    steps = []
    # 1. export_fx: ternary.pt -> .pt2 + weights.bin (vocab 从 meta.pkl 读)
    steps.append(("1. export_fx (ternary -> .pt2 + weights.bin)", [
        "cim_compiler/export/export_fx.py", "--ternary", args.ternary,
        "--out_graph", pt2, "--out_blob", weights,
        "--d_model", str(args.d_model), "--block_size", str(args.block_size),
        "--n_layer", str(args.n_layer), "--n_head", str(args.n_head),
        "--n_kv_head", str(args.n_kv_head), "--ffn_dim", str(args.ffn_dim),
    ]))
    # 2. L0 to_mlir: .pt2 -> bitnet_ternary.mlir
    steps.append(("2. L0 to_mlir (.pt2 -> mlir)", [
        "cim_compiler/ir/to_mlir.py", "--graph", pt2, "--out", mlir0]))
    # 3. partition: .pt2 -> partition.json
    steps.append(("3. partition (.pt2 -> partition.json)", [
        "cim_compiler/partition/partition.py", "--graph", pt2, "--out", partition]))
    # 4. C1 lower_to_cimres: partition.json + weights.bin -> cimres.mlir
    steps.append(("4. C1 lower_to_cimres (partition+weights -> cimres IR)", [
        "cim_compiler/cimres/lower_to_cimres.py",
        "--partition", partition, "--weights", weights, "--out", cimres]))
    # 5. cimres passes: canonicalize + cse (S0, 逻辑层冗余消除, 框架就位)
    steps.append(("5. cimres passes (canon+cse, S0 逻辑层)", [
        "cim_compiler/cimres/run_passes.py", "--in", cimres]))
    # 6. C2 place: cimres.mlir -> placed.mlir (容量校验 <=4096 Macro)
    steps.append(("6. C2 place (cimres -> placed, 容量校验)", [
        "cim_compiler/cimres/place.py", "--in", cimres, "--out", placed]))
    # 7. verify: placed.mlir 结构校验 (S0 安全网, dest_id/accum/PAGE 冲突, gate 失败中止)
    steps.append(("7. verify (placed 结构校验, S0 安全网 gate)", [
        "cim_compiler/cimres/verify.py", "--in", placed]))
    # 8. 调度分析: cost_model + scheduler + page_alloc (S1, makespan/最优性/PAGE 报告, 不改产物)
    steps.append(("8. 调度分析 (cost_model+scheduler+page_alloc, S1)", [
        "cim_compiler/cimres/run_sched_analysis.py", "--in", placed]))
    # 9. C3 emit_instr: placed.mlir + weights.bin -> forward.bin + preload.bin (容量校验)
    steps.append(("9. C3 emit_instr (placed+weights -> forward/preload, 容量校验)", [
        "cim_compiler/cimres/emit_instr.py"]))
    # 10. L1 cim_lowering: bitnet_ternary.mlir -> placeholder.mlir
    steps.append(("10. L1 cim_lowering (mlir -> placeholder)", [
        "cim_compiler/lowering/cim_lowering.py", "--in", mlir0, "--out", placeholder]))
    # 11. cim_jit --sim: JIT + 数值验证 (max_diff=0)  [A1] .so 固定, 不再 gen_cim_stub
    sim_flag = [] if args.no_sim else ["--sim"]
    steps.append(("11. cim_jit (JIT + 数值验证)", [
        "cim_compiler/lowering/cim_jit.py",
        "--in", placeholder, "--ternary", args.ternary, "--so", so,
        "--T", str(args.T),
        "--vocab_size", str(args.vocab_size),
        "--d_model", str(args.d_model), "--block_size", str(args.block_size),
        "--n_layer", str(args.n_layer), "--n_head", str(args.n_head),
        "--n_kv_head", str(args.n_kv_head), "--ffn_dim", str(args.ffn_dim),
    ] + sim_flag))
    # 12. AOT 构建: to_object + make -> cim_sim + model_config.bin (cim_compiler/lowering/aot/)
    #    make 依赖链自动: gen_config (.pt2 -> model_config.bin) + 链接 cim_sim (-lffi 运行时变参)
    #    cim_main.c 固定通用宿主, 任意模型规模复用 (超参运行时从 model_config.bin 读)
    steps.append(("12. AOT 构建 (to_object + make -> cim_sim + model_config.bin)", [
        "make", "-C", "cim_compiler/lowering/aot"]))

    for i, (name, cmd) in enumerate(steps, 1):
        if i < args.start_step:
            print(f"[pipeline] === {name} === (跳过)", file=sys.stderr)
            continue
        if cmd[0] in ("make", "gcc"):   # 非 python 命令, 不加 py 前缀
            print(f"\n[pipeline] === {name} ===", file=sys.stderr)
            print(f"[pipeline] $ {' '.join(cmd)}", file=sys.stderr)
            subprocess.run(cmd, check=True, cwd=REPO)
        else:
            run(name, cmd, py)

    print("\n[pipeline] ✓ 全流程完成 (任意规模自动适配, 硬件约束内)", file=sys.stderr)


if __name__ == "__main__":
    main()
