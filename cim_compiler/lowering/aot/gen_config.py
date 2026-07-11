#!/usr/bin/env python3
"""AOT: 从 .pt2 提取模型配置 -> model_config.bin (cim_main.c 固定宿主运行时读)。

让 cim_main.c 成为固定通用宿主: 换模型只换 .pt2/forward.bin/preload.bin + 重编译
forward.o, cim_main.c 不改一行。forward 入口参数个数随 n_layer 变, 由 libffi 运行时
变参调用解决; 其余规模相关点 (inv_freq/causal_mask/w_packed shape/vocab/tokenizer)
运行时从本配置读。

提取 (从 ExportedProgram.graph_signature.input_specs, 复用 cim_jit.build_inputs 逻辑):
  - PARAMETER 跳过 (constant-fold 内嵌)
  - USER_INPUT = idx (最后, 不入 buffer 表)
  - BUFFER 按 target 后缀定 kind: inv_freq / causal_mask / lm_head.w_packed / w_packed
  - shape 从 placeholder 节点 meta.val.shape 读
超参反推: d_head=inv_freq_shape*2, block=causal_mask_shape[-1],
         vocab=lm_head_shape[0], n_layer=(n_buffer-1)//8
tokenizer: CharTokenizer (bitnet/data_char.py) itos/stoi 导出 (char-level)。

用法:
  python cim_compiler/lowering/aot/gen_config.py
  python cim_compiler/lowering/aot/gen_config.py --pt2 checkpoints/bitnet_ternary.pt2 \\
      --out cim_compiler/cimres/checkpoints/model_config.bin
"""
import os
import sys
import struct
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
LOWERING = os.path.dirname(HERE)
CIM_COMPILER = os.path.dirname(LOWERING)
REPO = os.path.dirname(CIM_COMPILER)
EXPORT_DIR = os.path.join(CIM_COMPILER, "export")
BITNET = os.path.join(REPO, "bitnet")
for _p in (REPO, EXPORT_DIR, BITNET):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import torch.export  # noqa: E402
import cim_op  # noqa: E402,F401  (注册 cim::matmul, 反序列化 .pt2 需要)
from data_char import CharTokenizer  # noqa: E402

MC_MAGIC = b"CIMC"
KIND_INVFREQ, KIND_CAUSAL_MASK, KIND_W_PACKED, KIND_LMHEAD = 0, 1, 2, 3
_KIND_NAME = {0: "inv_freq", 1: "causal_mask", 2: "w_packed", 3: "lmhead_w"}


def classify(target):
    """按 target 后缀定 buffer kind (与 cim_jit.build_inputs 一致)。"""
    if "inv_freq" in target:
        return KIND_INVFREQ
    if "causal_mask" in target:
        return KIND_CAUSAL_MASK
    if "lm_head" in target and target.endswith("w_packed"):
        return KIND_LMHEAD
    if target.endswith("w_packed"):
        return KIND_W_PACKED
    raise ValueError(f"未知 buffer target: {target}")


def extract(pt2_path):
    """从 .pt2 提取 buffer 描述表 + 反推超参。"""
    prog = torch.export.load(pt2_path)
    # placeholder name -> shape (从节点 meta.val)
    ph_shape = {}
    for node in prog.graph.nodes:
        if node.op == "placeholder":
            v = node.meta.get("val")
            ph_shape[node.name] = [int(s) for s in v.shape] if (v is not None and hasattr(v, "shape")) else []

    buffers = []
    idx_pos = None
    for s in prog.graph_signature.input_specs:
        k = s.kind.name
        if k == "PARAMETER":
            continue
        if k == "USER_INPUT":
            idx_pos = len(buffers)        # idx 在 buffer 表之后 (最后)
            continue
        # BUFFER
        shape = ph_shape.get(s.arg.name, [])
        buffers.append((classify(s.target), len(shape), shape))

    if idx_pos is None:
        raise ValueError(".pt2 无 USER_INPUT (idx)")

    # 超参反推
    inv = next(b for b in buffers if b[0] == KIND_INVFREQ)
    cm = next(b for b in buffers if b[0] == KIND_CAUSAL_MASK)
    lmh = next(b for b in buffers if b[0] == KIND_LMHEAD)
    d_head = inv[2][0] * 2
    block_size = cm[2][-1]               # [1,1,B,B] -> B
    vocab = lmh[2][0]
    n_buffer = len(buffers)
    n_layer = (n_buffer - 1) // 8         # 减 lm_head, 每层 8 buffer
    return buffers, n_buffer, n_layer, vocab, block_size, d_head


def build_bin(buffers, n_layer, vocab, block_size):
    """序列化为 model_config.bin (格式见 model_config.h)。"""
    buf = bytearray()
    buf += MC_MAGIC
    buf += struct.pack("<IIII", len(buffers), n_layer, vocab, block_size)
    for kind, rank, shape in buffers:
        buf += struct.pack("<BB", kind, rank)
        buf += struct.pack(f"<{rank}q", *shape)
    # tokenizer (char-level): itos[vocab] + stoi[128]
    tok = CharTokenizer()
    itos = tok.itos                        # {id: char} 或 list
    getter = (lambda i: itos[i]) if hasattr(itos, "__getitem__") else None
    if getter is None:
        raise ValueError(f"itos 不可索引: {type(itos)}")
    itos_bytes = bytes(ord(getter(i)) for i in range(vocab))
    if len(itos_bytes) != vocab:
        raise ValueError(f"itos 长度 {len(itos_bytes)} != vocab {vocab}")
    buf += itos_bytes
    stoi = tok.stoi                        # {char: id}
    stoi_arr = [-1] * 128
    for c, i in stoi.items():
        stoi_arr[ord(c)] = i
    buf += struct.pack("<128i", *stoi_arr)
    return bytes(buf)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pt2", default=os.path.join(REPO, "checkpoints", "bitnet_ternary.pt2"))
    p.add_argument("--out", default=os.path.join(REPO, "cim_compiler", "cimres", "checkpoints", "model_config.bin"))
    args = p.parse_args()

    buffers, n_buffer, n_layer, vocab, block_size, d_head = extract(args.pt2)
    data = build_bin(buffers, n_layer, vocab, block_size)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(data)

    print(f"[gen_config] {args.pt2}", file=sys.stderr)
    print(f"  n_buffer={n_buffer} n_layer={n_layer} vocab={vocab} "
          f"block_size={block_size} d_head={d_head}", file=sys.stderr)
    print(f"  buffers:", file=sys.stderr)
    for i, (kind, rank, shape) in enumerate(buffers):
        print(f"    [{i:2d}] {_KIND_NAME[kind]:14s} shape={shape}", file=sys.stderr)
    print(f"[gen_config] saved: {args.out} ({len(data)} 字节)", file=sys.stderr)


if __name__ == "__main__":
    main()
