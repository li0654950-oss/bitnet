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

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
import torch.export  # noqa: E402
from cim_compiler.export import cim_op  # noqa: E402,F401  (注册 cim::matmul, 反序列化 .pt2 需要)
from bitnet.data_char import CharTokenizer  # noqa: E402
from cim_compiler.lowering.buffer_kind import (  # noqa: E402
    classify_buffer, KIND_INVFREQ, KIND_CAUSAL_MASK, KIND_W_PACKED, KIND_LMHEAD, KIND_NAME)

MC_MAGIC = b"CIMC"


def extract(pt2_path, kv=False):
    """从 .pt2 提取 buffer 描述表 + 反推超参。

    kv=False (全序列): USER_INPUT 仅 idx (1个), n_kv=0, n_layer=(n_buffer-1)//8。
    kv=True  (增量):   USER_INPUT = idx, k_caches, v_caches, cos, sin (5个),
                       n_kv/head_dim/n_layer 从 k_caches shape [n_layer,B,T,n_kv,hd] 反推。
    """
    prog = torch.export.load(pt2_path)
    # placeholder name -> shape (从节点 meta.val)
    ph_shape = {}
    for node in prog.graph.nodes:
        if node.op == "placeholder":
            v = node.meta.get("val")
            ph_shape[node.name] = [int(s) for s in v.shape] if (v is not None and hasattr(v, "shape")) else []

    buffers = []
    ui_shapes = []                          # USER_INPUT shapes 按出现顺序
    for s in prog.graph_signature.input_specs:
        k = s.kind.name
        if k == "PARAMETER":
            continue
        if k == "USER_INPUT":
            ui_shapes.append(ph_shape.get(s.arg.name, []))
            continue
        # BUFFER
        shape = ph_shape.get(s.arg.name, [])
        buffers.append((classify_buffer(s.target), len(shape), shape))

    if not ui_shapes:
        raise ValueError(".pt2 无 USER_INPUT")

    # 超参反推
    inv = next(b for b in buffers if b[0] == KIND_INVFREQ)
    cm = next(b for b in buffers if b[0] == KIND_CAUSAL_MASK)
    lmh = next(b for b in buffers if b[0] == KIND_LMHEAD)
    head_dim = inv[2][0] * 2                # inv_freq[d_head/2] -> d_head
    block_size = cm[2][-1]                  # [1,1,B,B] -> B
    vocab = lmh[2][0]
    n_buffer = len(buffers)
    if kv:
        # USER_INPUT 顺序: idx, k_caches, v_caches, cos, sin
        if len(ui_shapes) < 2:
            raise ValueError(f"增量 .pt2 USER_INPUT < 2 (需 k_caches): {len(ui_shapes)}")
        kc = ui_shapes[1]                   # [n_layer, B, T, n_kv, head_dim]
        if len(kc) != 5:
            raise ValueError(f"k_caches rank != 5: shape={kc}")
        n_layer = kc[0]
        n_kv = kc[3]
        head_dim = kc[4]
    else:
        n_layer = (n_buffer - 1) // 8        # 减 lm_head, 每层 8 buffer
        n_kv = 0
    # inv_freq data (和 model 一致): 训练时 float64 计算 -> .pt 存 bfloat16 (0.749894->0.75 舍入) -> float32
    # 必须经 bfloat16 中间, 否则 float32 直接算得 0.749894 (与 model 0.75 微差致 RoPE argmax 翻转)
    inv_freq_data = (1.0 / (10000 ** (torch.arange(0, head_dim, 2) / head_dim))).to(torch.bfloat16).to(torch.float32)
    return buffers, n_buffer, n_layer, vocab, block_size, head_dim, n_kv, inv_freq_data


def build_bin(buffers, n_layer, vocab, block_size, n_kv, head_dim, inv_freq_data):
    """序列化为 model_config.bin (格式见 model_config.h)。"""
    buf = bytearray()
    buf += MC_MAGIC
    buf += struct.pack("<IIIIII", len(buffers), n_layer, vocab, block_size, n_kv, head_dim)
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
    # inv_freq data (float32, head_dim/2 个, 和 model 一致 -- 避免 C powf vs torch pow 浮点差)
    buf += inv_freq_data.numpy().tobytes()
    return bytes(buf)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pt2", default=None,
                   help=".pt2 路径 (默认: --kv 用 bitnet_ternary_kv.pt2, 否则 bitnet_ternary.pt2)")
    p.add_argument("--out", default=None,
                   help="输出 model_config.bin (默认: --kv 用 model_config_kv.bin, 否则 model_config.bin)")
    p.add_argument("--kv", action="store_true", help="增量 KV cache .pt2 (提取 n_kv/head_dim)")
    args = p.parse_args()

    default_pt2 = os.path.join(REPO, "checkpoints",
                               "bitnet_ternary_kv.pt2" if args.kv else "bitnet_ternary.pt2")
    default_out = os.path.join(REPO, "cim_compiler", "cimres", "checkpoints",
                               "model_config_kv.bin" if args.kv else "model_config.bin")
    args.pt2 = args.pt2 or default_pt2
    args.out = args.out or default_out

    buffers, n_buffer, n_layer, vocab, block_size, head_dim, n_kv, inv_freq_data = extract(args.pt2, kv=args.kv)
    data = build_bin(buffers, n_layer, vocab, block_size, n_kv, head_dim, inv_freq_data)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(data)

    mode = "增量 KV" if args.kv else "全序列"
    print(f"[gen_config] {mode} {args.pt2}", file=sys.stderr)
    print(f"  n_buffer={n_buffer} n_layer={n_layer} vocab={vocab} "
          f"block_size={block_size} head_dim={head_dim} n_kv={n_kv}", file=sys.stderr)
    print(f"  buffers:", file=sys.stderr)
    for i, (kind, rank, shape) in enumerate(buffers):
        print(f"    [{i:2d}] {KIND_NAME[kind]:14s} shape={shape}", file=sys.stderr)
    print(f"[gen_config] saved: {args.out} ({len(data)} 字节)", file=sys.stderr)


if __name__ == "__main__":
    main()
