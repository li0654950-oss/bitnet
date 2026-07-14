"""export 公共逻辑: 模型构建参数 + 模型构建 (export_fx / export_kv / verify_export 共用)。

作为架构参数 (d_model/n_layer/n_head/n_kv_head/ffn_dim/block_size) 的单一来源,
消除三个入口重复的 argparse + build_inference_model 调用, 避免硬编码默认值散落、
verify 漏传参数静默用错配置 (checkpoint 不存 config, 故默认值在此集中定义)。
"""
import argparse

from cim_compiler.export.inference_model import build_inference_model
from bitnet.data_char import get_meta

# 架构默认值 (单一来源, 对应 bitnet/train_shakespeare_char.py 的 BitNet 配置)
_ARCH_DEFAULTS = dict(
    d_model=512, block_size=256, n_layer=6,
    n_head=8, n_kv_head=4, ffn_dim=1664,
)


def add_model_args(p: argparse.ArgumentParser) -> None:
    """给 parser 加 --ternary + 架构参数 (默认值单一来源)。各入口的 IO 路径参数自行加。"""
    p.add_argument("--ternary", default="checkpoints/bitnet_shakespeare_char_ternary.pt")
    for name, default in _ARCH_DEFAULTS.items():
        p.add_argument(f"--{name}", type=int, default=default)


def build_model_from_args(args):
    """按 args 构建 inference 模型 (vocab 从 meta.pkl 读, 架构参数从 args)。"""
    return build_inference_model(
        args.ternary,
        vocab_size=get_meta()["vocab_size"],
        d_model=args.d_model, block_size=args.block_size,
        n_layer=args.n_layer, n_head=args.n_head,
        n_kv_head=args.n_kv_head, ffn_dim=args.ffn_dim,
    )
