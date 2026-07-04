#!/usr/bin/env python3
"""
char-level data layer for shakespeare_char (nanoGPT-style).

Reads train.bin / val.bin (uint16) + meta.pkl produced by
data/shakespeare_char/prepare.py. No HF transformers/datasets dependency.

Usage:
    from data_char import get_batch, CharTokenizer, get_meta
    x, y = get_batch("train", batch_size=32, block_size=256, device="cuda")
    tok  = CharTokenizer()              # wraps itos/stoi, model.generate-compatible
"""
import os
import pickle

import numpy as np
import torch

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "shakespeare_char"
)

_train = None
_val = None
_meta = None


def _load():
    """Lazily memmap the .bin files and load meta.pkl once (module-level cache)."""
    global _train, _val, _meta
    if _train is None:
        _train = np.memmap(
            os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16, mode="r"
        )
        _val = np.memmap(
            os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16, mode="r"
        )
        with open(os.path.join(DATA_DIR, "meta.pkl"), "rb") as f:
            _meta = pickle.load(f)
    return _train, _val, _meta


def get_meta() -> dict:
    _, _, meta = _load()
    return meta


def get_batch(split: str, batch_size: int, block_size: int, device):
    """nanoGPT-style random-crop batch.

    Returns (x, y) LongTensors of shape (B, T) where y[t] = x[t+1].
    """
    train, val, _ = _load()
    data = train if split == "train" else val
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(np.array(data[i : i + block_size], dtype=np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy(np.array(data[i + 1 : i + 1 + block_size], dtype=np.int64))
            for i in ix
        ]
    )
    if device == "cpu":
        return x, y
    return (
        x.pin_memory().to(device, non_blocking=True),
        y.pin_memory().to(device, non_blocking=True),
    )


class CharTokenizer:
    """Wraps the char-level stoi/itos so BitNet.generate(tokenizer=...) works unchanged."""

    def __init__(self, meta: dict | None = None):
        m = meta or get_meta()
        self.stoi = m["stoi"]
        self.itos = m["itos"]
        self.vocab_size = m["vocab_size"]

    def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "".join(self.itos[int(i)] for i in ids)
