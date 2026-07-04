#!/usr/bin/env python3
"""
Prepare shakespeare_char dataset (nanoGPT-style).

Reuses the repo's existing data/input.txt — byte-for-byte identical to
nanoGPT's tinyshakespeare source (1,115,394 chars, 65 unique).

Generates (in this directory):
  input.txt   — symlink to ../../input.txt (kept for nanoGPT parity)
  train.bin   — np.uint16 token ids, first 90%
  val.bin     — np.uint16 token ids, last  10%
  meta.pkl    — {vocab_size, stoi, itos}

Run:
  python data/shakespeare_char/prepare.py
"""
import os
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_INPUT = os.path.join(HERE, "..", "input.txt")

# nanoGPT keeps a local input.txt in the dataset dir; symlink to the repo copy
local_input = os.path.join(HERE, "input.txt")
if not os.path.exists(local_input):
    if not os.path.exists(REPO_INPUT):
        raise FileNotFoundError(
            f"Expected {REPO_INPUT} (tinyshakespeare). Provide it or download from "
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        )
    os.symlink(os.path.relpath(REPO_INPUT, HERE), local_input)

with open(local_input, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]


data = np.array(encode(text), dtype=np.uint16)
n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9):]

train_data.tofile(os.path.join(HERE, "train.bin"))
val_data.tofile(os.path.join(HERE, "val.bin"))

meta = {"vocab_size": vocab_size, "itos": itos, "stoi": stoi}
with open(os.path.join(HERE, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print(f"vocab_size : {vocab_size}")
print(f"total chars: {n}")
print(f"train.bin  : {len(train_data)} tokens")
print(f"val.bin    : {len(val_data)} tokens")
