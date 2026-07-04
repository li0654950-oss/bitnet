#!/usr/bin/env bash
# Minimal env for the 15M ternary BitNet on shakespeare_char.
# Char-level and self-contained — no HF transformers/datasets needed.
set -e
python3 -m venv venv
source venv/bin/activate
pip install torch numpy pytest
