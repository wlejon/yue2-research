#!/usr/bin/env bash
# Research venv: the same CPython as ../YuE2-3B/.venv, which it reads through
# .venv/Lib/site-packages/yue2_ref.pth (one line: the absolute path of
# ../YuE2-3B/.venv/Lib/site-packages), giving torch+CUDA and the reference
# `yue2` package without a second copy. Copy scripts/yue2_ref.pth there after
# the venv exists, editing the path if the checkout lives elsewhere.
# numpy is pinned to the reference venv's version so the shadowing copy is the
# same build.
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv --python 3.11.14 .venv
cp scripts/yue2_ref.pth .venv/Lib/site-packages/yue2_ref.pth
uv pip install --python .venv/Scripts/python.exe numpy==2.2.6 scipy librosa soundfile pyloudnorm scikit-learn matplotlib
