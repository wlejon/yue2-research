#!/usr/bin/env bash
# Exp 7b on GPU 0: trimmed bridges under a score, adaptive lyrics; CLAP,
# separation and Whisper after.
set -u
cd "$(dirname "$0")/.."
source scripts/gpu0.sh
YUE_JOB=jobs/exp7b.json bin2/bro-headless.exe native native/endless.js > runs/exp7b.log 2>&1
YUE_JOB=jobs/exp7_clap.json bin2/bro-headless.exe native native/clap_embed.js > runs/exp7b_clap.log 2>&1
.venv/Scripts/python sep.py runs/exp7/roll_adapt_w4 runs/exp7/roll_adapt_w6 > runs/exp7b_sep.log 2>&1
YUE_JOB=jobs/exp7_stt.json bin2/bro-headless.exe native native/stt.js > runs/exp7b_stt.log 2>&1
echo "exp7b done"
