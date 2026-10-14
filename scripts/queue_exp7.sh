#!/usr/bin/env bash
# Exp 7 on GPU 0: the streams, CLAP per chunk, separation of the sung
# streams, Whisper on their vocal stems.
set -u
cd "$(dirname "$0")/.."
source scripts/gpu0.sh
YUE_JOB=jobs/exp7.json bin2/bro-headless.exe native native/endless.js > runs/exp7.log 2>&1
YUE_JOB=jobs/exp7_clap.json bin2/bro-headless.exe native native/clap_embed.js > runs/exp7_clap.log 2>&1
.venv/Scripts/python sep.py runs/exp7/roll_r1_w4 runs/exp7/roll_r2_w4 runs/exp7/sheet_long > runs/exp7_sep.log 2>&1
YUE_JOB=jobs/exp7_stt.json bin2/bro-headless.exe native native/stt.js > runs/exp7_stt.log 2>&1
echo "exp7 done"
