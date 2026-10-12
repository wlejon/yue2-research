#!/usr/bin/env bash
# Session queue: one GPU-0 job at a time (native and reference do not fit together).
set -u
cd "$(dirname "$0")/.."
source scripts/gpu0.sh
YUE_JOB=jobs/exp2c.json bin/bro-headless.exe native native/gen.js > runs/exp2c.log 2>&1
echo "exp2c done: $(grep -c wall runs/exp2c.log) runs"
.venv/Scripts/python stream_ref.py long synthpop > runs/exp1c_long2.log 2>&1
echo "long synthpop done"
.venv/Scripts/python stream_ref.py design folk rock synthpop > runs/exp1c_design.log 2>&1
echo "design done"
