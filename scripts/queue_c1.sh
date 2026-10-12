#!/usr/bin/env bash
# GPU 1 (bus 02) reference queue — only while Jonny lends GPU 1 (until 09:45 CDT
# 2026-10-12). Everything here must have exited by then.
set -u
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONUTF8=1
for s in folk rock; do .venv/Scripts/python interrupt.py $s --t0 60 --n 250 --nar 9:8,21:8,21:16 --ar 21:1,21:2 > runs/exp3_$s.log 2>&1; done
echo "interrupt done"
.venv/Scripts/python plan_inc.py folk synthpop > runs/exp4b.log 2>&1
echo "plan_inc done"
.venv/Scripts/python slide.py folk > runs/exp4.log 2>&1
echo "slide done"
