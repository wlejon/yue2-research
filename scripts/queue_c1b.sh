#!/usr/bin/env bash
# GPU 1 follow-up (until 09:45 CDT 2026-10-12): finer interrupt scales below the
# break points the first pass found.
set -u
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONUTF8=1
while ps -ef | grep -v grep | grep -q "slide.py\|plan_inc.py\|queue_c1.sh"; do sleep 10; done
for s in folk rock synthpop; do
  .venv/Scripts/python interrupt.py $s --t0 60 --n 250 --lat_scales 0.5,1,1.5 --nar 9:2,9:4,21:2,21:4,15:4 --ar 21:0.5,9:1 > runs/exp3b_$s.log 2>&1
done
echo "interrupt fine done"
