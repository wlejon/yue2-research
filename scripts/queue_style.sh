#!/bin/bash
# The style axes at a switch (Exp 3, D only): for each axis, the AR steer at
# layer 24 on a continuation from second 60 of folk and rock, at the scales
# given, plus the unsteered continuation (C0); interrupt_eval.py then gives
# the change over the 10 s span and the onset step at the switch (switch_z).
#
#   scripts/queue_style.sh "-2,-1,1,2"          (GPU 0: source scripts/gpu0.sh first)
cd "$(dirname "$0")/.." || exit 1
scales=${1:--2,-1,1,2}
ar=$(echo "$scales" | tr ',' '\n' | sed 's/^/24:/' | paste -sd, -)
for axis in density timbre tempo; do
  for s in folk rock; do
    .venv/Scripts/python interrupt.py $s --t0 60 --n 250 --axis $axis --d_only --ar "$ar" --out runs/exp3_$axis \
      > runs/exp3_${axis}_$s.log 2>&1 || echo "$axis $s failed (runs/exp3_${axis}_$s.log)"
  done
  .venv/Scripts/python interrupt_eval.py runs/exp3_$axis folk rock > /dev/null 2>&1 || echo "$axis eval failed"
done
