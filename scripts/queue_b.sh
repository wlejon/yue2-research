#!/usr/bin/env bash
# Session queue B (after queue_a): native jobs, then the reference model.
set -u
cd "$(dirname "$0")/.."
source scripts/gpu0.sh
BRO=bin/bro-headless.exe
YUE_JOB=jobs/exp2c.json $BRO native native/gen.js > runs/exp2c_b.log 2>&1
echo "exp2c: $(grep -c wall runs/exp2c_b.log) more runs, errors: $(grep -c ERROR runs/exp2c_b.log)"
for s in base vae bf16 int8 int8_vae gen_default gen_bf16 gen_max4500 gen_cfg gen_W500 gen_novae; do
  YUE_JOB=jobs/vram_$s.json $BRO native native/vram.js > runs/vram_$s.log 2>&1
done
echo "vram done"
YUE_JOB=jobs/exp1d_sem.json $BRO native native/gen.js > runs/exp1d_sem.log 2>&1
YUE_JOB=jobs/exp1d_nar.json $BRO native native/gen.js > runs/exp1d_nar.log 2>&1
YUE_JOB=jobs/exp1d_sem_conc.json $BRO native native/gen.js > runs/exp1d_sem_conc.log 2>&1 &
YUE_JOB=jobs/exp1d_nar_conc.json $BRO native native/gen.js > runs/exp1d_nar_conc.log 2>&1
wait
echo "exp1d done"
YUE_JOB=jobs/exp4c.json $BRO native native/gen.js > runs/exp4c.log 2>&1
echo "exp4c done"
.venv/Scripts/python steer_eval.py energy > runs/steer_eval.log 2>&1
echo "steer_eval done"
for s in folk rock; do .venv/Scripts/python interrupt.py $s --t0 60 --n 250 --nar 9:8,21:8,21:16 --ar 21:1,21:2 > runs/exp3_$s.log 2>&1; done
echo "interrupt done"
.venv/Scripts/python slide.py folk > runs/exp4.log 2>&1
echo "slide done"
.venv/Scripts/python plan_inc.py folk synthpop > runs/exp4b.log 2>&1
echo "plan_inc done"
