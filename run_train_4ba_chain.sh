#!/bin/bash
# Train (B) SE D=20 4BA-filter and (C) EE D=10 4BA-filter (HC, RegEvol, Eagle, CMAES).
# Embedded fsync telemetry + auto-resume + 250-ep checkpoints. Symlink dirs prebuilt.
export PYTHONPATH=.
export WANDB_MODE=disabled
export OMP_NUM_THREADS=13 MKL_NUM_THREADS=13 NUMEXPR_NUM_THREADS=13
PY=/home/myoungjin/miniconda3/envs/RIBBO/bin/python   # explicit RIBBO env (terminal session lacks conda activate)
SP=/tmp/claude-1000/-home-myoungjin-01-Research-01-Projects-RIBBO/5a3b2b50-c7a4-4488-add4-3c82eb5c6b2a/scratchpad
BB=./log/gpu_blackbox.csv
mkdir -p "$SP" "$(dirname "$BB")"

( while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw,clocks.sm,clocks.mem,memory.used,clocks_throttle_reasons.active --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')" >> "$BB"
    sync "$BB" 2>/dev/null; sleep 2
  done ) & BBPID=$!
trap "kill $BBPID 2>/dev/null" EXIT

train() { # config name id log
  local rundir="./log/ifc/$2" resume="" L EP
  if [ -d "$rundir" ]; then
    L=$(ls "$rundir"/*/ckpt/*.ckpt 2>/dev/null | awk -F/ '{n=$NF;gsub(/\.ckpt/,"",n);print n,$0}' | sort -n | tail -1 | cut -d' ' -f2-)
    EP=$(basename "${L:-0.ckpt}" .ckpt)
    [ -n "$L" ] && [ "$EP" -gt 1 ] 2>/dev/null && resume="--load_ckpt $L --start_epoch $EP"
  fi
  echo "=== $2 START $(date) resume='${resume:-none}' ==="
  $PY -u scripts/run_dt.py --config "$1" --name "$2" --id "$3" \
    --embed_dim 256 --num_heads 8 --num_layers 12 --pos_encoding embed \
    --input_seq_len 50 --max_input_seq_len 300 \
    --num_epoch 3000 --batch_size 64 --eval_interval 3000 --save_interval 250 \
    --seed 0 --device cuda:0 $resume > "$4" 2>&1
  echo "=== $2 END $(date) exit=$? ==="
}

train scripts/configs/dt/ifc_d20_4bafilter.py    dt-ifc-d20-4ba    IFCSED20 "$SP/train_d20_4ba.log"
train scripts/configs/dt/ifc_d10_ee_4bafilter.py dt-ifc-ee-d10-4ba IFCEED10 "$SP/train_ee_d10_4ba.log"
echo "=== 4BA CHAIN COMPLETE $(date) ==="
