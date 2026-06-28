#!/bin/bash
# IFC SE D=20 training, 3BA, 3000 epochs. Scalability point D=3->10->20.
# Self-contained: embeds fsync'd GPU telemetry + resumes from latest ckpt + saves every
# 250 epochs (<=12.5min loss on a freeze). Launch detached/background.
export PYTHONPATH=.
export WANDB_MODE=disabled
export OMP_NUM_THREADS=13 MKL_NUM_THREADS=13 NUMEXPR_NUM_THREADS=13
LOGDIR=/tmp/claude-1000/-home-myoungjin-01-Research-01-Projects-RIBBO/5a3b2b50-c7a4-4488-add4-3c82eb5c6b2a/scratchpad
BB=./log/gpu_blackbox.csv
mkdir -p "$LOGDIR" "$(dirname "$BB")"

# embedded telemetry recorder
( while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw,clocks.sm,clocks.mem,memory.used,clocks_throttle_reasons.active --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')" >> "$BB"
    sync "$BB" 2>/dev/null; sleep 2
  done ) & BB_PID=$!
trap "kill $BB_PID 2>/dev/null" EXIT

# resume from latest checkpoint if present
RUNDIR=./log/ifc/dt-ifc-d20-3ba
RESUME=""
if [ -d "$RUNDIR" ]; then
  L=$(ls "$RUNDIR"/*/ckpt/*.ckpt 2>/dev/null | awk -F/ '{n=$NF;gsub(/\.ckpt/,"",n);print n,$0}' | sort -n | tail -1 | cut -d' ' -f2-)
  EP=$(basename "${L:-0.ckpt}" .ckpt)
  [ -n "$L" ] && [ "$EP" -gt 1 ] 2>/dev/null && RESUME="--load_ckpt $L --start_epoch $EP"
fi

echo "=== D=20 SE training START $(date) resume='${RESUME:-none}' ==="
python -u scripts/run_dt.py \
    --config scripts/configs/dt/ifc_d20.py \
    --name dt-ifc-d20-3ba \
    --embed_dim 256 --num_heads 8 --num_layers 12 \
    --pos_encoding embed \
    --input_seq_len 50 --max_input_seq_len 300 \
    --num_epoch 3000 --batch_size 64 \
    --eval_interval 3000 --save_interval 250 \
    --id IFCSED20 --seed 0 --device cuda:0 $RESUME \
    > "$LOGDIR/train_d20.log" 2>&1
echo "=== D=20 SE training END $(date) exit=$? ==="
