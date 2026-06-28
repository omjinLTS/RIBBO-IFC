#!/bin/bash
# IFC EE D=20 training, 3BA, 3000 epochs. Mirrors run_train_ee.sh (EE D=10) with D=20.
# Self-contained: fsync'd GPU telemetry + resume from latest ckpt + 250-epoch checkpoints.
export PYTHONPATH=.
export WANDB_MODE=disabled
export OMP_NUM_THREADS=13 MKL_NUM_THREADS=13 NUMEXPR_NUM_THREADS=13
LOGDIR=/tmp/claude-1000/-home-myoungjin-01-Research-01-Projects-RIBBO/1d24f5c6-1aef-4a4d-a86d-8c8be54b8907/scratchpad
BB=./log/gpu_blackbox.csv
mkdir -p "$LOGDIR" "$(dirname "$BB")"

# embedded telemetry recorder
( while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw,clocks.sm,clocks.mem,memory.used,clocks_throttle_reasons.active --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')" >> "$BB"
    sync "$BB" 2>/dev/null; sleep 2
  done ) & BB_PID=$!
trap "kill $BB_PID 2>/dev/null" EXIT

# resume from latest checkpoint if present
RUNDIR=./log/ifc/dt-ifc-ee-d20-3ba
RESUME=""
if [ -d "$RUNDIR" ]; then
  L=$(ls "$RUNDIR"/*/ckpt/*.ckpt 2>/dev/null | awk -F/ '{n=$NF;gsub(/\.ckpt/,"",n);print n,$0}' | sort -n | tail -1 | cut -d' ' -f2-)
  EP=$(basename "${L:-0.ckpt}" .ckpt)
  [ -n "$L" ] && [ "$EP" -gt 1 ] 2>/dev/null && RESUME="--load_ckpt $L --start_epoch $EP"
fi

echo "=== EE D=20 training START $(date) resume='${RESUME:-none}' ==="
python -u scripts/run_dt.py \
    --config scripts/configs/dt/ifc_d20_ee.py \
    --name dt-ifc-ee-d20-3ba \
    --embed_dim 256 --num_heads 8 --num_layers 12 \
    --pos_encoding embed \
    --input_seq_len 50 --max_input_seq_len 300 \
    --num_epoch 3000 --batch_size 64 \
    --eval_interval 3000 --save_interval 250 \
    --id IFCEED20 --seed 0 --device cuda:0 $RESUME \
    > "$LOGDIR/train_ee_d20.log" 2>&1
echo "=== EE D=20 training END $(date) exit=$? ==="
