#!/bin/bash
# Crash-resilient training launcher: auto-resumes from the latest checkpoint, runs the
# GPU black-box recorder alongside, and (optionally) applies a conservative GPU power
# cap to reduce transient-spike-induced silent freezes.
#
# Usage:
#   bash scripts/train_resilient.sh <config.py> <run_name> <id> <num_epoch> [power_cap_W]
# Example (resume/continue EE D=3 to 3000, cap 200W if permitted):
#   bash scripts/train_resilient.sh scripts/configs/dt/ifc_ee.py dt-ifc-ee-d3-3ba IFCEE 3000 200
#
# Notes:
#  - Power cap needs sudo/root; if it fails it is skipped with a warning (no crash).
#  - Resume uses run_dt.py --load_ckpt/--start_epoch (already supported).
set -u
export PYTHONPATH=.
export WANDB_MODE=disabled
export OMP_NUM_THREADS=13 MKL_NUM_THREADS=13 NUMEXPR_NUM_THREADS=13

CONFIG="$1"; NAME="$2"; ID="$3"; NUM_EPOCH="$4"; PCAP="${5:-}"
RUNDIR="./log/ifc/$NAME"

# optional GPU power cap (best-effort)
if [ -n "$PCAP" ]; then
  if nvidia-smi -pm 1 >/dev/null 2>&1 && nvidia-smi -pl "$PCAP" >/dev/null 2>&1; then
    echo "[resilient] GPU power cap set to ${PCAP}W"
  else
    echo "[resilient] WARN: could not set power cap (needs sudo) — continuing uncapped"
  fi
fi

# find latest checkpoint to resume from
LATEST=""; EP=0
if [ -d "$RUNDIR" ]; then
  LATEST=$(ls "$RUNDIR"/*/ckpt/*.ckpt 2>/dev/null | awk -F'/' '{n=$NF; gsub(/\.ckpt/,"",n); print n, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
  EP=$(basename "${LATEST:-0.ckpt}" .ckpt)
fi
RESUME_ARGS=""
if [ -n "$LATEST" ] && [ "$EP" -gt 1 ] 2>/dev/null; then
  RESUME_ARGS="--load_ckpt $LATEST --start_epoch $EP"
  echo "[resilient] resuming $NAME from epoch $EP ($LATEST)"
else
  echo "[resilient] starting $NAME from scratch"
fi

# start black-box recorder
BBLOG="./log/gpu_blackbox_${NAME}.csv"
bash scripts/gpu_blackbox.sh "$BBLOG" &
BB=$!
trap "kill $BB 2>/dev/null" EXIT

echo "=== $NAME train START $(date) (target ${NUM_EPOCH}ep) ==="
python -u scripts/run_dt.py \
    --config "$CONFIG" --name "$NAME" --id "$ID" \
    --embed_dim 256 --num_heads 8 --num_layers 12 --pos_encoding embed \
    --input_seq_len 50 --max_input_seq_len 300 \
    --num_epoch "$NUM_EPOCH" --batch_size 64 \
    --eval_interval "$NUM_EPOCH" --save_interval 250 \
    --seed 0 --device cuda:0 $RESUME_ARGS
echo "=== $NAME train END $(date) exit=$? ==="
