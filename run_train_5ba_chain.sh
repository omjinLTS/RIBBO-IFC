#!/bin/bash
# Build 5BA-filter symlink datasets, then train: (B) SE D=20 5BA-filter, (C) EE D=10 5BA-filter.
# Embeds fsync telemetry + auto-resume + 250-ep checkpoints. Run after run_datagen_additions.sh.
export PYTHONPATH=.
export WANDB_MODE=disabled
export OMP_NUM_THREADS=13 MKL_NUM_THREADS=13 NUMEXPR_NUM_THREADS=13
SP=/tmp/claude-1000/-home-myoungjin-01-Research-01-Projects-RIBBO/5a3b2b50-c7a4-4488-add4-3c82eb5c6b2a/scratchpad
BB=./log/gpu_blackbox.csv
mkdir -p "$SP" "$(dirname "$BB")"

( while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw,clocks.sm,clocks.mem,memory.used,clocks_throttle_reasons.active --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')" >> "$BB"
    sync "$BB" 2>/dev/null; sleep 2
  done ) & BBPID=$!
trap "kill $BBPID 2>/dev/null" EXIT

build_symlinks() { # base filter ssid
  mkdir -p "data/generated_data/$2/seed0"
  for des in HillClimbing RegularizedEvolution EagleStrategy CMAES BotorchBO; do
    for f in data/generated_data/$1/seed0/${des}_${3}_*.json; do
      [ -e "$f" ] && ln -sf "$(realpath "$f")" "data/generated_data/$2/seed0/$(basename "$f")"
    done
  done
  echo "  $2: $(ls data/generated_data/$2/seed0/ | sed -E 's/_IFC.*//' | sort | uniq -c | tr '\n' ' ')"
}
echo "=== build 5BA-filter symlink dirs $(date) ==="
build_symlinks ifc_d20    ifc_d20_5bafilter    IFCSED20
build_symlinks ifc_d10_ee ifc_d10_ee_5bafilter IFCEED10

train() { # config name id log
  local rundir="./log/ifc/$2" resume="" L EP
  if [ -d "$rundir" ]; then
    L=$(ls "$rundir"/*/ckpt/*.ckpt 2>/dev/null | awk -F/ '{n=$NF;gsub(/\.ckpt/,"",n);print n,$0}' | sort -n | tail -1 | cut -d' ' -f2-)
    EP=$(basename "${L:-0.ckpt}" .ckpt)
    [ -n "$L" ] && [ "$EP" -gt 1 ] 2>/dev/null && resume="--load_ckpt $L --start_epoch $EP"
  fi
  echo "=== $2 START $(date) resume='${resume:-none}' ==="
  python -u scripts/run_dt.py --config "$1" --name "$2" --id "$3" \
    --embed_dim 256 --num_heads 8 --num_layers 12 --pos_encoding embed \
    --input_seq_len 50 --max_input_seq_len 300 \
    --num_epoch 3000 --batch_size 64 --eval_interval 3000 --save_interval 250 \
    --seed 0 --device cuda:0 $resume > "$4" 2>&1
  echo "=== $2 END $(date) exit=$? ==="
}

train scripts/configs/dt/ifc_d20_5bafilter.py    dt-ifc-d20-5ba     IFCSED20 "$SP/train_d20_5ba.log"
train scripts/configs/dt/ifc_d10_ee_5bafilter.py dt-ifc-ee-d10-5ba  IFCEED10 "$SP/train_ee_d10_5ba.log"
echo "=== 5BA CHAIN COMPLETE $(date) ==="
