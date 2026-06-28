#!/bin/bash
# Resume-aware chain for the work the 6/23 crash interrupted: D=10 EE then D=20 SE.
# Auto-resumes each run from its latest checkpoint (save_interval=250 => <=12.5min loss
# on a crash). A global black-box recorder is expected to be running separately.
export PYTHONPATH=.
export WANDB_MODE=disabled
export OMP_NUM_THREADS=13 MKL_NUM_THREADS=13 NUMEXPR_NUM_THREADS=13
SP=/tmp/claude-1000/-home-myoungjin-01-Research-01-Projects-RIBBO/5a3b2b50-c7a4-4488-add4-3c82eb5c6b2a/scratchpad
mkdir -p "$SP"

train_one() {
  local config=$1 name=$2 id=$3 nep=$4 log=$5
  local rundir="./log/ifc/$name" latest ep resume=""
  if [ -d "$rundir" ]; then
    latest=$(ls "$rundir"/*/ckpt/*.ckpt 2>/dev/null | awk -F/ '{n=$NF;gsub(/\.ckpt/,"",n);print n,$0}' | sort -n | tail -1 | cut -d' ' -f2-)
    ep=$(basename "${latest:-0.ckpt}" .ckpt)
    if [ -n "$latest" ] && [ "$ep" -gt 1 ] 2>/dev/null; then resume="--load_ckpt $latest --start_epoch $ep"; fi
  fi
  echo "=== $name START $(date) resume='${resume:-none}' target=${nep}ep ==="
  python -u scripts/run_dt.py --config "$config" --name "$name" --id "$id" \
    --embed_dim 256 --num_heads 8 --num_layers 12 --pos_encoding embed \
    --input_seq_len 50 --max_input_seq_len 300 \
    --num_epoch "$nep" --batch_size 64 --eval_interval "$nep" --save_interval 250 \
    --seed 0 --device cuda:0 $resume > "$log" 2>&1
  echo "=== $name END $(date) exit=$? ==="
}

train_one scripts/configs/dt/ifc_d10_ee.py dt-ifc-ee-d10-3ba IFCEED10 3000 "$SP/train_ee_d10.log"
train_one scripts/configs/dt/ifc_d20.py    dt-ifc-d20-3ba    IFCSED20 3000 "$SP/train_d20.log"
echo "=== CHAIN COMPLETE $(date) ==="
