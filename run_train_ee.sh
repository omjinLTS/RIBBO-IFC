#!/bin/bash
# Chained EE training: D=3 (3BA) then D=10 (3BA), 3000 epochs each.
# Mirrors the SE 3BA training protocol (paper hyperparameters, no tuning).
export PYTHONPATH=.
export WANDB_MODE=disabled
export OMP_NUM_THREADS=13 MKL_NUM_THREADS=13 NUMEXPR_NUM_THREADS=13

LOGDIR=/tmp/claude-1000/-home-myoungjin-01-Research-01-Projects-RIBBO/5a3b2b50-c7a4-4488-add4-3c82eb5c6b2a/scratchpad
mkdir -p "$LOGDIR"

echo "=== EE D=3 training START $(date) ==="
python -u scripts/run_dt.py \
    --config scripts/configs/dt/ifc_ee.py \
    --name dt-ifc-ee-d3-3ba \
    --embed_dim 256 --num_heads 8 --num_layers 12 \
    --pos_encoding embed \
    --input_seq_len 50 --max_input_seq_len 300 \
    --num_epoch 3000 --batch_size 64 \
    --eval_interval 3000 --save_interval 500 \
    --id IFCEE --seed 0 --device cuda:0 \
    > "$LOGDIR/train_ee_d3.log" 2>&1
echo "=== EE D=3 training END $(date) exit=$? ==="

echo "=== EE D=10 training START $(date) ==="
python -u scripts/run_dt.py \
    --config scripts/configs/dt/ifc_d10_ee.py \
    --name dt-ifc-ee-d10-3ba \
    --embed_dim 256 --num_heads 8 --num_layers 12 \
    --pos_encoding embed \
    --input_seq_len 50 --max_input_seq_len 300 \
    --num_epoch 3000 --batch_size 64 \
    --eval_interval 3000 --save_interval 500 \
    --id IFCEED10 --seed 0 --device cuda:0 \
    > "$LOGDIR/train_ee_d10.log" 2>&1
echo "=== EE D=10 training END $(date) exit=$? ==="
