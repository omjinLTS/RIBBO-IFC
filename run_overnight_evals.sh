#!/bin/bash
# Overnight GPU eval chain: E-eval (EE D=20), C (500-ch CI), D (hybrid). Sequential,
# continue-on-error, each step logged separately. Uses conda python directly.
PY=/home/myoungjin/miniconda3/envs/RIBBO/bin/python
export PYTHONPATH=. WANDB_MODE=disabled OMP_NUM_THREADS=13 MKL_NUM_THREADS=13
SP=/tmp/claude-1000/-home-myoungjin-01-Research-01-Projects-RIBBO/1d24f5c6-1aef-4a4d-a86d-8c8be54b8907/scratchpad

run() { echo "### $1 START $(date '+%H:%M:%S')"; eval "$2" > "$SP/$1.log" 2>&1; echo "### $1 END $(date '+%H:%M:%S') exit=$?"; }

# E-eval: EE D=20 curve + headline
run eval_ee_d20 "$PY scripts/eval_ee_d20.py --n 200 --n_restart 40"

# C: 500-channel CI on the two key results
run ci_ee_d10 "$PY scripts/eval_ci.py --run_dir ./log/ifc/dt-ifc-ee-d10-3ba --ssid IFCEED10 --data_dir ./data/generated_data/ifc_d10_ee --cache_dir ./cache/ifc_d10_ee/ --x_dim 10 --reward EE --n 500 --tag ee_d10"
run ci_se_d20 "$PY scripts/eval_ci.py --run_dir ./log/ifc/dt-ifc-d20-4ba --ssid IFCSED20 --data_dir ./data/generated_data/ifc_d20_4bafilter --cache_dir ./cache/ifc_d20_4bafilter/ --x_dim 20 --reward SE --n 500 --tag se_d20_4ba"

# D: hybrid RIBBO + local polish
run hybrid_ee_d10 "$PY scripts/eval_hybrid.py --run_dir ./log/ifc/dt-ifc-ee-d10-3ba --ssid IFCEED10 --data_dir ./data/generated_data/ifc_d10_ee --cache_dir ./cache/ifc_d10_ee/ --x_dim 10 --reward EE --n 150 --tag ee_d10"
run hybrid_se_d20 "$PY scripts/eval_hybrid.py --run_dir ./log/ifc/dt-ifc-d20-4ba --ssid IFCSED20 --data_dir ./data/generated_data/ifc_d20_4bafilter --cache_dir ./cache/ifc_d20_4bafilter/ --x_dim 20 --reward SE --n 150 --tag se_d20_4ba"

echo "### ALL EVALS DONE $(date '+%H:%M:%S')"
