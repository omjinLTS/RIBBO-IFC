#!/bin/bash
# Add RegEvol/Eagle/BotorchBO to existing IFC D=10 dataset (currently has 3BA only).
# RegEvol + Eagle: parallel (fast). BotorchBO: sequential (avoid contention).
export PYTHONPATH=/home/myoungjin/01_Research/01_Projects/RIBBO
PYBIN=/home/myoungjin/miniconda3/envs/RIBBO/bin/python
DATA_DIR=data/generated_data/ifc_d10/seed0

# ── parallel: RegEvol + Eagle ──
MAX_PROC=12
fifo=/tmp/$$.fifo
mkfifo $fifo
exec 7<>$fifo
for ((i=0; i<MAX_PROC; i++)); do echo $i; done >&7

for designer in RegularizedEvolution EagleStrategy; do
    for task in $(seq 0 19); do
        out=${DATA_DIR}/${designer}_IFCSED10_${task}_0.json
        if [ -f "$out" ]; then
            echo "skip (exists): $out"
            continue
        fi
        read -u7 proc
        {
            $PYBIN data_gen/data_gen_main.py \
                --problem=ifc --designer=$designer \
                --search_space_id=IFCSED10 --dataset_id=$task \
                --out_name=$out --length=300 --seed=0 2>/dev/null
            echo >&7 $proc
        } &
    done
done
wait
exec 7>&-
rm -f $fifo
echo "[$(date +%H:%M:%S)] RegEvol+Eagle done"

# ── sequential: BotorchBO (GP scales with history, take care) ──
for task in $(seq 0 19); do
    out=${DATA_DIR}/BotorchBO_IFCSED10_${task}_0.json
    if [ -f "$out" ]; then
        echo "skip (exists): $out"
        continue
    fi
    echo "[$(date +%H:%M:%S)] generating $out"
    $PYBIN data_gen/data_gen_main.py \
        --problem=ifc --designer=BotorchBO \
        --search_space_id=IFCSED10 --dataset_id=$task \
        --out_name=$out --length=300 --seed=0
    if [ $? -ne 0 ]; then
        echo "FAILED: $out"
    fi
done

echo ""
echo "Total: $(ls $DATA_DIR | wc -l) files (expected 120)"
ls $DATA_DIR | awk -F'_' '{print $1}' | sort | uniq -c
