#!/bin/bash
# Generate BotorchBO + Vizier IFC D=3 trajectories that were missing
# from the original 120-file run. Sequential to avoid GP/GPU contention.
export PYTHONPATH=/home/myoungjin/01_Research/01_Projects/RIBBO

total=0
for designer in BotorchBO Vizier; do
    for task in $(seq 0 19); do
        out="data/generated_data/ifc/seed0/${designer}_IFCSE_${task}_0.json"
        if [ -f "$out" ]; then
            echo "skip (exists): $out"
            continue
        fi
        echo "[$(date +%H:%M:%S)] generating $out"
        /home/myoungjin/miniconda3/envs/RIBBO/bin/python data_gen/data_gen_main.py \
            --problem=ifc \
            --designer=$designer \
            --search_space_id=IFCSE \
            --dataset_id=$task \
            --out_name=$out \
            --length=300 \
            --seed=0
        if [ $? -ne 0 ]; then
            echo "FAILED: $out"
        fi
        total=$((total+1))
    done
done

echo ""
echo "Generated $total new files"
echo "Total IFC seed0 files: $(ls data/generated_data/ifc/seed0/ | wc -l) (expected 160)"
echo "Per-designer breakdown:"
ls data/generated_data/ifc/seed0/ | awk -F'_' '{print $1}' | sort | uniq -c
