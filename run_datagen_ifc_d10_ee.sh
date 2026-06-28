#!/bin/bash
# IFC EE (energy efficiency) D=10 data generation, 3BA pool matched to the SE D=10 run
# (Random + HillClimbing + CMAES, 20 channels, 300 steps each => 60 trajectories).
export PYTHONPATH=.

mkdir -p data/generated_data/ifc_d10_ee/seed0
MAX_PROC=12
fifo_name="/tmp/$$.fifo"
mkfifo $fifo_name
exec 7<>$fifo_name
for ((i=0; i<MAX_PROC; i++)); do echo $i; done >&7

for designer in Random HillClimbing CMAES; do
    for task in $(seq 0 19); do
        read -u7 proc_id
        {
            python data_gen/data_gen_main.py \
                --problem=ifc \
                --designer=$designer \
                --search_space_id=IFCEED10 \
                --dataset_id=$task \
                --out_name=data/generated_data/ifc_d10_ee/seed0/${designer}_IFCEED10_${task}_0.json \
                --length=300 \
                --seed=0 2>/dev/null
            echo >&7 $proc_id
        } &
    done
done

wait
exec 7>&-
rm -f $fifo_name
echo "Done: $(ls data/generated_data/ifc_d10_ee/seed0/ 2>/dev/null | wc -l)/60 files"
