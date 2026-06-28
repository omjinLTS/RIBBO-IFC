#!/bin/bash
# IFC EE D=10, 3BA pool, LENGTH=1000 (horizon-1000 retraining for fair extended-budget comparison).
# Same 3BA (Random, HillClimbing, CMAES) x 20 channels, but 1000 steps/trajectory.
export PYTHONPATH=.

OUTDIR=data/generated_data/ifc_d10_ee_h1000/seed0
mkdir -p "$OUTDIR"
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
                --out_name=${OUTDIR}/${designer}_IFCEED10_${task}_0.json \
                --length=1000 \
                --seed=0 2>/dev/null
            echo >&7 $proc_id
        } &
    done
done

wait
exec 7>&-
rm -f $fifo_name
echo "Done: $(ls $OUTDIR 2>/dev/null | wc -l)/60 files (length=1000)"
