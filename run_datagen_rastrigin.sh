#!/bin/bash
export PYTHONPATH=.

for designer in Random HillClimbing; do
(
  for task in $(seq 2 19); do
    python data_gen/data_gen_main.py \
      --problem=synthetic \
      --designer=$designer \
      --search_space_id=Rastrigin \
      --dataset_id=$task \
      --out_name=data/generated_data/synthetic/seed0/${designer}_Rastrigin_${task}_0.json \
      --length=300 \
      --seed=0 2>/dev/null
  done
  echo "$designer done"
) &
done

for task in $(seq 3 19); do
  python data_gen/data_gen_main.py \
    --problem=synthetic \
    --designer=CMAES \
    --search_space_id=Rastrigin \
    --dataset_id=$task \
    --out_name=data/generated_data/synthetic/seed0/CMAES_Rastrigin_${task}_0.json \
    --length=300 \
    --seed=0 2>/dev/null
done
echo "CMAES done"
wait
echo "All done: $(ls data/generated_data/synthetic/seed0/ | wc -l) files"
