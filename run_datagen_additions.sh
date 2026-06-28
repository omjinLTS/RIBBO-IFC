#!/bin/bash
# Add the curated-pool BAs (RegularizedEvolution, EagleStrategy, BotorchBO) to:
#   - ifc_d20    (IFCSED20, SE)   — already has Random/HillClimbing/CMAES
#   - ifc_d10_ee (IFCEED10, EE)   — already has Random/HillClimbing/CMAES
# Fast BAs in parallel; BotorchBO sequential (GP, avoid contention). Skips existing files.
export PYTHONPATH=.

gen() { # dir ssid designer task
  local out="data/generated_data/$1/seed0/$3_$2_$4_0.json"
  [ -f "$out" ] && { echo "skip $out"; return; }
  python data_gen/data_gen_main.py --problem=ifc --designer=$3 \
    --search_space_id=$2 --dataset_id=$4 --out_name=$out --length=300 --seed=0 2>/dev/null
}
export -f gen

echo "=== fast BAs (RegEvol, Eagle) in parallel $(date) ==="
MAX_PROC=12; fifo="/tmp/$$.fifo"; mkfifo $fifo; exec 7<>$fifo; rm -f $fifo
for ((i=0;i<MAX_PROC;i++)); do echo >&7; done
for spec in "ifc_d20 IFCSED20" "ifc_d10_ee IFCEED10"; do
  set -- $spec
  for des in RegularizedEvolution EagleStrategy; do
    for t in $(seq 0 19); do
      read -u7; { gen "$1" "$2" "$des" "$t"; echo >&7; } &
    done
  done
done
wait
echo "=== BotorchBO sequential (GP) $(date) ==="
for spec in "ifc_d20 IFCSED20" "ifc_d10_ee IFCEED10"; do
  set -- $spec
  for t in $(seq 0 19); do
    printf '[%s] botorch %s t%s ' "$(date +%H:%M:%S)" "$1" "$t"; gen "$1" "$2" "BotorchBO" "$t" && echo ok
  done
done
echo "=== DONE $(date) ==="
echo "ifc_d20:    $(ls data/generated_data/ifc_d20/seed0/ | sed -E 's/_IFC.*//' | sort | uniq -c | tr '\n' ' ')"
echo "ifc_d10_ee: $(ls data/generated_data/ifc_d10_ee/seed0/ | sed -E 's/_IFC.*//' | sort | uniq -c | tr '\n' ' ')"
