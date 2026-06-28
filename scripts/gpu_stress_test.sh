#!/bin/bash
# Self-contained controlled crash-reproduction / fix-validation test.
# Compresses the 4-freeze conditions into ~10 min: repeated CUDA context create/destroy
# (fresh process per cycle) + sustained matmul bursts + load<->idle transitions.
# Records its OWN fsync'd telemetry (so a freeze leaves a forensic trace) — no external
# black-box needed. Launch detached:  setsid bash scripts/gpu_stress_test.sh > stress.log 2>&1 &
export PYTHONPATH=.
CYCLES=${1:-70}; SIZE=${2:-6144}; DUR=${3:-6}
BB=./log/gpu_blackbox.csv
mkdir -p "$(dirname "$BB")"

# inline telemetry recorder (child) — temp/util/power/clk/mem/throttle every 2s, fsync'd
( while true; do
    r=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw,clocks.sm,clocks.mem,memory.used,clocks_throttle_reasons.active --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$r" >> "$BB"
    sync "$BB" 2>/dev/null || sync
    sleep 2
  done ) &
BBPID=$!
trap "kill $BBPID 2>/dev/null" EXIT

echo "=== GPU STRESS TEST START $(date) — $CYCLES cycles, ${SIZE}^2 matmul ${DUR}s, GSP=$(nvidia-smi -q 2>/dev/null | grep -m1 'GSP Firmware' | awk '{print $NF}') ==="
for i in $(seq 1 "$CYCLES"); do
  printf '[cycle %d/%d] %s ' "$i" "$CYCLES" "$(date +%H:%M:%S)"
  python scripts/stress_burst.py "$SIZE" "$DUR" || echo "  >>> burst FAILED rc=$?"
  sleep 1.5
done
echo "=== STRESS_SURVIVED: $CYCLES cycles without freeze $(date) ==="
