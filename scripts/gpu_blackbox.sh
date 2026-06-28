#!/bin/bash
# GPU/CPU "black-box flight recorder" — append timestamped telemetry every 2s and
# fsync each line so the LAST readings survive a hard freeze (the kernel itself logs
# nothing on these silent wedges, so this file is our only forensic trace).
#
# Usage:  bash scripts/gpu_blackbox.sh [logfile]
# Stop:   kill the process (or it dies with the session). Keep it running for the
#         entire duration of any GPU training.
#
# After a crash, inspect the TAIL of the logfile: a temp/power spike or a throttle
# flag in the final rows points to thermal/power; a clean cutoff mid-normal-load
# points to a driver/PCIe hard-hang.

LOG="${1:-./log/gpu_blackbox.csv}"
mkdir -p "$(dirname "$LOG")"
if [ ! -s "$LOG" ]; then
  echo "timestamp,temp_C,util_%,power_W,sm_MHz,mem_MHz,memUsed_MiB,throttle" >> "$LOG"
fi
echo "[blackbox] logging to $LOG every 2s (PID $$). Ctrl-C / kill to stop."
while true; do
  ts=$(date '+%Y-%m-%d %H:%M:%S')
  row=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw,clocks.sm,clocks.mem,memory.used,clocks_throttle_reasons.active \
        --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  echo "$ts,$row" >> "$LOG"
  # best-effort flush so the final lines reach disk before a freeze
  sync "$LOG" 2>/dev/null || sync
  sleep 2
done
