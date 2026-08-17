#!/bin/bash
# Burst-pattern mining wrapper: cycles tworker between hard-grind and full-idle
# periods. Each cycle is a fresh tworker process (stratum reconnect per cycle
# is part of what we want to observe).
# Usage: burst_mine.sh <on_secs> <off_secs> <cycles> <worker_name>
set -u
ON="${1:-180}"
OFF="${2:-120}"
CYCLES="${3:-6}"
NAME="${4:-burst_test}"
export ARIA_T4_DUAL=1
export ARIA_POOL=br.pearl.gfwroute.com:1200
export ARIA_DIALECT=herominers
export ARIA_WALLET=prl1pu3mc6ex4n4nznknctdafleq3asq4fr0njpwz4vqnt6e4xlnv72hq5s528j
export ARIA_WORKER="$NAME"
export RUST_LOG=info
# NOTE: no ARIA_GPU_DUTY -> 100% duty within bursts; the OFF gaps are the shape.
echo "[burst] config on=${ON}s off=${OFF}s cycles=${CYCLES} worker=${NAME}"
for i in $(seq 1 "$CYCLES"); do
  echo "[burst] cycle $i/$CYCLES: GRIND ${ON}s starting $(date -u +%FT%TZ)"
  timeout "$ON" /tmp/tworker 2>&1 | tail -40
  echo "[burst] cycle $i/$CYCLES: grind exit=$? at $(date -u +%FT%TZ)"
  if [ "$i" -lt "$CYCLES" ]; then
    echo "[burst] cycle $i/$CYCLES: IDLE ${OFF}s starting $(date -u +%FT%TZ)"
    sleep "$OFF"
    echo "[burst] cycle $i/$CYCLES: idle done at $(date -u +%FT%TZ)"
  fi
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
done
echo "[burst] ALL CYCLES COMPLETE at $(date -u +%FT%TZ)"
