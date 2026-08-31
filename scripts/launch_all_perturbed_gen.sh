#!/bin/bash
# Launch 9 condition-replay processes across 8 GPUs.
# Each condition is one process (50 demos x 10 tasks).
# The 9th process waits for GPU 0 to free up.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=${MLVLA_LIBERO_PYTHON:-/home/zhy/miniconda3/envs/libero/bin/python}
SCRIPT=$REPO_ROOT/scripts/generate_perturbed_demos.py
LOG_DIR=${MLVLA_LOG_DIR:-$REPO_ROOT/logs/perturbed_gen}
GPU_COUNT=${MLVLA_GPU_COUNT:-8}
mkdir -p "$LOG_DIR"

CONDS=(v1_azimuth30 v2_azimuth60 v3_elev15_zoom125 \
       l1_warm_dim l2_cool_bright l3_directional_low \
       c1_v1l1 c2_v2l2 c3_v3l3)
NUM_CONDS=${#CONDS[@]}

PIDS=()
for i in "${!CONDS[@]}"; do
    COND=${CONDS[$i]}
    GPU=$((i % GPU_COUNT))
    # 9th condition queues after the 1st finishes (FIFO slot reuse).
    if [ $i -ge $GPU_COUNT ]; then
        PREV_PID=${PIDS[$((i - GPU_COUNT))]}
        echo "[$(date +%H:%M:%S)] Waiting on PID $PREV_PID before launching $COND on GPU $GPU"
        wait "$PREV_PID" || true
    fi
    LOG="$LOG_DIR/${COND}.log"
    echo "[$(date +%H:%M:%S)] Launching $COND on GPU $GPU -> $LOG"
    CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl \
        "$PYTHON" "$SCRIPT" --conditions "$COND" > "$LOG" 2>&1 &
    PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "[$(date +%H:%M:%S)] PID $pid failed"
        FAIL=1
    fi
done
exit $FAIL
