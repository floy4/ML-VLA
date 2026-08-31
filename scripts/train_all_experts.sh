#!/bin/bash
set -euo pipefail

# Train all 10 libero_goal experts on PI05_BASE.
#
# Strategy: single-GPU per task. Single-GPU batch_size=8 is stable.
# Multi-GPU DDP (4 GPUs, batch=16, fsdp_devices=1) also works but
# single-GPU-per-task has higher throughput for a 10-task sweep
# (8 parallel x 5.5h vs 2 parallel x 3.6h x 5 batches).
#
# Concurrency: NUM_GPUS workers pull tasks from a shared FIFO queue
# (file-based flock). Phase 1 (main training) then Phase 2 (short polish).
#
# Usage: bash scripts/train_all_experts.sh
# Env:   MLVLA_TRAIN_TASKS="task1;task2;..." to override the task list,
#        MLVLA_PERTURBED_DATASET_ROOT=<cond dataset root> for condition experts.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=${MLVLA_OPENVLA_PYTHON:-/home/zhy/miniconda3/envs/openvla/bin/python}
OUTPUT_DIR=${MLVLA_EXPERT_OUTPUT:-/data2/zhy/models/wizard/experts}
BASE_CKPT=${MLVLA_BASE_CKPT:-/data4/zhy/models/openpi-assets/checkpoints/pi05_base}
LOG_DIR="${OUTPUT_DIR}/logs"
QUEUE_FILE="${LOG_DIR}/.task_queue.$$"
LOCK_FILE="${LOG_DIR}/.task_lock.$$"

export HF_LEROBOT_HOME=${MLVLA_LEROBOT_HOME:-/data2/zhy}
export HF_HOME=${MLVLA_HF_CACHE:-/data6/zhy/hf_cache}
export HF_DATASETS_CACHE=${MLVLA_HF_DATASETS_CACHE:-/data2/zhy/hf_datasets_cache}
export PYTHONPATH="${REPO_ROOT}/src"
export TMPDIR=${MLVLA_TMPDIR:-/data2/tmp}
export XLA_PYTHON_CLIENT_GPU_ALLOCATOR="bin-boost"
export JAX_ENABLE_X64=False

mkdir -p "${LOG_DIR}"
rm -f "${QUEUE_FILE}" "${LOCK_FILE}"

TASKS=(
    "turn on the stove"
    "put the bowl on top of the cabinet"
    "put the bowl on the stove"
    "put the bowl on the plate"
    "put the wine bottle on the rack"
    "open the top drawer and put the bowl inside"
    "put the cream cheese in the bowl"
    "put the wine bottle on top of the cabinet"
    "push the plate to the front of the stove"
    "open the middle drawer of the cabinet"
)
if [ -n "${MLVLA_TRAIN_TASKS:-}" ]; then
    IFS=';' read -r -a TASKS <<< "${MLVLA_TRAIN_TASKS}"
fi

PHASE1_STEPS=${MLVLA_PHASE1_STEPS:-10000}
PHASE2_STEPS=${MLVLA_PHASE2_STEPS:-500}
BATCH_SIZE=${MLVLA_BATCH_SIZE:-8}
NUM_GPUS=${MLVLA_NUM_GPUS:-8}

printf '%s\n' "${TASKS[@]}" > "${QUEUE_FILE}"

run_next_task() {
    local gpu_id=$1
    while true; do
        # Atomic claim of next task: open FD 9 on the lock file, flock it,
        # read the first line, drop it from the queue, release.
        exec 9>"${LOCK_FILE}"
        flock 9
        local task
        task=$(head -n 1 "${QUEUE_FILE}" 2>/dev/null || true)
        if [ -n "${task}" ]; then
            sed -i "1d" "${QUEUE_FILE}" 2>/dev/null
        fi
        flock -u 9
        exec 9>&-

        if [ -z "${task}" ]; then
            break  # queue empty
        fi

        # Don't let a single task failure kill the worker (set -e is on globally).
        if ! _run_one_task "${gpu_id}" "${task}"; then
            echo "[GPU ${gpu_id} $(date '+%H:%M:%S')] TASK FAILED: ${task} — continuing to next"
        fi
    done
}

_run_one_task() {
    local gpu_id=$1
    local task=$2
    local hash=$(echo -n "${task}" | md5sum | cut -c1-8)
    local expert_dir="${OUTPUT_DIR}/expert_${hash}"
    local p1_dir="${expert_dir}/phase1"
    local p2_dir="${expert_dir}/phase2"
    local p1_log="${LOG_DIR}/expert_${hash}_phase1.log"
    local p2_log="${LOG_DIR}/expert_${hash}_phase2.log"

    if [ ! -f "${p1_dir}/final/params.canonical.npz" ]; then
        echo "[GPU ${gpu_id} $(date '+%H:%M:%S')] START Phase 1: ${task}"
        CUDA_VISIBLE_DEVICES=${gpu_id} ${PYTHON} "${REPO_ROOT}/scripts/train_expert.py" \
            --task "${task}" \
            --phase 1 --num_steps ${PHASE1_STEPS} \
            --output_dir "${OUTPUT_DIR}" \
            --base_checkpoint "${BASE_CKPT}" \
            --batch_size ${BATCH_SIZE} \
            --gpu "${gpu_id}" \
            > "${p1_log}" 2>&1
        echo "[GPU ${gpu_id} $(date '+%H:%M:%S')] DONE Phase 1: ${task}"
    else
        echo "[GPU ${gpu_id}] SKIP Phase 1 (already complete): ${task}"
    fi

    echo "[GPU ${gpu_id} $(date '+%H:%M:%S')] START Phase 2: ${task}"
    CUDA_VISIBLE_DEVICES=${gpu_id} ${PYTHON} "${REPO_ROOT}/scripts/train_expert.py" \
        --task "${task}" \
        --phase 2 --num_steps ${PHASE2_STEPS} \
        --output_dir "${OUTPUT_DIR}" \
        --base_checkpoint "${BASE_CKPT}" \
        --resume_state "${p1_dir}/final/resume_state" \
        --batch_size ${BATCH_SIZE} \
        --gpu "${gpu_id}" \
        > "${p2_log}" 2>&1
    echo "[GPU ${gpu_id} $(date '+%H:%M:%S')] DONE Phase 2: ${task}"
}

export -f run_next_task
export OUTPUT_DIR BASE_CKPT LOG_DIR PHASE1_STEPS PHASE2_STEPS BATCH_SIZE PYTHON QUEUE_FILE LOCK_FILE REPO_ROOT

echo "[$(date)] Launching ${NUM_GPUS} GPU workers for ${#TASKS[@]} tasks"

pids=()
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    run_next_task ${gpu} &
    pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        echo "[$(date)] Worker ${pid} FAILED"
        fail=1
    fi
done

rm -f "${QUEUE_FILE}" "${LOCK_FILE}" "${LOCK_FILE}.pid"

if [ ${fail} -eq 0 ]; then
    echo "[$(date)] ALL EXPERTS TRAINED"
else
    echo "[$(date)] SOME TASKS FAILED — check logs in ${LOG_DIR}"
    exit 1
fi
