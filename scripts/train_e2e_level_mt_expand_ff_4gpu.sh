#!/usr/bin/env bash
# First-frame evidence retrain: warm-start expand step20000, first-frame cache,
# 4-GPU DDP (effective batch 32), 30k steps (user-set; ckpt every 1000).
# Usage: nohup bash scripts/train_e2e_level_mt_expand_ff_4gpu.sh > <out>/launch.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=/data2/zhy/meta_lora_offload/hypernet/e2e_concat_level_mt_expand_firstframe
INIT=/data2/zhy/meta_lora_offload/hypernet/e2e_concat_level_mt_expand/step20000.pt
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0,2,6,7 \
MLVLA_FAST_PATH=1 \
MLVLA_DDP_STAGGER_S=180 \
MLVLA_DDP_NCCL_TIMEOUT_MIN=120 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=4 --master_port=29733 \
  scripts/train_e2e.py \
  --config configs/hypernet_e2e_concat_level_mt_expand_firstframe.yaml \
  --variant concat \
  --output "$OUT" \
  --init-checkpoint "$INIT" \
  --ddp > "$OUT/train_rank0.log" 2>&1
