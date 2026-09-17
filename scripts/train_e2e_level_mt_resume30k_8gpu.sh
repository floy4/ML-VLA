#!/usr/bin/env bash
# scripts/train_e2e_level_mt_resume30k_8gpu.sh — Resume from step12000.pt
# (8-domain L1+L2) and continue to 30k steps. 8-GPU DDP with stagger 420s.
# Output: /data2/zhy/meta_lora_offload/hypernet/e2e_concat_level_mt_resume30k/
# Usage: nohup bash scripts/train_e2e_level_mt_resume30k_8gpu.sh > <out>/ddp.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=/data2/zhy/meta_lora_offload/hypernet/e2e_concat_level_mt_resume30k
mkdir -p "$OUT"

INIT_CKPT=/data2/zhy/meta_lora_offload/hypernet/e2e_concat_level_mt/step12000.pt

MLVLA_DDP_STAGGER_S=420 torchrun --nproc_per_node=8 --master_port=29718 \
  scripts/train_e2e.py \
  --config configs/hypernet_e2e_concat_level_mt_resume30k.yaml \
  --variant concat \
  --output "$OUT" \
  --init-checkpoint "$INIT_CKPT" \
  --ddp > "$OUT/train_rank0.log" 2>&1
