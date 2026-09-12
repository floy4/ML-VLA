#!/usr/bin/env bash
# scripts/train_e2e_level_mt_fixlight2_8gpu.sh — fixlight2 加权续训(方案A升级):
# warm-start fixlight step5000.pt + 极暗伪域 lighting_L3__deep(加 3x 权,
# 训练池=bank train split 90 eps)+ lighting_L3 2x。8 卡 DDP,rank 错峰
# 420s,有效 batch 8*8=64。
# 用法: nohup conda run --no-capture-output -n openvla bash scripts/train_e2e_level_mt_fixlight2_8gpu.sh > <out>/ddp.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=/data2/zhy/meta_lora_offload/hypernet/e2e_concat_level_mt_fixlight2
INIT=/data2/zhy/meta_lora_offload/hypernet/e2e_concat_level_mt_fixlight/step5000.pt
mkdir -p "$OUT"

MLVLA_DDP_STAGGER_S=420 torchrun --nproc_per_node=8 --master_port=29719 \
  scripts/train_e2e.py \
  --config configs/hypernet_e2e_concat_level_mt_fixlight2.yaml \
  --variant concat \
  --output "$OUT" \
  --init-checkpoint "$INIT" \
  --ddp > "$OUT/train_rank0.log" 2>&1
