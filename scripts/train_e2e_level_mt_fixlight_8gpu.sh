#!/usr/bin/env bash
# scripts/train_e2e_level_mt_fixlight_8gpu.sh — fixlight 续训(方案A):
# warm-start e2e_concat_level_mt/step12000.pt + lighting_L3 第 9 训练域。
# 8 卡 DDP,rank 错峰 420s 防 host-RAM 启动尖峰 OOM,有效 batch 8*8=64。
# 用法: nohup bash scripts/train_e2e_level_mt_fixlight_8gpu.sh > <out>/ddp.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=/data2/zhy/meta_lora_offload/hypernet/e2e_concat_level_mt_fixlight
INIT=/data2/zhy/meta_lora_offload/hypernet/e2e_concat_level_mt/step12000.pt
mkdir -p "$OUT"

MLVLA_DDP_STAGGER_S=420 torchrun --nproc_per_node=8 --master_port=29717 \
  scripts/train_e2e.py \
  --config configs/hypernet_e2e_concat_level_mt_fixlight.yaml \
  --variant concat \
  --output "$OUT" \
  --init-checkpoint "$INIT" \
  --ddp > "$OUT/train_rank0.log" 2>&1
