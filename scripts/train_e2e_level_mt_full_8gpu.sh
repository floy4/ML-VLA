#!/usr/bin/env bash
# scripts/train_e2e_level_mt_full_8gpu.sh — 全覆盖 16 域(12 level + 3 add_object
# + clean)60k 步 e2e(concat)8 卡 DDP。每进程单卡(torchrun + train_e2e.py 的
# CUDA_VISIBLE_DEVICES 重映射),rank 错峰 MLVLA_DDP_STAGGER_S=420s 防 host-RAM
# 启动尖峰 OOM。有效 batch 8*8=64,60k 步 ≈ 3 天。
# 用法: nohup bash scripts/train_e2e_level_mt_full_8gpu.sh > <out>/ddp.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=/data2/zhy/meta_lora_offload/hypernet/e2e_concat_level_mt_full
mkdir -p "$OUT"

MLVLA_DDP_STAGGER_S=420 torchrun --nproc_per_node=8 --master_port=29717 \
  scripts/train_e2e.py \
  --config configs/hypernet_e2e_concat_level_mt_full.yaml \
  --variant concat \
  --output "$OUT" \
  --ddp > "$OUT/train_rank0.log" 2>&1
