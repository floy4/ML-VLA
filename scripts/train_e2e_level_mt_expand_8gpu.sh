#!/usr/bin/env bash
# scripts/train_e2e_level_mt_expand_8gpu.sh — 扩域续训: warm-start fixlight2
# step5000.pt (12 域均值 0.654 历史最佳) + add_object L1/L2/L3 + clean 原始
# LIBERO,共 14 训练域;权重配方沿用 fixlight2(L3 x2, deep x3),新域 1.0。
# 8 卡 DDP + dlpack fast path(2026-09-17 八卡验证 PASS: bit-sync 通过,
# 50 步有限,-17.6% s/it),rank 错峰 420s,有效 batch 8*8=64,20k 步(用户设定)。
# 用法: nohup conda run --no-capture-output -n openvla bash scripts/train_e2e_level_mt_expand_8gpu.sh > <out>/ddp.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=/data2/zhy/meta_lora_offload/hypernet/e2e_concat_level_mt_expand
INIT=/data2/zhy/meta_lora_offload/hypernet/e2e_concat_level_mt_fixlight2/step5000.pt
mkdir -p "$OUT"

MLVLA_FAST_PATH=1 \
MLVLA_DDP_STAGGER_S=420 \
MLVLA_DDP_NCCL_TIMEOUT_MIN=120 \
torchrun --nproc_per_node=8 --master_port=29731 \
  scripts/train_e2e.py \
  --config configs/hypernet_e2e_concat_level_mt_expand.yaml \
  --variant concat \
  --output "$OUT" \
  --init-checkpoint "$INIT" \
  --ddp > "$OUT/train_rank0.log" 2>&1
