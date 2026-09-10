#!/usr/bin/env bash
# scripts/train_e2e_level_mt_8gpu.sh — max-CHR 多任务 e2e(concat)8 卡 DDP。
# 每进程单卡(torchrun + train_e2e.py 的 CUDA_VISIBLE_DEVICES 重映射),
# rank 错峰 MLVLA_DDP_STAGGER_S=420s 防 host-RAM 启动尖峰 OOM(503GB 机器
# 4 并发尖峰曾 OOM-kill;8 rank 必须错峰)。有效 batch 8*8=64。
# 用法: nohup bash scripts/train_e2e_level_mt_8gpu.sh > <out>/ddp.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=/data2/zhy/meta_lora_offload/hypernet/e2e_concat_level_mt
mkdir -p "$OUT"

# rank0 立即启动,rank r 延迟 r*420s(7 档错峰共 49min,一次性墙钟);
# NCCL 进程组超时 60min 已在 train_bridge 内设,覆盖错峰后的首个集合通信。
MLVLA_DDP_STAGGER_S=420 torchrun --nproc_per_node=8 --master_port=29715 \
  scripts/train_e2e.py \
  --config configs/hypernet_e2e_concat_level_mt.yaml \
  --variant concat \
  --output "$OUT" \
  --ddp > "$OUT/train_rank0.log" 2>&1
