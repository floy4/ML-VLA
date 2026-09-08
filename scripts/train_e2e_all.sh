#!/usr/bin/env bash
# scripts/train_e2e_all.sh — 三变体三卡并行;输出全部落 /data2
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=/data2/zhy/meta_lora_offload/hypernet
mkdir -p "$OUT"/e2e_film "$OUT"/e2e_film_dropout "$OUT"/e2e_concat
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n openvla python scripts/train_e2e.py \
  --config configs/hypernet_e2e_film.yaml --variant film \
  --output "$OUT/e2e_film" > "$OUT/e2e_film/train.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n openvla python scripts/train_e2e.py \
  --config configs/hypernet_e2e_film_dropout.yaml --variant film_dropout \
  --output "$OUT/e2e_film_dropout" > "$OUT/e2e_film_dropout/train.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n openvla python scripts/train_e2e.py \
  --config configs/hypernet_e2e_concat.yaml --variant concat \
  --output "$OUT/e2e_concat" > "$OUT/e2e_concat/train.log" 2>&1 &
wait
