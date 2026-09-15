#!/usr/bin/env bash
# Task 6: three pose-scheme arms in parallel (single-task e2e, frozen protocol).
#
#   GPU0 vggt      — VGGT CAMERA-token layer-23 pose (cond 1024+2048)
#   GPU1 raymap    — Plücker ray-map DINO pose (cond 1024+2048)
#   GPU2 zeropose  — zeros(7) pose ablation (cond 1024+7)
#
# Protocol (identical to hypernet_e2e_concat.yaml baseline except pose channel):
#   max_steps 60000 stays as the cosine/warmup/val/save anchor; the training
#   LOOP is capped at 40000 via --max-steps-override (the concat@40k baseline
#   row is the same 60k-schedule checkpoint at step 40000).
# MLVLA_FAST_PATH=1: controller ruling — A/B proved the fast path bit-exact
#   under pinned compile; production speedup 14.6% (2.96->2.53 s/step).
set -u
cd "$(dirname "$0")/.."

BASE=/data2/zhy/meta_lora_offload/hypernet

for spec in 0:vggt 1:raymap 2:zeropose; do
  gpu=${spec%%:*}; scheme=${spec#*:}
  out=$BASE/e2e_concat_${scheme}
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$gpu MLVLA_FAST_PATH=1 \
    conda run --no-capture-output -n openvla \
    python scripts/train_e2e.py \
      --config configs/hypernet_e2e_concat_${scheme}.yaml \
      --variant concat \
      --max-steps-override 40000 \
      --output "$out" > "$out/train.log" 2>&1 &
  echo "launched ${scheme} on GPU${gpu} (pid $!) -> $out"
done
wait
