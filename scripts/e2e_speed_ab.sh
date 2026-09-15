#!/usr/bin/env bash
# A/B equivalence + speed gate for the e2e train-bridge fast path (Task 4).
#
# Runs the SAME config twice on one GPU — MLVLA_FAST_PATH=0 (original per-tensor
# CPU round-trip path) vs MLVLA_FAST_PATH=1 (dlpack zero-copy flat fast path) —
# capped at --max-steps-override steps (cosine anchoring etc. untouched), then
# compares the per-step loss dumps.
#
# Gate: max relative per-step loss deviation < 1e-4, no NaN, >=25% speedup.
#
# Usage: bash scripts/e2e_speed_ab.sh [GPU_ID]      (default 3; never 0/1/2)
# Env:   STEPS (default 300)
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${1:-3}
STEPS=${STEPS:-300}
CFG=configs/hypernet_e2e_concat.yaml
BASE=/data2/zhy/meta_lora_offload/hypernet

run() {  # run <fast 0|1> <outdir>
  local fast=$1 outdir=$2
  echo "=== MLVLA_FAST_PATH=$fast -> $outdir (GPU $GPU, $STEPS steps) ==="
  # xla_gpu_autotune_level=0: XLA's timing-based autotuner can pick different
  # GEMM algorithms in different processes (observed ~2e-4 relative loss jitter
  # at step 1 with identical inputs/weights), which would break the 1e-4
  # equivalence gate for reasons unrelated to the fast path. Disabling it makes
  # both arms compile deterministically; the speedup measurement stays
  # like-for-like. Production runs may keep autotune on.
  XLA_FLAGS="--xla_gpu_autotune_level=0" MLVLA_FAST_PATH=$fast CUDA_VISIBLE_DEVICES=$GPU \
    conda run --no-capture-output -n openvla \
    python scripts/train_e2e.py --config "$CFG" --variant concat \
      --max-steps-override "$STEPS" --output "$outdir" 2>&1 | tee "$outdir.log"
}

run 0 "$BASE/ab_slow"
run 1 "$BASE/ab_fast"

conda run --no-capture-output -n openvla python - "$BASE" <<'EOF'
import json, math, sys

base = sys.argv[1]

def load(p):
    recs = [json.loads(l) for l in open(p)]
    return {r["step"]: r for r in recs}

slow, fast = load(f"{base}/ab_slow/per_step_loss.jsonl"), load(f"{base}/ab_fast/per_step_loss.jsonl")
assert slow and set(slow) == set(fast), f"step sets differ: {len(slow)} vs {len(fast)}"

nan = any(math.isnan(v["loss"]) for v in list(slow.values()) + list(fast.values()))
worst, worst_step = 0.0, None
for s in sorted(slow):
    a, b = slow[s]["loss"], fast[s]["loss"]
    r = abs(a - b) / max(abs(a), 1e-12)
    if r > worst:
        worst, worst_step = r, s

def steady(d):  # s/step over the last half, skipping step-1 XLA compile
    steps = sorted(d)
    lo, hi = steps[len(steps) // 2], steps[-1]
    return (d[hi]["t"] - d[lo]["t"]) / (hi - lo)

s_slow, s_fast = steady(slow), steady(fast)
print(f"steps={len(slow)}  nan={nan}")
print(f"max_rel_dev={worst:.3e} at step {worst_step}  "
      f"(loss {slow[worst_step]['loss']:.6f} vs {fast[worst_step]['loss']:.6f})")
print(f"s/step steady-state: slow={s_slow:.3f}  fast={s_fast:.3f}  "
      f"speedup={(s_slow - s_fast) / s_slow:.1%}")
ok = worst < 1e-4 and not nan
sp = (s_slow - s_fast) / s_slow
print(f"GATE: {'PASS' if ok else 'FAIL'} (equivalence); "
      f"speedup {sp:.1%} ({'>=25% merge bar met' if sp >= 0.25 else '<25% merge bar NOT met'})")
sys.exit(0 if ok else 1)
EOF
