# ML-VLA

Meta-LoRA for vision-language-action models: perturbed-condition data collection,
π0.5 LoRA expert training, and a DirectAB meta-network that generates LoRA weights
from environment evidence.

Three pipelines, one package (`mlvla/`, src-layout):

1. **Data collection** (`mlvla.data_gen`) — replay source demos under new camera /
   lighting conditions via LIBERO-plus, writing LeRobot-format parquet datasets.
2. **LoRA experts** (`mlvla.jax` / `mlvla.adapters` / `mlvla.experts` / `mlvla.eval`) —
   rank-16 LoRA fine-tuning of π0.5 (JAX/openpi), orbax→canonical-npz export, and
   closed-loop / train-loss / action-prediction validation.
3. **Meta-network** (`mlvla.meta`) — DirectAB hypernetwork: DINOv3 image evidence +
   view parameters → 458-module LoRA weight deltas; train / eval / export / serve.

## Setup

Three conda environments (the pipelines have incompatible deps — MuJoCo stack vs
JAX/openpi vs torch). Exact exports of the environments used in development are in
[`envs/`](envs/):

| Env | Used for | Entry points |
|---|---|---|
| `libero` (py3.8) | data collection, LIBERO env rendering (`MUJOCO_GL=egl`) | `generate_perturbed_conditions.py`, `generate_perturbed_demos.py`, `verify_perturbed_data.py`, `filter_failed_demos.py` |
| `openvla` (py3.12) | JAX: LoRA expert training, conversion, closed-loop evals, serving | `train_expert.py`, `convert_orbax_to_canonical.py`, `eval_closed_loop.py`, `eval_train_loss.py`, `eval_action_prediction.py`, `serve_lora_policy.py` |
| `qwen3vl` | torch: meta-network training / eval / export | `train_hypernet.py`, `eval_hypernet.py`, `export_generated_lora.py`, `cache_domain_features.py`, tests |

Recreate an environment:

```bash
conda env create -f envs/libero.yml     # or openvla.yml / qwen3vl.yml
conda activate libero
```

The exports are full snapshots (`--no-builds`) of the dev machine, so they pin
working versions of the MuJoCo/robosuite stack, JAX+openpi deps, and torch; expect
some churn when recreating on a different CUDA/driver base. After creating an env,
install this package into it (deps come from the env, not from pip):

```bash
pip install -e . --no-deps
```

**Machine-specific paths live in `configs/paths.yaml` only** (override with the
`MLVLA_PATHS` env var). External repos (openpi, LIBERO-plus, LIBERO) are not
vendored; `mlvla.paths.add_to_sys_path()` adds them at runtime. `configs/domains.yaml`
holds absolute dataset paths and must be regenerated per machine via
`build_domain_registry.py`.

Key env overrides: `MLVLA_PERTURBED_OUT_ROOT` (data collection output),
`MLVLA_PERTURBED_DATASET_ROOT` (collection input reading), plus the usual
`CUDA_VISIBLE_DEVICES` / `MUJOCO_GL`.

## End-to-end sequence

```bash
# ── 1. data collection (libero env) ─────────────────────────────────────
python scripts/generate_perturbed_conditions.py          # authors conditions.yaml + light scene XMLs
bash scripts/launch_all_perturbed_gen.sh                 # replay all conditions x tasks (or:)
python scripts/generate_perturbed_demos.py --conditions v1_azimuth30 --num_demos 50
python scripts/generate_perturbed_demos.py --smoke --conditions v1_azimuth30 --tasks <stem>   # 1-demo smoke
python scripts/verify_perturbed_data.py --root <perturbed_root>
python scripts/filter_failed_demos.py --src_root <raw> --dst_root <clean>

# ── 2. LoRA expert training + validation (openvla env) ──────────────────
bash scripts/train_all_experts.sh                        # FIFO queue over all condition__task domains (or:)
python scripts/train_expert.py --task <task> --output_dir <experts_root>
python scripts/convert_orbax_to_canonical.py --orbax <.../phase1/final/params> --output <...>/params.canonical.npz
python scripts/eval_closed_loop.py --checkpoint_type wizard \
    --canonical_adapter <params.canonical.npz> --task <task> \
    --perturbed_condition <cond> --num_trials 20
python scripts/eval_train_loss.py                        # train-loss check (set via module args)
python scripts/eval_action_prediction.py                 # offline action pred vs recorded actions

# ── 3. meta-network (qwen3vl env) ───────────────────────────────────────
python scripts/build_domain_registry.py --output configs/domains.yaml   # per machine
python scripts/cache_domain_features.py                  # DINOv3 evidence cache (expensive; reuse if present)
python scripts/train_hypernet.py --rung fullvw4          # DirectAB on 90-domain registry
python scripts/eval_hypernet.py --rung fullvw4           # weight-space + offline metrics
python scripts/export_generated_lora.py --rung fullvw4   # export canonical npz for eval/serving
python scripts/verify_generated_schema.py --output <npz> --template <oracle npz>

# serving / closed loop with generated weights (openvla env)
python scripts/serve_lora_policy.py --adapters <root> --task <task> &
python scripts/libero_closed_loop.py --variants <name>   # websocket client
```

Rung splits (`vw1`..`vw5`, `fullvw4`) live in `configs/splits_view*.yaml`; hypernet
settings in `configs/hypernet_view.yaml`; selected-module subsets in
`configs/selected_modules_*.json`.

## Canonical LoRA format

All experts and generated weights use `params.canonical.npz`: 458 module slots,
rank 16 / alpha 16, `manifest_json` carrying module keys + jax stems. I/O:
`mlvla.meta.weights.lora_io`; merging into an OpenPI param tree:
`mlvla.meta.openpi_bridge.merge`.

## Tests

```bash
# qwen3vl env
python -m pytest tests/ -q
```

Registry-dependent tests need `configs/domains.yaml` + the data roots it points to;
they skip on a fresh checkout.

## Notes

- `train_all_experts.sh` / `launch_all_perturbed_gen.sh` keep absolute defaults in
  file-header variables (shells don't read yaml); override via env or edit the header.
- Expert training env: `XLA_PYTHON_CLIENT_GPU_ALLOCATOR=bin-boost`; `train.py`
  derives `TMPDIR` / `HF_*` from `configs/paths.yaml` automatically.
- Oracle experts must pass closed-loop validation before entering the meta-network
  training registry.
