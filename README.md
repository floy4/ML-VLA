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
4. **T2L-style meta-network** (`mlvla.meta.t2l_style`) — Text-to-LoRA inspired architecture
   with two-branch conditioning (semantic DINO + geometric pose) and layer embeddings
   to fix "universal LoRA collapse" problem. See [T2L-style Meta-Network](#t2l-style-meta-network) below.

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

## T2L-style Meta-Network

A new meta-network architecture inspired by [Text-to-LoRA](https://github.com/sakanaai/text-to-lora) (Sakana AI), designed to fix the "universal LoRA collapse" problem in V4 DirectAB.

### Motivation

V4 DirectAB (713M parameters) suffers from **universal LoRA collapse**: cross-domain relative L2 distance ~0.0001, meaning the generator outputs nearly identical LoRAs regardless of input evidence. This is because:
1. No layer identity information (all layers treated identically)
2. End-to-end training with homogeneous batches → no input-dependent gradient
3. Over-parameterized direct heads (17 shape groups × large weight matrices)

### Architecture

T2L-style addresses these issues with:

1. **Two-branch conditioning**: Semantic (DINO 1024-d) + Geometric (pose 7-d) → additive fusion with LayerNorm for zero-shot composition
2. **Layer embeddings**: Depth embedding (168 layers) + Type embedding (17 module types) to distinguish layers
3. **Shared trunk**: Mixer + 2× ResidualBlocks processing concatenated [task_emb + depth_emb + type_emb]
4. **Per-type heads**: Single Linear head per shape group (17 groups)
5. **Reconstruction anchor**: Pre-train by regressing to oracle expert LoRAs before end-to-end fine-tuning

### Parameter Count

| Configuration | Parameters | vs V4 |
|---------------|------------|-------|
| **V4 DirectAB** | 713M | baseline |
| **T2L-L (this work)** | ~72M | **10× reduction** |
| **T2L-S (shared head)** | ~10M | 70× reduction |

The key hyperparameter is `head_in_size` (information bottleneck between shared trunk and output heads):
- `head_in_size=128`: ~72M (recommended, aligns with T2L-L)
- `head_in_size=64`: ~40M (T2L-M)
- `head_in_size=32`: ~10M (T2L-S)

### Code Structure

```
src/mlvla/meta/t2l_style/
├── __init__.py              # Exports all classes
├── condition_encoder.py     # TwoBranchConditionEncoder (semantic + geometric)
├── hypernet.py              # T2LStyleHyperNet (main generator)
├── losses.py                # ReconstructionLoss, CombinedLoss
├── train_recon.py           # Standalone reconstruction trainer
└── README.md                # Detailed documentation
```

### Quick Start

```python
from mlvla.meta.t2l_style import (
    TwoBranchConditionEncoder,
    T2LStyleHyperNet,
    ReconstructionTrainer,
)

# 1. Create condition encoder
encoder = TwoBranchConditionEncoder(
    dino_dim=1024,
    geo_dim=7,
    task_emb_size=256,
    fusion="additive",  # LayerNorm + addition for compositionality
)

# 2. Create hypernetwork
hypernet = T2LStyleHyperNet(
    module_shapes=module_shapes,  # {group_key: {"A": (r, in), "B": (out, r)}}
    max_layers=168,
    head_in_size=128,  # T2L-L configuration
)

# 3. Generate LoRA
task_emb = encoder(dino_features, pose_features)  # [B, 256]
A, B = hypernet(task_emb, layer_indices, module_key)
# A: [B, 16, in_features]
# B: [B, out_features, 16]
```

### Training Pipeline

#### Phase 1: Reconstruction Pre-training

Prevent collapse by regressing to 12 oracle expert LoRAs (camera/lighting/noise/texture × L1/L2/L3).

```bash
# After implementing train_t2l_recon.py
python scripts/train_t2l_recon.py --config configs/t2l_recon.yaml
```

**Success criteria**:
- Validation cosine similarity > 0.95 per domain
- Cross-domain relative L2 >> 0.001 (evidence actually matters)

#### Phase 2: End-to-End Training

Integrate with existing `train_bridge.py` for joint optimization: action loss + reconstruction loss.

```bash
# After integrating into train_bridge.py
python scripts/train_e2e_t2l.py --config configs/t2l_e2e.yaml
```

**Recommended hyperparameters** (for ~72M model):
- `lr = 5e-4` (vs V4's 8.5e-4)
- `batch_size = 16` (reconstruction), `8` (e2e)
- `weight_decay = 3e-5` (vs V4's 1e-4)
- `max_steps = 20000` (vs V4's 30000)
- `lambda_recon = 0.3` (stronger reconstruction constraint)

### Testing

```bash
# qwen3vl env
cd /home/zhy/vla/ML-VLA
PYTHONPATH=src:$PYTHONPATH python scripts/test_t2l_architecture.py
```

Expected output:
```
✓ Forward pass shapes correct
✓ Gradient flow normal (condition encoder: 0.03, hypernet: 0.19)
✓ Loss computation correct
✓ Cosine similarity computation correct
```

### Comparison with V4

| Component | V4 (current) | T2L-style |
|-----------|--------------|-----------|
| **Conditioning** | Concat[DINO, pose] → MLP | Two-branch proj → Add |
| **Layer identity** | None | Depth emb + Type emb |
| **Trunk** | ResidualBlock | Mixer + 2× ResBlock |
| **Output heads** | 17 shape-group heads | 17 shape-group heads |
| **Parameters** | 713M | 72M (10× reduction) |
| **Training** | E2E only | Recon anchor + E2E |
| **Collapse?** | Yes (rel-L2 ~0.0001) | Target: rel-L2 > 0.01 |

### Next Steps

1. **Integrate into train_bridge.py**: Add T2L variant dispatch
2. **Prepare oracle LoRAs**: Load 12 domain experts from `meta_lora/outputs/direct_targets/`
3. **Reconstruction pre-training**: 100 epochs, validate cosine > 0.95
4. **E2E training**: 20k steps, evaluate 12-domain closed-loop performance
5. **Composition generalization**: Test camera+lighting, noise+texture combos

### References

- [Text-to-LoRA: Large Language Models as Implicit Task Adapters](https://arxiv.org/abs/2311.10473)
- V4 implementation: `src/mlvla/meta/hypernet/v4_head.py`
- Detailed docs: `src/mlvla/meta/t2l_style/README.md`

