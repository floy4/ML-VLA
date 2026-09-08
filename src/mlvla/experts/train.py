"""JAX residual-expert trainer for the Table-8 π0.5 topology.

Usage:
    python scripts/train_expert.py \
        --task "turn on the stove" \
        --num_steps 3000 \
        --output_dir <expert_output_root>/<label>

Env vars (defaulted from configs/paths.yaml):
    TMPDIR, HF_DATASETS_CACHE, HF_HOME, HF_LEROBOT_HOME
Opt-in perturbed-data redirect (else the default source dataset is used):
    MLVLA_PERTURBED_DATASET_ROOT (= legacy WIZARD_PERTURBED_DATASET_ROOT)
"""
# MUST be set before any JAX imports to control the GPU memory allocator.
# - GPU_ALLOCATOR=bin-boost: maintains better free lists than BFC, more
#   fragmentation-resistant. Should find a contiguous 2 GiB block in the
#   ~10 GB free space even when arena is fragmented by multiple checkpoint loads.
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_GPU_ALLOCATOR", "bin-boost")

import dataclasses
import hashlib
import logging
import json
import pathlib
import shutil
import sys

import etils.epath as epath
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tqdm_loggable.auto as tqdm

from mlvla import paths as _paths

_paths.set_derived_env()  # TMPDIR / HF_* from paths.yaml
_paths.add_to_sys_path()  # openpi_root/src

from mlvla.jax._compat import isolate_unused_pytorch_backend
isolate_unused_pytorch_backend()

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config

# ── LeRobot compatibility patches ──────────────────────────────────────────
import lerobot.datasets.lerobot_dataset as _lerobot_dataset
_lerobot_dataset.CODEBASE_VERSION = "v3.0"

# Module-level variable to control task filtering in the init patch.
_TARGET_TASK: str | None = None
_ACTION_HORIZON: int = 10  # WizardPi05Config default; exclude last H-1 frames per episode
DIAGNOSTIC_STEPS: int = 100  # First N steps log per-step finite checks + per-module grad norms

_EPISODE_FILTER: set[int] | None = None  # optional episode whitelist (e2e train/val splits)


def intersect_episodes(available: list[int], extra: set[int] | None) -> list[int]:
    """Intersect available episode indices with an optional whitelist (sorted)."""
    if extra is None:
        return sorted(available)
    kept = sorted(set(available) & extra)
    if not kept:
        raise ValueError("episode filter produced an empty set; check split ids vs dataset")
    return kept


def _get_perturbed_root() -> str | None:
    """Opt-in redirect to a perturbed dataset; None = use default source."""
    root = os.environ.get("MLVLA_PERTURBED_DATASET_ROOT",
                          os.environ.get("WIZARD_PERTURBED_DATASET_ROOT"))
    if root:
        if not pathlib.Path(root).is_dir():
            raise FileNotFoundError(
                f"MLVLA_PERTURBED_DATASET_ROOT={root!r} is not a directory")
    return root


def _dataset_repo_id_and_root() -> tuple[str, str]:
    """Return (repo_id, root) for the source dataset, honoring the perturbed-redirect env var."""
    if _get_perturbed_root() is not None:
        return "wizard/perturbed", _get_perturbed_root()
    return "physical-intelligence/libero", _paths.get("source_dataset")


def _load_task_to_episodes():
    """Load mapping from task description → list of episode indices."""
    repo_id, root = _dataset_repo_id_and_root()
    meta = _lerobot_dataset.LeRobotDatasetMetadata(
        repo_id,
        root=root,
        revision="v3.0",
        force_cache_sync=False,
    )
    task_to_eps = {}
    for ep in meta.episodes:
        tasks = ep.get("tasks", [])
        if tasks:
            task_name = tasks[0]
            task_to_eps.setdefault(task_name, []).append(int(ep["episode_index"]))
    return task_to_eps
def _get_available_episodes() -> list[int]:
    """Return episode indices that exist in the local data cache.

    v3.0 stores episodes in sharded parquet files (file-*.parquet), not as
    individual episode_*.parquet files.  Derive the available set from the
    dataset metadata instead of file-name globbing.
    """
    repo_id, root = _dataset_repo_id_and_root()
    meta = _lerobot_dataset.LeRobotDatasetMetadata(
        repo_id,
        root=root,
        revision="v3.0",
        force_cache_sync=False,
    )
    return sorted(set(int(ep["episode_index"]) for ep in meta.episodes))
def _patch_lerobot_dataset():
    """Patch LeRobotDataset to work with locally cached subset + task filtering."""
    _orig_init = _lerobot_dataset.LeRobotDataset.__init__
    _orig_get_query_indices = _lerobot_dataset.LeRobotDataset._get_query_indices

    def _patched_init(self, repo_id, *args, **kwargs):
        hf_home = pathlib.Path(os.environ.get("HF_LEROBOT_HOME", ""))
        if not hf_home.is_dir():
            hf_home = pathlib.Path(os.environ.get("HF_HOME", _paths.get("hf_cache"))) / "lerobot"

        # v2.0: derive available episodes from metadata (sharded parquet, not
        # per-episode files), then optionally filter by target task.
        available = _get_available_episodes()

        # Filter by target task if specified
        if _TARGET_TASK is not None and available:
            task_to_eps = _load_task_to_episodes()
            task_eps = set(task_to_eps.get(_TARGET_TASK, []))
            available = [e for e in available if e in task_eps]
            available = intersect_episodes(available, _EPISODE_FILTER)
            if not available:
                raise ValueError(
                    f"Task {_TARGET_TASK!r} has no locally available episodes; "
                    "refusing to silently train on the full LIBERO dataset"
                )

        if available and not kwargs.get("episodes"):
            kwargs["episodes"] = available
        kwargs["force_cache_sync"] = False
        # The episode filter above is computed from the perturbed-root metadata,
        # but repo_id still resolves to the ORIGINAL libero dataset via the HF
        # cache symlink. Without redirecting root too, training silently loads
        # original-view episodes of unrelated tasks.  (Found 2026-08-16: all
        # perturbed experts trained this way.)
        perturbed_root = _get_perturbed_root()
        if perturbed_root is not None:
            kwargs["root"] = perturbed_root
            kwargs["revision"] = "v3.0"
        _orig_init(self, repo_id, *args, **kwargs)

        # Build per-episode GLOBAL index bounds from the loaded parquet, NOT from
        # metadata dataset_from_index/dataset_to_index. For this LIBERO v3.0
        # download the metadata from/to resets per shard file (per-shard row
        # offsets), while hf_dataset["index"] is the global absolute index. Mixing
        # the two coordinate systems clamps every delta query to one frame, which
        # produces constant action chunks and trains a degenerate policy.
        import numpy as np
        hf_ep = np.asarray(self.hf_dataset["episode_index"])
        hf_idx = np.asarray(self.hf_dataset["index"])
        self._v2_ep_global = {}
        for ep_idx in self.episodes:
            mask = hf_ep == ep_idx
            idxs = hf_idx[mask]
            self._v2_ep_global[int(ep_idx)] = (int(idxs.min()), int(idxs.max()) + 1)
        # _absolute_to_relative_idx is already built correctly by lerobot's own
        # __init__ (from hf_dataset["index"]) — do not override it.

    def _patched_get_query_indices(self, abs_idx: int, ep_idx: int):
        """Compute delta-timestamp queries in the global "index" coordinate."""
        import torch
        ep_start, ep_end = self._v2_ep_global[ep_idx]
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, abs_idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(abs_idx + delta < ep_start) | (abs_idx + delta >= ep_end) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    _lerobot_dataset.LeRobotDataset.__init__ = _patched_init
    _lerobot_dataset.LeRobotDataset.download = lambda s, *a, **kw: None
    _lerobot_dataset.LeRobotDataset._get_query_indices = _patched_get_query_indices
_patch_lerobot_dataset()

# Fix bug in openpi's create_torch_dataset: tasks.index is the pandas index (int),
# not the task description column. Use tasks["task"] instead.
import openpi.training.data_loader as _dl_module
import openpi.transforms as _transforms

class _IdentityTransform:
    """Passthrough transform that satisfies the DataTransformFn Protocol."""
    def __call__(self, data):
        return data

def _patched_create_torch_dataset(data_config, action_horizon, model_config):
    from openpi.training.data_loader import lerobot_dataset, TransformedDataset

    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set.")
    if repo_id == "fake":
        return _dl_module.FakeDataset(model_config, num_samples=1024)

    # Metadata must come from the same source the dataset loads from: under the
    # perturbed redirect, task_index→prompt mapping from the ORIGINAL metadata
    # mislabels every episode.
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(
        repo_id, root=_get_perturbed_root(), revision="v3.0", force_cache_sync=False,
    )

    # Filter episodes by task if a target is set (required for Phase 2 correctness).
    # Without this, LeRobotDataset loads ALL cached episodes but the per-episode
    # global index map only covers the task's episodes → indices go out of bounds.
    episodes_filter = None
    if _TARGET_TASK is not None:
        task_to_eps = _load_task_to_episodes()
        task_eps = set(task_to_eps.get(_TARGET_TASK, []))
        available = _get_available_episodes()
        episodes_filter = intersect_episodes(
            [e for e in available if e in task_eps], _EPISODE_FILTER)
        if not episodes_filter:
            raise ValueError(
                f"Task {_TARGET_TASK!r} has no locally available episodes; "
                "check the exact metadata task string"
            )

    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)]
            for key in (data_config.base_config.action_sequence_keys
                        if hasattr(data_config, "base_config") and data_config.base_config is not None
                        else data_config.action_sequence_keys)
        },
        episodes=episodes_filter,
    )

    # LeRobotDataset v2.0 outputs flat keys (image, wrist_image, state, actions).
    # LiberoInputs accesses with string keys like data["observation/image"], so we need
    # a flat dict with slash-separated keys matching what RepackTransform expects.
    # flatten_dict on a nested dict {"observation": {"image": x}} → {"observation/image": x}
    def _flatten_lerobot_item(data):
        # v2.0: data["task_index"] is int, data["task"] is the string description
        task_idx = data.get("task_index", data.get("task", 0))
        nested = {
            "observation": {
                "image": data["image"],
                "wrist_image": data["wrist_image"],
                "state": data["state"],
            },
            "actions": data["actions"],
            "task_index": task_idx,
        }
        return _transforms.flatten_dict(nested)

    dataset = TransformedDataset(dataset, [_flatten_lerobot_item])

    # prompt_from_task may be on base_config (DataConfig) or directly on the factory (LeRobotLiberoDataConfig)
    prompt_flag = False
    if hasattr(data_config, "base_config") and data_config.base_config is not None:
        prompt_flag = bool(getattr(data_config.base_config, "prompt_from_task", False))
    elif hasattr(data_config, "prompt_from_task"):
        prompt_flag = bool(data_config.prompt_from_task)
    if prompt_flag:
        # v2.0 style: 'task' string is the DataFrame index, 'task_index' the sole
        # column.  v3.0 style (perturbed datasets): row index is positional, the
        # task string lives in a 'task' column.  Handle both.
        if "task" in dataset_meta.tasks.columns:
            task_names = dataset_meta.tasks["task"].tolist()
        else:
            task_names = dataset_meta.tasks.index.tolist()
        task_indices = dataset_meta.tasks["task_index"].tolist()
        tasks_dict = dict(zip(task_indices, task_names))
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(tasks_dict)])

    return dataset

_dl_module.create_torch_dataset = _patched_create_torch_dataset

# Also patch transform_dataset to skip RepackTransform (which expects flat keys
# from RLDS/nested-from-flat conversion, but LeRobotDataset already gives flat keys
# and we convert to nested via _nest_lerobot_item).
_orig_transform_dataset = _dl_module.transform_dataset

def _patched_transform_dataset(dataset, data_config, *, skip_norm_stats=False):
    real_config = data_config.create(assets_dirs={}, model_config=get_model_config()) if isinstance(data_config, _config.DataConfigFactory) else data_config
    norm_stats = {}
    if real_config.repo_id != "fake" and not skip_norm_stats:
        if real_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = real_config.norm_stats

    return _dl_module.TransformedDataset(
        dataset,
        [
            _IdentityTransform(),
            *real_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=real_config.use_quantile_norm),
            *real_config.model_transforms.inputs,
        ],
    )

_dl_module.transform_dataset = _patched_transform_dataset

import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as weight_loaders
from mlvla.jax.pi05_table8 import WizardPi05Config
from mlvla.adapters.table8_converter import export_table8_adapter
def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
def get_model_config() -> WizardPi05Config:
    return WizardPi05Config()
def get_data_config(base_checkpoint: str = _paths.get("base_checkpoint")) -> _config.LeRobotLiberoDataConfig:
    return _config.LeRobotLiberoDataConfig(
        repo_id="physical-intelligence/libero",
        assets=_config.AssetsConfig(
            assets_dir=f"{base_checkpoint}/assets",
            asset_id="physical-intelligence/libero",
        ),
        base_config=_config.DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
    )
def get_train_config(
    task_name: str,
    output_dir: pathlib.Path,
    num_steps: int,
    phase: int = 1,
    batch_size: int = 4,
    save_interval: int | None = None,
    keep_period: int | None = None,
    base_checkpoint: str = _paths.get("base_checkpoint"),
) -> _config.TrainConfig:
    safe_name = hashlib.md5(task_name.encode()).hexdigest()[:8]
    phase1_warmup = min(25, max(0, num_steps - 1))
    return _config.TrainConfig(
        name=f"expert_{safe_name}",
        model=get_model_config(),
        data=get_data_config(base_checkpoint),
        batch_size=batch_size,
        num_workers=0,
        lr_schedule=_optimizer.CosineDecaySchedule(**(
            {"warmup_steps": phase1_warmup, "peak_lr": 5e-5,
             "decay_steps": max(num_steps, phase1_warmup + 1), "decay_lr": 1e-7}
            if phase == 1 else
            {"warmup_steps": 0, "peak_lr": 1e-5, "decay_steps": 500, "decay_lr": 1e-7}
        )),
        optimizer=_optimizer.AdamW(b1=0.9, b2=0.999, clip_gradient_norm=1.0),
        ema_decay=None,
        freeze_filter=nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex(".*lora.*"))),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            f"{base_checkpoint}/params"
        ),
        num_train_steps=num_steps,
        checkpoint_base_dir=output_dir,
        project_name="wizard_residual_experts",
        log_interval=50,
        save_interval=save_interval if save_interval is not None else (500 if phase == 1 else 50),
        keep_period=keep_period,
    )
def init_train_state(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    skip_base_weight_load: bool = False,
):
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng, partial_params=None):
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)
            graphdef, state = nnx.split(model)
        else:
            graphdef, state = nnx.split(model)

        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        trainable_params = params.filter(config.trainable_filter)
        opt_state = tx.init(trainable_params)

        return training_utils.TrainState(
            step=0, params=params, model_def=graphdef,
            tx=tx, opt_state=opt_state, ema_decay=None, ema_params=None,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=False)

    if skip_base_weight_load:
        return jax.jit(init, out_shardings=state_sharding)(init_rng), state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding
def _load_weights_and_validate(loader: weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    import flax.traverse_util as traverse_util
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )
@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )

    # Per-module grad norms for diagnosing which LoRA region first produces
    # non-finite values. PathRegex matches against the slash-joined nnx path.
    def _safe_grad_norm(pattern: str) -> at.Array:
        try:
            sub = grads.filter(nnx_utils.PathRegex(pattern))
            return optax.global_norm(sub)
        except Exception:
            return jnp.asarray(0.0)

    grad_norm_vision = _safe_grad_norm(r".*PaliGemma.*img.*")
    grad_norm_gemma_llm = _safe_grad_norm(r".*PaliGemma.*llm.*")
    grad_norm_action_time = _safe_grad_norm(r".*(action_in_proj|action_out_proj|time_mlp).*")
    grad_norm_lm_head = _safe_grad_norm(r".*lm_head.*")

    all_finite = jnp.all(jnp.isfinite(loss)) & jnp.all(jnp.isfinite(optax.global_norm(grads))) & jnp.all(jnp.isfinite(optax.global_norm(updates)))

    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        "grad_norm_vision": grad_norm_vision,
        "grad_norm_gemma_llm": grad_norm_gemma_llm,
        "grad_norm_action_time": grad_norm_action_time,
        "grad_norm_lm_head": grad_norm_lm_head,
        "all_finite": all_finite,
    }
    return new_state, info
def analyze_lora_topology(state: training_utils.TrainState) -> dict:
    params = state.params
    lora_params = {}
    total_lora_params = 0
    total_params = 0
    for path, var_state in params.flat_state().items():
        path_str = "/".join(str(p) for p in path)
        if hasattr(var_state, 'value') and hasattr(var_state.value, 'size'):
            count = var_state.value.size
            total_params += count
            if "lora" in path_str.lower():
                lora_params[path_str] = list(var_state.value.shape)
                total_lora_params += count
    return {
        "num_lora_modules": len(lora_params),
        "total_lora_params": total_lora_params,
        "module_shapes": lora_params,
        "total_params": total_params,
    }
def _save_params(path: pathlib.Path, train_state, config: _config.TrainConfig, save_canonical: bool = False, base_checkpoint: str = _paths.get("base_checkpoint")):
    trainable_params = train_state.params.filter(config.trainable_filter)
    params_dict = trainable_params.to_pure_dict()
    # Save orbax format (for compatibility)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(epath.Path(path), params_dict, force=True)
    if save_canonical:
        # Save the framework-independent Table-8 A/B representation. The old
        # fused `w_0...w_n` NPZ format is intentionally not produced here.
        canonical = export_table8_adapter(
            train_state.params,
            base_checkpoint=base_checkpoint,
        )
        canonical.save_npz(path.parent / f"{path.name}.canonical.npz")


def _save_resume_state(path: pathlib.Path, train_state, config: _config.TrainConfig):
    """Save LoRA params plus optimizer moments for exact Phase-2 continuation."""
    payload = {
        "step": np.asarray(train_state.step),
        "params": train_state.params.filter(config.trainable_filter).to_pure_dict(),
        "opt_state": train_state.opt_state,
    }
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(epath.Path(path), payload, force=True)


def _restore_phase1_params(path: pathlib.Path, shape_template, config: _config.TrainConfig):
    """Restore Phase-1 LoRA params as a plain dict {key_tuple: array}.

    Returns real arrays keyed by full nnx.State tuple-keys (including 'PaliGemma'
    prefix where applicable), so the caller can use ``fresh_flat[key].replace(value=arr)``
    without going through broken placeholder wrappers.

    Handles shape mismatches from Phase 1 checkpoints trained with mixed rank=32/16
    to the current rank=16 everywhere topology.
    """
    import flax.traverse_util as traverse_util
    logger = logging.getLogger(__name__)
    with ocp.PyTreeCheckpointer() as checkpointer:
        restored = checkpointer.restore(epath.Path(path))

    restored_params = restored["params"]
    flat = traverse_util.flatten_dict(restored_params)

    # Collect (shape, dtype) for every array leaf in shape_template, keyed by full tuple.
    # shape_template can be either nnx.State (has flat_state) or a pure dict from to_pure_dict().
    def _collect_shape_dtype(node, prefix=()):
        out = {}
        if hasattr(node, "flat_state"):
            for k, vs in node.flat_state().items():
                v = getattr(vs, "value", vs)
                if hasattr(v, "shape"):
                    out[k] = (tuple(v.shape), v.dtype)
        elif isinstance(node, dict):
            for k, v in node.items():
                out.update(_collect_shape_dtype(v, prefix + (k,)))
        elif hasattr(node, "shape"):
            out[prefix] = (tuple(node.shape), node.dtype)
        return out

    fresh_shapes = _collect_shape_dtype(shape_template)

    def _map_value(cv, fv_shape, fv_dtype):
        """Map checkpoint value cv to fresh model's shape fv_shape."""
        fv_ndim = len(fv_shape)
        fv_size = 1
        for dim in fv_shape:
            fv_size *= dim
        if cv.shape == fv_shape:
            return cv.astype(fv_dtype)
        if cv.size == fv_size:
            return cv.reshape(fv_shape).astype(fv_dtype)
        # Handle 4D->3D reshape: (..., 8, rank, hidden) -> (..., rank, hidden)
        if cv.ndim == fv_ndim + 1 and cv.ndim >= 3:
            diff_dim = None
            for d in range(cv.ndim):
                if d < fv_ndim and cv.shape[d] != fv_shape[d]:
                    diff_dim = d
                    break
            if diff_dim is not None:
                reduced = jnp.sum(cv, axis=diff_dim)
                if reduced.size == fv_size:
                    return reduced.astype(fv_dtype)
        flat_ck = cv.reshape(-1)
        n = min(flat_ck.size, fv_size)
        return flat_ck[:n].reshape(fv_shape).astype(fv_dtype)

    # Match by stripped key (drop leading 'PaliGemma' segment so both sides agree).
    def _strip_prefix(key):
        key = list(key)
        while key and str(key[0]) == 'PaliGemma':
            key.pop(0)
        return tuple(key)

    fresh_by_stripped = {_strip_prefix(fk): (fk, shape, dtype)
                         for fk, (shape, dtype) in fresh_shapes.items()}

    params = {}
    for ck, cv in flat.items():
        if not isinstance(cv, jax.Array):
            continue
        stripped = _strip_prefix(ck)
        if not stripped:
            continue
        if stripped in fresh_by_stripped:
            orig_fk, fv_shape, fv_dtype = fresh_by_stripped[stripped]
            params[orig_fk] = jnp.asarray(_map_value(cv, fv_shape, fv_dtype))
        else:
            logger.warning(f"No match for checkpoint key {ck} (stripped={stripped})")

    logger.info(f"Restored {len(params)} params from Phase 1 checkpoint")
    return params


def _restore_phase1_state(path: pathlib.Path, train_state, config: _config.TrainConfig):
    """Restore parameters and Adam moments, resetting Phase-2 schedule counters.

    Handles three kinds of shape mismatch between the Phase-1 checkpoint (trained
    with an older gemma_lora.py that used rank=32 for some expert layers) and the
    current fresh model (rank=16 everywhere):

    1. Same shape: direct assign.
    2. Different ndim but same total size: reshape (flatten → reshape to fresh shape).
       Used for attention params where the old code stored 4D but the new code stores
       3D (e.g. attn_vec_einsum/lora_b: (18,8,16,2048) → (18,16,2048)).
    3. Different total size: sum over extra dims to match, then truncate the rank
       dimension if the checkpoint has rank=32 but the fresh model has rank=16.
    """
    import flax.traverse_util as traverse_util
    logger = logging.getLogger(__name__)
    with ocp.PyTreeCheckpointer() as checkpointer:
        restored = checkpointer.restore(epath.Path(path))

    restored_params = restored["params"]
    flat = traverse_util.flatten_dict(restored_params)
    lora_keys = [(k, v.shape) for k, v in flat.items() if any("lora" in str(p) for p in k)]
    logger.info(f"Phase1 lora param shapes (first 15): {lora_keys[:15]}")
    fresh_lora = [(k, v.value.shape) for k, v in train_state.params.flat_state().items()
                  if any("lora" in str(p) for p in k)]
    logger.info(f"Fresh model lora shapes (first 15): {fresh_lora[:15]}")

    fresh_by_key = {fk: fvs for fk, fvs in train_state.params.flat_state().items()
                    if isinstance(fvs.value, jax.Array)}

    def _map_value(ck, cv, fv):
        """Map checkpoint array cv to the fresh model shape fv."""
        if fv.shape == cv.shape:
            return cv.astype(fv.dtype)
        if fv.size == cv.size:
            return cv.reshape(fv.shape).astype(fv.dtype)
        # Size mismatch: need to reduce dimensions or truncate rank.
        # Common case: checkpoint has extra leading dims (e.g. 4D→3D for attention
        # lora_b where old code stored (18,8,R,W) but new stores (18,R,W)).
        if cv.ndim > fv.ndim and cv.ndim - fv.ndim == 1:
            # Find the dimension where sizes differ (usually the num_heads dim).
            # Sum over that dimension to match the fresh model count.
            diff_dim = None
            for d in range(cv.ndim):
                if d < fv.ndim and cv.shape[d] != fv.shape[d]:
                    diff_dim = d
                    break
            if diff_dim is not None:
                # Sum over the extra dimension.
                reduced = jnp.sum(cv, axis=diff_dim)
                if reduced.size == fv.size:
                    return reduced.astype(fv.dtype)
        # Fallback: truncate/pad to match size (for rank changes).
        flat_ck = cv.reshape(-1)
        n = min(flat_ck.size, fv.size)
        return flat_ck[:n].reshape(fv.shape).astype(fv.dtype)

    params = {}
    for ck, cv in flat.items():
        if not isinstance(cv, jax.Array):
            continue

        fk = ck
        if fk in fresh_by_key:
            fvs = fresh_by_key[fk]
            mapped = _map_value(ck, cv, fvs.value)
            params[fk] = fvs.replace(value=mapped)
        else:
            # No exact key match — try partial key matching.
            partial_key = ck[-1] if ck else None
            matched = False
            for fk, fvs in fresh_by_key.items():
                if fk not in params and fk[-1] == partial_key:
                    mapped = _map_value(ck, cv, fvs.value)
                    params[fk] = fvs.replace(value=mapped)
                    matched = True
                    break
            if not matched:
                logger.warning(f"Could not match checkpoint param {ck} shape {list(cv.shape)}")

    nnx_params = nnx.State(params)
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    trainable_params = nnx_params.filter(config.trainable_filter)
    opt_state = tx.init(trainable_params)
    return dataclasses.replace(train_state, step=0, params=nnx_params, opt_state=opt_state)


def _transplant_optimizer_moments(fresh, restored):
    if restored is None:
        return fresh
    if hasattr(fresh, "ndim") and hasattr(fresh, "shape"):
        if fresh.ndim == 0:
            return fresh
        old = jnp.asarray(restored, dtype=fresh.dtype)
        if old.shape != fresh.shape:
            raise ValueError(f"Optimizer moment shape mismatch: {old.shape} != {fresh.shape}")
        return old
    if fresh is None:
        return None
    if isinstance(fresh, nnx.State):
        rebuilt = jax.tree.map(lambda value: value, fresh)
        def strip_variable_state(value):
            if isinstance(value, dict) and set(value) == {"value"}:
                return value["value"]
            if isinstance(value, dict):
                return {key: strip_variable_state(item) for key, item in value.items()}
            return value
        rebuilt.replace_by_pure_dict(strip_variable_state(restored))
        return rebuilt
    if hasattr(fresh, "value") and hasattr(fresh, "replace"):
        old_value = restored["value"] if isinstance(restored, dict) else restored.value
        return fresh.replace(value=_transplant_optimizer_moments(fresh.value, old_value))
    if isinstance(fresh, tuple) and hasattr(fresh, "_fields"):
        values = []
        for index, field in enumerate(fresh._fields):
            old_value = restored[field] if isinstance(restored, dict) else restored[index]
            values.append(_transplant_optimizer_moments(getattr(fresh, field), old_value))
        return type(fresh)(*values)
    if isinstance(fresh, tuple):
        return tuple(_transplant_optimizer_moments(value, restored[index])
                     for index, value in enumerate(fresh))
    if isinstance(fresh, list):
        return [_transplant_optimizer_moments(value, restored[index])
                for index, value in enumerate(fresh)]
    if isinstance(fresh, dict):
        return {key: _transplant_optimizer_moments(value, restored[key])
                for key, value in fresh.items()}
    if hasattr(fresh, "items"):
        values = {key: _transplant_optimizer_moments(value, restored[key])
                  for key, value in fresh.items()}
        return type(fresh)(values)
    return fresh


def _restore_phase1_optimizer_moments(path: pathlib.Path, fresh_opt_state):
    """Restore Adam moments while retaining fresh Phase-2 scalar counters.

    The two phases intentionally use different schedules.  Scalar optimizer
    counters therefore come from ``fresh_opt_state`` while moment tensors are
    transplanted from the Phase-1 resume payload, matching Appendix E's direct
    optimizer-state continuation without carrying the old LR horizon forward.
    """
    with ocp.PyTreeCheckpointer() as checkpointer:
        restored = checkpointer.restore(epath.Path(path))
    if "opt_state" not in restored:
        raise ValueError(f"Resume checkpoint has no optimizer state: {path}")
    result = _transplant_optimizer_moments(fresh_opt_state, restored["opt_state"])
    del restored
    return result


def _save_as_npz(path: pathlib.Path, params_dict: dict):
    """Save parameters as a simple .npz file alongside the orbax checkpoint."""
    lora_weights = {}
    def _collect(prefix, d):
        for k, v in d.items():
            key = "/".join(str(p) for p in prefix + (k,))
            if isinstance(v, dict):
                _collect(prefix + (k,), v)
            elif hasattr(v, "shape"):
                lora_weights[key] = np.array(v)
    _collect((), params_dict)
    npz_path = path.parent / f"{path.name}.npz"
    np.savez_compressed(npz_path, lora_paths=np.array(sorted(lora_weights.keys())),
                        **{f"w_{i}": v for i, (_, v) in enumerate(sorted(lora_weights.items()))})
def main():
    global _TARGET_TASK

    init_logging()
    logger = logging.getLogger(__name__)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Task description to train on")
    parser.add_argument(
        "--num_steps", type=int, default=None,
        help="Override steps (defaults: Phase 1=10000, Phase 2=500)",
    )
    parser.add_argument("--output_dir", type=str, default=_paths.get("expert_output_root"), help="Output directory")
    parser.add_argument(
        "--gpu",
        type=str,
        default="",
        help="GPU device IDs for CUDA_VISIBLE_DEVICES. Empty (default) keeps the parent env intact, allowing multi-GPU training via the launcher.",
    )
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1)
    parser.add_argument("--resume_state", type=str, default=None,
                        help="Phase-1 resume_state directory; required for phase 2")
    parser.add_argument("--resume_state_phase1", type=str, default=None,
                        help="Phase-1 resume_state directory for extending a previous Phase 1 run. "
                             "Restores LoRA params + Adam moments, resets step=0, uses a fresh "
                             "CosineDecay schedule. Output goes to phase1_resume/ subdirectory.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size (default: 4; global across all devices)")
    parser.add_argument("--fsdp_devices", type=int, default=1,
                        help="Number of FSDP devices for model sharding. "
                             "Total devices = jax.device_count() must be divisible. "
                             "Data parallelism = total / fsdp_devices. Default 1 = pure DDP.")
    parser.add_argument("--reset_phase2_optimizer", action="store_true",
                        help="Memory fallback/deviation: do not restore Phase-1 Adam moments")
    parser.add_argument("--save_interval", type=int, default=None,
                        help="Checkpoint interval (defaults: Phase 1=500, Phase 2=10)")
    parser.add_argument("--keep_period", type=int, default=None,
                        help="If set, keep checkpoints at steps that are multiples of this value "
                             "(in addition to the latest). Other intermediate checkpoints are deleted. "
                             "Mirrors openpi's keep_period behavior.")
    parser.add_argument("--base_checkpoint", type=str,
                        default=_paths.get("base_checkpoint"),
                        help="Base checkpoint directory (default: pi05_base)")
    parser.add_argument("--rulebook", default="full", choices=["full", "reduced"],
                        help="LoRA rulebook: full (configs/pi05_lora_rulebook.yaml) "
                             "or reduced (configs/pi05_lora_rulebook_reduced.yaml, "
                             "top-50% sensitive modules). Affects only the recorded "
                             "metadata.json path; LoRA module topology is fixed.")
    parsed_args = parser.parse_args()

    # Set target task BEFORE using data loader (controls episode filtering in patch)
    _TARGET_TASK = parsed_args.task

    # Set GPU only when explicitly passed; otherwise inherit the parent env
    # (so multi-GPU launchers like train_all_experts.sh can request 8 devices).
    if parsed_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = parsed_args.gpu

    task_name = parsed_args.task
    num_steps = parsed_args.num_steps
    if num_steps is None:
        num_steps = 500 if parsed_args.phase == 2 else 10000
    safe_name = hashlib.md5(task_name.encode()).hexdigest()[:8]
    expert_root = pathlib.Path(parsed_args.output_dir) / f"expert_{safe_name}"
    if parsed_args.phase == 1 and parsed_args.resume_state_phase1 is not None:
        output_dir = expert_root / "phase1_resume"
    else:
        output_dir = expert_root / f"phase{parsed_args.phase}"

    output_dir.mkdir(parents=True, exist_ok=True)
    expert_root.mkdir(parents=True, exist_ok=True)
    (expert_root / "metadata.json").write_text(json.dumps({
        "prompt": task_name, "expert_hash": safe_name, "phase": parsed_args.phase,
        "base_checkpoint": parsed_args.base_checkpoint,
        "rank": 16, "alpha": 16, "dropout": 0.0,
        "table8_rulebook": (
            "configs/pi05_lora_rulebook_reduced.yaml" if parsed_args.rulebook == "reduced"
            else "configs/pi05_lora_rulebook.yaml"
        ),
        "phase2_optimizer": (
            "reset_deviation" if parsed_args.reset_phase2_optimizer else "resume_moments_reset_schedule"
        ) if parsed_args.phase == 2 else None,
    }, indent=2))
    logger.info(f"Task: {task_name}")
    logger.info(f"Output: {output_dir}")

    # Log episode count for this task
    task_to_eps = _load_task_to_episodes()
    available = _get_available_episodes()
    task_eps = set(task_to_eps.get(task_name, [])) & set(available)
    logger.info(f"Available episodes for task: {len(task_eps)}")

    if parsed_args.save_interval is not None and parsed_args.save_interval <= 0:
        parser.error("--save_interval must be positive")
    if parsed_args.keep_period is not None and parsed_args.keep_period <= 0:
        parser.error("--keep_period must be positive")
    config = get_train_config(
        task_name, output_dir, num_steps, parsed_args.phase,
        parsed_args.batch_size, parsed_args.save_interval,
        keep_period=parsed_args.keep_period,
        base_checkpoint=parsed_args.base_checkpoint,
    )

    def _mem_usage():
        import subprocess
        try:
            cmd = ["nvidia-smi", "--query-gpu=memory.used,memory.free",
                   "--format=csv,noheader,nounits"]
            if parsed_args.gpu:
                cmd += ["-i", parsed_args.gpu]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            lines = r.stdout.strip().splitlines()
            if not lines:
                return "GPU mem unavailable"
            # Sum across all visible GPUs when multi-device.
            used = sum(int(l.split(",")[0]) for l in lines)
            free = sum(int(l.split(",")[1]) for l in lines)
            return f"GPU mem used={used} MiB free={free} MiB"
        except Exception:
            return "GPU mem unavailable"

    logger.info("Initializing training state...")
    mesh = sharding.make_mesh(num_fsdp_devices=parsed_args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    num_devices = jax.device_count()
    num_data_parallel = num_devices // parsed_args.fsdp_devices
    # OpenPI's DATA_AXIS = (BATCH_AXIS, FSDP_AXIS) — data is sharded across BOTH
    # mesh axes, i.e. across ALL devices. So global batch must divide total device
    # count, not just the data-parallel group. With fsdp_devices=1 this collapses
    # to the same check, but with fsdp_devices>1 it's strictly tighter.
    if config.batch_size % num_devices != 0:
        parser.error(
            f"batch_size {config.batch_size} must be divisible by total device count "
            f"{num_devices} (data is sharded across both batch and fsdp mesh axes)."
        )
    logger.info(
        f"Mesh: {mesh.shape} | devices={num_devices} fsdp={parsed_args.fsdp_devices} "
        f"data_parallel={num_data_parallel} batch={config.batch_size} "
        f"per_shard={config.batch_size // num_devices}"
    )
    init_rng = jax.random.PRNGKey(42)
    train_state, state_sharding = init_train_state(config, init_rng, mesh)
    if parsed_args.phase == 2:
        if parsed_args.resume_state is None:
            parser.error("--resume_state is required for Phase 2")
        # Build a minimal shape template WITHOUT loading base weights.
        # This saves ~6.7 GB vs init_train_state which loads them.
        # Using a different RNG seed so the random init doesn't overlap with
        # the final model's rng stream.
        template_rng = jax.random.PRNGKey(99)
        template_model = config.model.create(template_rng)
        shape_template = nnx.state(template_model).to_pure_dict()
        del template_model
        import gc; gc.collect()
        logger.info(f"[MEM] After del template_model (shape only): {_mem_usage()}")
        # Restore Phase-1 LoRA params first; optimizer moments are loaded only
        # after the fresh Phase-2 optimizer tree has been constructed.
        restored_params = _restore_phase1_params(
            pathlib.Path(parsed_args.resume_state), shape_template, config
        )
        del shape_template
        import gc; gc.collect()
        logger.info(f"[MEM] After restore_phase1_params + del shape_template: {_mem_usage()}")
        # Build fresh_model fresh, overlay LoRA params, create train_state.
        fresh_rng = jax.random.PRNGKey(42)
        _, model_rng = jax.random.split(fresh_rng)
        fresh_model = config.model.create(model_rng)
        logger.info(f"[MEM] After model.create: {_mem_usage()}")
        fresh_params_shape = nnx.state(fresh_model).to_pure_dict()
        partial = _load_weights_and_validate(config.weight_loader, fresh_params_shape)
        gdef, fstate = nnx.split(fresh_model)
        fstate.replace_by_pure_dict(partial)
        fresh_model = nnx.merge(gdef, fstate)
        del partial, fresh_params_shape
        import gc; gc.collect()
        logger.info(f"[MEM] After base weight load: {_mem_usage()}")
        _, fresh_params = nnx.split(fresh_model, nnx.Param)
        # Overlay Phase 1 LoRA weights. flat_state() returns a *copy*, so we
        # must rebuild an nnx.State from the modified flat dict via
        # nnx.State.from_flat_path() — otherwise the overlay silently no-ops
        # and Phase 2 trains from random init (lora_b stays 0).
        fresh_flat = fresh_params.flat_state()
        overlay_count = 0
        missing_in_fresh = []
        for key, arr in restored_params.items():
            if any(p in str(key).lower() for p in ["lora_a", "lora_b", "lora_alpha", "lora_dropout"]):
                if key in fresh_flat:
                    fresh_flat[key] = fresh_flat[key].replace(value=jnp.asarray(arr))
                    overlay_count += 1
                else:
                    missing_in_fresh.append(key)
        logger.info(
            f"LoRA overlay: {overlay_count}/{len(restored_params)} params copied"
            + (f", {len(missing_in_fresh)} missing in fresh" if missing_in_fresh else "")
        )
        del restored_params
        fresh_params = nnx.State.from_flat_path(list(fresh_flat.items()))
        del fresh_flat
        import gc; gc.collect()
        logger.info(f"[MEM] After LoRA overlay: {_mem_usage()}")
        # Match Phase 1 dtype policy: frozen base params → bfloat16, LoRA stays float32.
        fresh_params = nnx_utils.state_map(
            fresh_params, config.freeze_filter,
            lambda p: p.replace(p.value.astype(jnp.bfloat16)),
        )
        nnx.update(fresh_model, fresh_params)
        del fresh_params
        import gc; gc.collect()
        graphdef2, state2 = nnx.split(fresh_model)
        logger.info(f"[MEM] After nnx.split (graphdef2, state2): {_mem_usage()}")
        trainable_params = state2.filter(config.trainable_filter)
        tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
        opt_state = tx.init(trainable_params)
        if not parsed_args.reset_phase2_optimizer:
            logger.info("Restoring Phase-1 Adam moments; Phase-2 schedule counters remain fresh")
            opt_state = _restore_phase1_optimizer_moments(
                pathlib.Path(parsed_args.resume_state), opt_state
            )
        else:
            logger.warning("Resetting Phase-2 optimizer state (explicit paper deviation)")
        logger.info(f"[MEM] After tx.init: {_mem_usage()}")
        train_state = training_utils.TrainState(
            step=0, params=state2, model_def=graphdef2,
            tx=tx, opt_state=opt_state, ema_decay=None, ema_params=None,
        )
        # state_sharding was computed against the placeholder train_state from
        # init_train_state() at line 816, which used a *different* tx instance.
        # Recompute against the real Phase 2 train_state so jit shardings match.
        train_state_shape = jax.eval_shape(lambda: train_state)
        state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=False)
        del train_state_shape

    if parsed_args.phase == 1 and parsed_args.resume_state_phase1 is not None:
        # Phase 1 resume: load LoRA params + Adam moments from a previous Phase 1
        # final/resume_state checkpoint, reset step=0 so a fresh CosineDecay schedule
        # (peak_lr=5e-5 → 1e-7 over num_steps) starts from the beginning.
        # Pattern follows Phase 2's overlay approach (lines 850-911) — using
        # _restore_phase1_state would strip params to LoRA-only and break the
        # subsequent Adam moments transplant.
        resume_path = pathlib.Path(parsed_args.resume_state_phase1)
        logger.info(f"[RESUME P1] Restoring LoRA params + Adam moments from {resume_path}")
        # 1. Restore LoRA params as {key_tuple: array} dict.
        restored_params = _restore_phase1_params(resume_path, train_state.params, config)
        # 2. Overlay onto current params (which already have pi05_base loaded by
        #    init_train_state). flat_state() returns a copy, so we modify in-place
        #    and rebuild via nnx.State.from_flat_path.
        fresh_flat = train_state.params.flat_state()
        overlay_count = 0
        missing = []
        for key, arr in restored_params.items():
            if key in fresh_flat:
                fresh_flat[key] = fresh_flat[key].replace(value=jnp.asarray(arr))
                overlay_count += 1
            else:
                missing.append(key)
        logger.info(
            f"[RESUME P1] LoRA overlay: {overlay_count}/{len(restored_params)} params copied"
            + (f", {len(missing)} missing" if missing else "")
        )
        new_params = nnx.State.from_flat_path(list(fresh_flat.items()))
        del fresh_flat, restored_params
        import gc; gc.collect()
        # 3. Build fresh opt_state with new schedule. The new tx replaces the one
        #    from init_train_state (which used the same config.lr_schedule, but
        #    rebuilding ensures the opt_state struct matches what
        #    _restore_phase1_optimizer_moments expects).
        tx_new = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
        trainable_params = new_params.filter(config.trainable_filter)
        opt_state = tx_new.init(trainable_params)
        # 4. Transplant Adam moments from previous run (mu, nu trees). The fresh
        #    opt_state's scalar step counter (used by the new schedule) is preserved.
        opt_state = _restore_phase1_optimizer_moments(resume_path, opt_state)
        # 5. Update train_state. Replace tx too so train_step uses the new schedule.
        train_state = dataclasses.replace(
            train_state, step=0, params=new_params, opt_state=opt_state, tx=tx_new,
        )
        del new_params
        import gc; gc.collect()
        # 6. Recompute sharding against the resumed train_state.
        train_state_shape = jax.eval_shape(lambda: train_state)
        state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=False)
        del train_state_shape
        logger.info(
            "[RESUME P1] Done. LoRA params + Adam moments restored; step=0; "
            f"new schedule: peak_lr=5e-5 decay over {num_steps} steps"
        )


    topology = analyze_lora_topology(train_state)
    logger.info(f"LoRA: {topology['num_lora_modules']} modules, {topology['total_lora_params']:,} params")

    # Debug: report GPU memory allocated at each stage.
    logger.info(f"[MEM] After train_state init: {_mem_usage()}")

    logger.info("Setting up data loader...")
    try:
        data_loader = _data_loader.create_data_loader(
            config, sharding=data_sharding, shuffle=True, skip_norm_stats=False,
        )
        data_iter = iter(data_loader)
        batch = next(data_iter)
        obs, actions = batch
        action_values = np.asarray(jax.device_get(actions))
        dynamic_chunks = np.any(np.diff(action_values, axis=1) != 0, axis=(1, 2))
        if action_values.shape[1] > 1 and not bool(dynamic_chunks.any()):
            raise RuntimeError(
                "Every action chunk in the first batch is temporally constant. "
                "Check LeRobot dataset_from/to indices before training."
            )
        logger.info(f"Data loader OK - batch: actions={actions.shape}, images={list(obs.images.keys())}")
        logger.info(
            f"Data preflight: {int(dynamic_chunks.sum())}/{len(dynamic_chunks)} "
            "first-batch action chunks contain temporal variation"
        )
    except Exception as e:
        raise RuntimeError(
            "Real action-labelled demonstrations are required for expert training; "
            "refusing the old mock-data fallback"
        ) from e

    logger.info(f"Training for {num_steps} steps...")
    # Pre-compile the train step with JIT
    import functools
    if parsed_args.phase == 2:
        # Force a full GC + JAX cache clear before Phase 2 JIT.
        # The Phase 2 path avoids loading base weights twice (see above), so
        # at this point the arena holds only model+opt state (~13.97 GB).
        # Clearing caches gives XLA the best chance to find a contiguous block.
        import gc as gc_module
        gc_module.collect()
        try:
            jax.clear_caches()
        except AttributeError:
            pass
        try:
            import jax._src.xla_bridge as xb
            backend = xb.get_backend()
            if hasattr(backend, '_allocator') and hasattr(backend._allocator, 'reset'):
                backend._allocator.reset()
        except Exception:
            pass
        logger.info(f"[MEM] Pre-JIT: {_mem_usage()}")
    p_train_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    # Step-0 diagnostic: compute loss + param_norm WITHOUT taking a gradient step.
    # Critical for Phase 2 — verifies the bf16 fix produces a continuous handoff
    # from Phase 1 final state. Expected: Phase 2 step-0 loss ≈ Phase 1 final loss.
    def _init_metrics(state, batch):
        m = nnx.merge(state.model_def, state.params)
        m.train()
        obs, acts = batch
        rng0 = jax.random.fold_in(jax.random.PRNGKey(0), state.step)
        loss = jnp.mean(m.compute_loss(rng0, obs, acts, train=True))
        kp = nnx.state(
            m,
            nnx.All(
                nnx.Param,
                nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
                lambda _, x: x.value.ndim > 1,
            ),
        )
        # Isolate LoRA-only norm to detect overlay bugs.
        lora_only = nnx.state(m, nnx.All(nnx.Param, nnx_utils.PathRegex(".*lora.*")))
        # And frozen-base-only norm (everything kernel-shaped that's not LoRA).
        base_only = nnx.state(
            m,
            nnx.All(
                nnx.Param,
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
                nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
                lambda _, x: x.value.ndim > 1,
            ),
        )
        return loss, optax.global_norm(kp), optax.global_norm(lora_only), optax.global_norm(base_only)
    _init_metrics_jit = jax.jit(
        _init_metrics,
        in_shardings=(state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    init_loss, init_pnorm, init_lora_norm, init_base_norm = _init_metrics_jit(train_state, batch)
    logger.info(
        f"[INIT] step=0 loss={float(init_loss):.4f} param_norm={float(init_pnorm):.4f} "
        f"lora_norm={float(init_lora_norm):.4f} base_norm={float(init_base_norm):.4f}"
        + (f" (Phase 1 final: loss~0.012 pnorm~1803.01 lora~47.7)" if parsed_args.phase == 2 else "")
        + (" (Phase 1 RESUME — should match previous Phase 1 final loss)" if (parsed_args.phase == 1 and parsed_args.resume_state_phase1) else "")
    )

    # Warmup compilation with first batch. Capture step-0 metrics instead of
    # discarding them — first-update finite check is critical for distinguishing
    # "NaN at step 0" from "NaN accumulated over dozens of steps".
    logger.info("Compiling train_step (this may take 1-2 minutes)...")
    rng = jax.random.PRNGKey(0)
    with sharding.set_mesh(mesh):
        train_state, info_step0 = p_train_step(rng, train_state, batch)
    logger.info("Compilation complete. Starting training loop.")

    def _log_step(step: int, info: dict, *, force_detail: bool = False):
        """Log per-step metrics. First `detail_steps` steps always log full detail."""
        detail = force_detail or step < DIAGNOSTIC_STEPS
        loss_v = float(info["loss"])
        grad_v = float(info["grad_norm"])
        finite_v = bool(info["all_finite"])
        if detail:
            logger.info(
                f"Step {step}/{num_steps} | loss: {loss_v:.4f} | grad: {grad_v:.4f} | "
                f"pnorm: {float(info['param_norm']):.4f} | finite: {finite_v} | "
                f"vision_g: {float(info['grad_norm_vision']):.4f} | "
                f"gemma_g: {float(info['grad_norm_gemma_llm']):.4f} | "
                f"act_time_g: {float(info['grad_norm_action_time']):.4f} | "
                f"lm_head_g: {float(info['grad_norm_lm_head']):.4f}"
            )
        else:
            logger.info(
                f"Step {step}/{num_steps} | loss: {loss_v:.4f} | grad: {grad_v:.4f} | "
                f"pnorm: {float(info['param_norm']):.4f}"
            )

    # A non-finite update cannot recover reliably: Adam moments and LoRA
    # parameters may already contain NaN/Inf.  Abort before saving a checkpoint
    # that merely looks like a completed expert.
    if not bool(info_step0["all_finite"]):
        raise FloatingPointError(
            f"Step 0 produced non-finite values: loss={float(info_step0['loss'])} "
            f"grad={float(info_step0['grad_norm'])} "
            f"vision_g={float(info_step0['grad_norm_vision'])} "
            f"gemma_g={float(info_step0['grad_norm_gemma_llm'])} "
            f"act_time_g={float(info_step0['grad_norm_action_time'])} "
            f"lm_head_g={float(info_step0['grad_norm_lm_head'])}"
        )
    _log_step(0, info_step0, force_detail=True)

    metrics_path = output_dir / "metrics.csv"
    with open(metrics_path, "w") as f:
        f.write("step,loss,grad_norm,param_norm,all_finite,grad_vision,grad_gemma,grad_act_time,grad_lm_head\n")
        f.write(
            f"0,{float(info_step0['loss']):.6f},{float(info_step0['grad_norm']):.6f},"
            f"{float(info_step0['param_norm']):.4f},{int(bool(info_step0['all_finite']))},"
            f"{float(info_step0['grad_norm_vision']):.6f},{float(info_step0['grad_norm_gemma_llm']):.6f},"
            f"{float(info_step0['grad_norm_action_time']):.6f},{float(info_step0['grad_norm_lm_head']):.6f}\n"
        )

    for step in tqdm.tqdm(range(1, num_steps)):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)

        rng = jax.random.PRNGKey(step)
        with sharding.set_mesh(mesh):
            train_state, info = p_train_step(rng, train_state, batch)

        # Stop immediately rather than propagating a poisoned optimizer state.
        if not bool(info["all_finite"]):
            raise FloatingPointError(
                f"Step {step} produced non-finite values: "
                f"loss={float(info['loss'])} grad={float(info['grad_norm'])} "
                f"vision_g={float(info['grad_norm_vision'])} "
                f"gemma_g={float(info['grad_norm_gemma_llm'])} "
                f"act_time_g={float(info['grad_norm_action_time'])} "
                f"lm_head_g={float(info['grad_norm_lm_head'])}"
            )

        if step < DIAGNOSTIC_STEPS or step % config.log_interval == 0:
            _log_step(step, info, force_detail=(step < DIAGNOSTIC_STEPS))
            with open(metrics_path, "a") as f:
                f.write(
                    f"{step},{float(info['loss']):.6f},{float(info['grad_norm']):.6f},"
                    f"{float(info['param_norm']):.4f},{int(bool(info['all_finite']))},"
                    f"{float(info['grad_norm_vision']):.6f},{float(info['grad_norm_gemma_llm']):.6f},"
                    f"{float(info['grad_norm_action_time']):.6f},{float(info['grad_norm_lm_head']):.6f}\n"
                )

        if step > 0 and step % config.save_interval == 0:
            ckpt_path = output_dir / f"step_{step}"
            ckpt_path.mkdir(exist_ok=True)
            _save_params(ckpt_path / "params", train_state, config, base_checkpoint=parsed_args.base_checkpoint)
            _save_resume_state(ckpt_path / "resume_state", train_state, config)
            logger.info(f"Checkpoint saved to {ckpt_path}")

            if config.keep_period:
                existing = sorted(
                    int(p.name[5:]) for p in output_dir.glob("step_*") if p.name[5:].isdigit()
                )
                if existing:
                    latest = existing[-1]
                    to_delete = [s for s in existing if s != latest and s % config.keep_period != 0]
                    for s in to_delete:
                        target = output_dir / f"step_{s}"
                        shutil.rmtree(target, ignore_errors=True)
                        logger.info(f"[keep_period={config.keep_period}] removed intermediate ckpt {target.name}")

    final_path = output_dir / "final"
    final_path.mkdir(exist_ok=True)
    _save_params(final_path / "params", train_state, config, save_canonical=True, base_checkpoint=parsed_args.base_checkpoint)
    _save_resume_state(final_path / "resume_state", train_state, config)
    logger.info(f"Training complete. Final checkpoint: {final_path}")
if __name__ == "__main__":
    main()
