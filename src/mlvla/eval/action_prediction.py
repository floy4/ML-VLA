"""Open-loop action prediction evaluation for Wizard/OpenPI checkpoints.

Evaluates action prediction accuracy on random samples from LIBERO dataset.
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_GPU_ALLOCATOR", "bin-boost")
os.environ.setdefault("WANDB_MODE", "disabled")

import argparse
import dataclasses
import functools
import json
import pathlib
import sys

from mlvla import paths as _paths

_paths.set_derived_env()
_paths.add_to_sys_path()
sys.path.insert(0, str(pathlib.Path(_paths.get("openpi_root")) / "scripts"))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import flax.traverse_util as traverse_util
from etils import epath
import orbax.checkpoint as ocp

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.config as _config
import openpi.training.sharding as sharding
import openpi.training.data_loader as _data_loader
import openpi.training.weight_loaders as weight_loaders

import lerobot.datasets.lerobot_dataset as _lerobot_dataset
_lerobot_dataset.CODEBASE_VERSION = "v3.0"

import mlvla.experts.train as tre


def make_eval_config(checkpoint_type: str, task: str, batch_size: int, base_checkpoint: str):
    """Return the exact training architecture for the requested checkpoint."""
    if checkpoint_type == "wizard":
        return tre.get_train_config(
            task_name=task,
            output_dir=pathlib.Path("/tmp/wizard_eval_unused"),
            num_steps=1,
            phase=1,
            batch_size=batch_size,
            base_checkpoint=base_checkpoint,
        )

    from openpi.models import pi0_config
    base = _config.get_config("pi0_libero_low_mem_finetune")
    model_config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    return dataclasses.replace(
        base,
        name="pi05_libero_lora_goal_expert_eval",
        model=model_config,
        data=tre.get_data_config(base_checkpoint),
        freeze_filter=model_config.get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(f"{base_checkpoint}/params"),
        batch_size=batch_size,
        num_workers=0,
        ema_decay=None,
    )


def _load_checkpoint_fast(params_path: pathlib.Path, checkpoint_type: str, restore_type=jnp.ndarray):
    """Fast checkpoint restore."""
    print(f"Loading checkpoint from {params_path}...")

    if checkpoint_type == "openpi":
        from openpi.models import pi0_config
        model_config = pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        )
        model = model_config.create(jax.random.key(0))
        state = nnx.state(model)
        loader = weight_loaders.CheckpointWeightLoader(str(params_path))
        loaded = loader.load(state.to_pure_dict())
        partial = traverse_util.unflatten_dict({
            k: v for k, v in traverse_util.flatten_dict(loaded).items()
            if not isinstance(v, jax.ShapeDtypeStruct)
        })
        print(f"OpenPI checkpoint loaded. Loaded {len(traverse_util.flatten_dict(partial))} params")
        return partial
    else:
        with ocp.PyTreeCheckpointer() as ckptr:
            metadata = ckptr.metadata(params_path)
            mesh = jax.sharding.Mesh(jax.devices(), ("x",))
            sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
            item = metadata
            restore_args = jax.tree.map(
                lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type),
                item,
            )
            result = ckptr.restore(
                params_path,
                ocp.args.PyTreeRestore(item=item, restore_args=restore_args),
            )
        print(f"Wizard checkpoint loaded. Top-level keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
        return result


def _overlay_params(target_state, loaded_params):
    """Overlay loaded params onto fresh train_state params."""
    fresh_flat = target_state.params.flat_state()

    if isinstance(loaded_params, dict):
        loaded_flat = traverse_util.flatten_dict(loaded_params)
    else:
        loaded_flat = {}
        for k, v in loaded_params.flat_state().items():
            loaded_flat[tuple(str(p) for p in k)] = getattr(v, "value", v)

    def _strip(key):
        key = list(key)
        while key and str(key[0]) == "PaliGemma":
            key.pop(0)
        return tuple(key)

    fresh_by_stripped = {_strip(tuple(str(p) for p in k)): (k, v) for k, v in fresh_flat.items()}

    overlay = 0
    skip_shape = 0
    for ck, cv in loaded_flat.items():
        ck_tuple = ck if isinstance(ck, tuple) else (ck,)
        stripped = _strip(ck_tuple)
        if stripped in fresh_by_stripped:
            orig_k, fv = fresh_by_stripped[stripped]
            arr = jnp.asarray(cv)
            if hasattr(fv, "value") and hasattr(arr, "shape"):
                if arr.shape == fv.value.shape:
                    fresh_flat[orig_k] = fv.replace(value=arr)
                    overlay += 1
                else:
                    skip_shape += 1

    new_params = nnx.State.from_flat_path(list(fresh_flat.items()))
    return dataclasses.replace(target_state, params=new_params), overlay, skip_shape


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="put the bowl on top of the cabinet")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--base_checkpoint", type=str,
                        default=_paths.get("base_checkpoint"))
    parser.add_argument("--checkpoint_type", choices=["wizard", "openpi"], required=True)
    parser.add_argument("--params_dir", type=str, required=True)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    num_devices = jax.device_count()
    assert num_devices == 1, f"Single-GPU only, got {num_devices}"
    print(f"Devices: {num_devices}")

    cfg = make_eval_config(
        args.checkpoint_type, args.task, args.batch_size, args.base_checkpoint
    )
    cfg = dataclasses.replace(cfg, num_workers=0)

    # CRITICAL: set target task BEFORE creating data_loader, otherwise the
    # patched LeRobotDataset loads ALL libero_goal episodes (every task),
    # and action-prediction metrics average over off-task samples.
    tre._TARGET_TASK = args.task
    print(f"Set _TARGET_TASK={tre._TARGET_TASK!r} (filters data to this task only)")

    mesh = sharding.make_mesh(num_fsdp_devices=1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize model
    init_rng = jax.random.PRNGKey(42)
    print("Initializing fresh model...")
    train_state, state_sharding = tre.init_train_state(cfg, init_rng, mesh, skip_base_weight_load=False)
    print("Fresh model initialized")

    # Load checkpoint
    params = _load_checkpoint_fast(pathlib.Path(args.params_dir), args.checkpoint_type, restore_type=jnp.ndarray)
    train_state, overlay, skip = _overlay_params(train_state, params)
    loaded_count = len(traverse_util.flatten_dict(params)) if isinstance(params, dict) else len(params.flat_state())
    if overlay != loaded_count or skip:
        raise RuntimeError(
            f"Partial restore rejected: restored={overlay}/{loaded_count}, shape_mismatches={skip}"
        )
    print(f"Strict restore succeeded: {overlay}/{loaded_count} params")

    # Load samples
    print(f"Loading {args.num_samples} samples...")
    loader = _data_loader.create_data_loader(
        cfg, sharding=data_sharding, shuffle=True, skip_norm_stats=False,
    )
    it = iter(loader)

    # Collect samples
    all_obs = []
    all_acts_gt = []
    collected = 0
    while collected < args.num_samples:
        batch = next(it)
        obs, acts = batch
        batch_size = acts.shape[0]
        needed = min(batch_size, args.num_samples - collected)
        all_obs.append(jax.tree.map(lambda x: x[:needed], obs))
        all_acts_gt.append(acts[:needed])
        collected += needed

    obs = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *all_obs)
    acts_gt = jnp.concatenate(all_acts_gt, axis=0)
    print(f"Collected {acts_gt.shape[0]} samples")

    # Predict actions
    print("Predicting actions...")
    rng = jax.random.PRNGKey(0)

    # Rebuild model from train_state
    model = nnx.merge(train_state.model_def, train_state.params)

    # Sample actions (sample_actions is already JIT-friendly)
    with sharding.set_mesh(mesh):
        acts_pred = model.sample_actions(rng, obs, num_steps=10)

    print(f"Predicted actions shape: {acts_pred.shape}")
    print(f"GT actions shape: {acts_gt.shape}")

    # Compute metrics
    mse_per_sample = jnp.mean(jnp.square(acts_pred - acts_gt), axis=(1, 2))  # [N]
    l2_per_sample = jnp.sqrt(jnp.sum(jnp.square(acts_pred - acts_gt), axis=(1, 2)))  # [N]

    mean_mse = float(jnp.mean(mse_per_sample))
    mean_l2 = float(jnp.mean(l2_per_sample))
    min_l2 = float(jnp.min(l2_per_sample))
    max_l2 = float(jnp.max(l2_per_sample))
    std_l2 = float(jnp.std(l2_per_sample))

    print()
    print("=" * 60)
    print(f"Action Prediction Evaluation Results ({args.num_samples} samples)")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Checkpoint type: {args.checkpoint_type}")
    print(f"Checkpoint: {args.params_dir}")
    print()
    print(f"Mean MSE:  {mean_mse:.6f}")
    print(f"Mean L2:  {mean_l2:.6f}")
    print(f"Min L2:   {min_l2:.6f}")
    print(f"Max L2:   {max_l2:.6f}")
    print(f"Std L2:   {std_l2:.6f}")
    print("=" * 60)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "checkpoint_type": args.checkpoint_type,
            "task": args.task,
            "num_samples": args.num_samples,
            "mean_mse": mean_mse,
            "mean_l2": mean_l2,
            "min_l2": min_l2,
            "max_l2": max_l2,
            "std_l2": std_l2,
            "per_sample_mse": [float(x) for x in mse_per_sample],
            "per_sample_l2": [float(x) for x in l2_per_sample],
        }, indent=2) + "\n")
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
