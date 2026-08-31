"""Forward-only train-set loss: expert (goal v3) vs base pi05.

Iterates over the training data_loader (same loss formulation as train_step)
WITHOUT updating params, and reports mean per-batch loss for both the loaded
expert and the frozen pi05 base. The gap between them shows how much the LoRA
expert has fit the training distribution.

Single-GPU only.
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_GPU_ALLOCATOR", "bin-boost")

import argparse
import dataclasses
import functools
import json
import pathlib
import sys
import time

from mlvla import paths as _paths

_paths.set_derived_env()
_paths.add_to_sys_path()

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import openpi.shared.array_typing as at
import openpi.training.sharding as sharding
import openpi.training.data_loader as _data_loader

import mlvla.experts.train as tre


def _restore_expert_into_state(train_state, resume_path: pathlib.Path, cfg):
    restored_params = tre._restore_phase1_params(resume_path, train_state.params, cfg)
    fresh_flat = train_state.params.flat_state()
    overlay = 0
    skip = 0
    for key, arr in restored_params.items():
        if key in fresh_flat:
            arr = jnp.asarray(arr)
            if hasattr(fresh_flat[key], "value") and hasattr(arr, "shape"):
                if arr.shape == fresh_flat[key].value.shape:
                    fresh_flat[key] = fresh_flat[key].replace(value=arr)
                    overlay += 1
                else:
                    skip += 1
    new_params = nnx.State.from_flat_path(list(fresh_flat.items()))
    print(f"Overlaid {overlay} LoRA tensors" + (f" (skipped {skip} shape mismatches)" if skip else ""))
    return dataclasses.replace(train_state, params=new_params)


def _make_eval_loss(cfg, state, batch):
    """Forward-only loss using model.compute_loss (no grad, no optimizer step).

    Note: the model MUST be rebuilt inside _fn from `state.params`. Building it
    once outside _fn (closure capture) freezes the base params and the expert
    LoRA overlay is silently ignored — every call reports the BASE loss.
    """
    @functools.wraps(tre.train_step)
    def _fn(rng, state, batch):
        observation, actions = batch
        train_rng = jax.random.fold_in(rng, state.step)
        model = nnx.merge(state.model_def, state.params)
        model.eval()
        chunked_loss = model.compute_loss(train_rng, observation, actions, train=False)
        return jnp.mean(chunked_loss)

    return _fn


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="put the bowl on top of the cabinet")
    parser.add_argument("--resume_state", type=pathlib.Path, required=True)
    parser.add_argument("--base_checkpoint", type=str,
                        default=_paths.get("base_checkpoint"))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_batches", type=int, default=50,
                        help="Number of train batches to evaluate (default covers 46 episodes)")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    num_devices = jax.device_count()
    assert num_devices == 1, f"Single-GPU only, got {num_devices}"

    cfg = tre.get_train_config(
        task_name=args.task,
        output_dir=pathlib.Path("/tmp/dummy"),
        num_steps=1,
        phase=1,
        batch_size=args.batch_size,
        base_checkpoint=args.base_checkpoint,
    )

    # CRITICAL: set target task BEFORE creating data_loader, otherwise the
    # patched LeRobotDataset will load ALL libero_goal episodes (every task),
    # not just this expert's task. Without this, expert loss == base loss
    # because the LoRA only fits one task and the eval averages over all 10.
    tre._TARGET_TASK = args.task
    print(f"Set _TARGET_TASK={tre._TARGET_TASK!r} (filters data to this task only)")

    mesh = sharding.make_mesh(num_fsdp_devices=1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    print(f"Building train_state (jit init, zero-LoRA = base pi05)...")
    init_rng = jax.random.PRNGKey(42)
    train_state, state_sharding = tre.init_train_state(cfg, init_rng, mesh)

    eval_fn = _make_eval_loss(cfg, train_state, None)
    p_eval = jax.jit(
        eval_fn,
        in_shardings=(replicated_sharding, state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    print(f"Creating data_loader (shuffle=True, but seeded)...")
    loader = _data_loader.create_data_loader(
        cfg, sharding=data_sharding, shuffle=True, skip_norm_stats=False,
    )
    it = iter(loader)

    # ---- Pass 1: evaluate BASE (zero-LoRA train_state = pi05 base) ----
    print(f"Compiling forward loss + evaluating BASE (1-2 min)...")
    rng = jax.random.PRNGKey(0)
    batch = next(it)
    with sharding.set_mesh(mesh):
        t0 = time.time()
        base_loss = float(p_eval(rng, train_state, batch))
        compile_time = time.time() - t0
    print(f"Compile + first batch: {compile_time:.1f}s  base={base_loss:.6f}")

    base_losses = [base_loss]
    t0 = time.time()
    with sharding.set_mesh(mesh):
        for step in range(1, args.num_batches):
            rng, _ = jax.random.split(rng)
            batch = next(it)
            base_losses.append(float(p_eval(rng, train_state, batch)))
            if step % 10 == 0:
                b_avg = np.mean(base_losses[-10:])
                print(f"  [base] step {step}/{args.num_batches}  base(recent10)={b_avg:.6f}")
    base_elapsed = time.time() - t0
    print(f"Base eval done: mean={np.mean(base_losses):.6f} ({base_elapsed:.1f}s)")

    # ---- Pass 2: overlay expert LoRA, evaluate EXPERT (no recompile) ----
    print(f"\nRestoring expert from {args.resume_state}")
    train_state = _restore_expert_into_state(train_state, args.resume_state, cfg)

    rng = jax.random.PRNGKey(0)
    it = iter(loader)  # restart iterator for same data order
    batch = next(it)
    print(f"Evaluating EXPERT (jit cached, no recompile)...")
    with sharding.set_mesh(mesh):
        t0 = time.time()
        expert_loss = float(p_eval(rng, train_state, batch))
        first_step_time = time.time() - t0
    print(f"First expert batch: {first_step_time:.1f}s  expert={expert_loss:.6f}")

    expert_losses = [expert_loss]
    t0 = time.time()
    with sharding.set_mesh(mesh):
        for step in range(1, args.num_batches):
            rng, _ = jax.random.split(rng)
            batch = next(it)
            expert_losses.append(float(p_eval(rng, train_state, batch)))
            if step % 10 == 0:
                e_avg = np.mean(expert_losses[-10:])
                print(f"  [expert] step {step}/{args.num_batches}  expert(recent10)={e_avg:.6f}")
    elapsed = time.time() - t0

    expert_mean = float(np.mean(expert_losses))
    base_mean = float(np.mean(base_losses))
    expert_std = float(np.std(expert_losses))
    base_std = float(np.std(base_losses))
    reduction_pct = (base_mean - expert_mean) / max(base_mean, 1e-12) * 100

    print()
    print("=" * 60)
    print(f"Batches evaluated: {len(expert_losses)} (batch_size={args.batch_size})")
    print(f"Expert loss: {expert_mean:.6f} ± {expert_std:.6f}  (min={min(expert_losses):.6f})")
    print(f"Base   loss: {base_mean:.6f} ± {base_std:.6f}  (min={min(base_losses):.6f})")
    print(f"Reduction:   {reduction_pct:+.1f}%  (expert vs base)")
    print(f"Eval wall:   {elapsed:.1f}s  ({elapsed/len(expert_losses)*1000:.1f}ms/batch)")
    print("=" * 60)
    if expert_mean < 1e-3:
        print("HEAVY OVERFIT: expert loss < 1e-3 on training data")
    elif expert_mean < base_mean * 0.1:
        print("STRONG FIT: expert loss < 10% of base loss")
    elif expert_mean < base_mean * 0.5:
        print("MODERATE FIT: expert loss < 50% of base loss")
    else:
        print("WEAK FIT: expert loss is comparable to or worse than base")

    result = {
        "task": args.task,
        "resume_state": str(args.resume_state),
        "batch_size": args.batch_size,
        "num_batches": len(expert_losses),
        "expert": {"mean": expert_mean, "std": expert_std, "min": float(min(expert_losses)),
                   "all": expert_losses},
        "base": {"mean": base_mean, "std": base_std, "min": float(min(base_losses)),
                 "all": base_losses},
        "reduction_pct": reduction_pct,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
