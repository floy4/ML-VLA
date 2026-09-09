# src/mlvla/meta/e2e/jax_backend.py
"""Frozen JAX pi05 base + differentiable LoRA-slot loss for e2e meta-net training.

One process holds torch (meta-net) and JAX (base) together; env vars must be set
before importing mlvla.experts.train (done at module import here).

GPU correctness (loss finite, grads nonzero through lora slots, loader round
trip) is validated by the Task 7 smoke job; this module only wires it up.
"""
from __future__ import annotations

import dataclasses
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from pathlib import Path
from typing import Iterator, Mapping

from mlvla import paths as _paths

_paths.set_derived_env()
_paths.add_to_sys_path()

import sys

sys.path.insert(0, str(Path(_paths.get("openpi_root")) / "scripts"))

import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx

from mlvla.meta.e2e.lora_mapping import LoRAMapping, build_mapping


def _collect_lora_paths(flat: Mapping[tuple, object]) -> dict[tuple[str, ...], tuple[int, ...]]:
    """Flat nnx state -> {path: shape} for leaves whose path contains 'lora'.

    Values may be nnx VarState leaves (``.value.shape``, from ``flat_state()``)
    or bare shape containers (plain tuples in tests); both normalize to
    {tuple-of-str path: tuple-of-int shape}.
    """
    out: dict[tuple[str, ...], tuple[int, ...]] = {}
    for path, var in flat.items():
        if "lora" not in "/".join(map(str, path)).lower():
            continue
        value = getattr(var, "value", var)
        out[tuple(map(str, path))] = tuple(getattr(value, "shape", value))
    return out


class JAXBackend:
    """Loads the frozen pi05 base once and exposes a differentiable LoRA-slot loss.

    Attributes:
        mapping: LoRAMapping validated against the real model's lora tensor
            shapes (458 modules / 916 sites).
        lora_paths: fixed ordering of ``mapping.tensors`` keys (tuple paths).
        loss_and_grad: jitted value_and_grad over a {path_tuple: array} pytree.
        val_loss: jitted loss without gradients.
    """

    def __init__(self, template_npz: Path, batch_size: int = 8) -> None:
        # Heavy imports (full openpi model + patched expert trainer) stay inside
        # __init__ so a plain module import remains cheap for CPU tests.
        import openpi.models.model as _model
        import openpi.training.sharding as sharding
        from mlvla.experts import train as tre

        self._tre = tre
        self._model_module = _model
        config = tre.get_train_config(
            task_name="e2e_meta_net", output_dir=Path("/tmp/mlvla_e2e_unused"),
            num_steps=1, phase=1, batch_size=batch_size)
        num_fsdp = int(os.environ.get("MLVLA_NUM_FSDP_DEVICES", "1"))
        self.mesh = sharding.make_mesh(num_fsdp_devices=num_fsdp)
        train_state, _ = tre.init_train_state(config, jax.random.PRNGKey(42), self.mesh)
        model = nnx.merge(train_state.model_def, train_state.params)
        model.train()
        self.graphdef, self.state = nnx.split(model)

        # build_mapping validates every canonical module against the real model
        # shapes; a raise here is a genuine topology bug, never suppressed.
        lora_shapes = _collect_lora_paths(self.state.flat_state())
        self.mapping: LoRAMapping = build_mapping(template_npz, lora_shapes)
        self.lora_paths: tuple[tuple[str, ...], ...] = tuple(self.mapping.tensors)

        def _replace_lora(lora_values):
            # flat_state() returns a copy; rebuild via nnx.State.from_flat_path.
            # lora_values is keyed by tuple paths (ruling B), matching
            # mapping.tensors / assemble() output.
            items = []
            for path, var in self.state.flat_state().items():
                key = tuple(map(str, path))
                if key in lora_values:
                    items.append((path, var.replace(
                        value=jnp.asarray(lora_values[key]).astype(var.value.dtype))))
                else:
                    items.append((path, var))
            return nnx.merge(self.graphdef, nnx.State.from_flat_path(items))

        def _loss(lora_values, rng, obs, act):
            m = _replace_lora(lora_values)
            m.train()
            return jnp.mean(m.compute_loss(rng, obs, act, train=True)).astype(jnp.float32)

        self._loss_fn = _loss
        # Explicit in_shardings (mirroring openpi train.py ptrain_step): without
        # them, a >1-device jit cannot reconcile data-sharded obs/act with
        # device-0-committed lora/rng arrays (UnspecifiedValue crash on
        # _array_shard_arg). Single-device runs are semantically unchanged.
        replicated = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())
        data_shard = jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        in_shards = (replicated, replicated, data_shard, data_shard)

        # LoRA values enter as ONE flat replicated arg instead of 916 small
        # dict args: jax 0.5.3's SPMD partitioner produces NaN on multi-device
        # runs when many small replicated traced args are merged into the
        # model state inside the jit while obs/act are data-sharded (probe8/9:
        # in-jit merge with lora args -> nan; same computation with lora as
        # constants, or one packed arg -> finite, identical values). Unpacked
        # back into a per-path dict before returning so callers are unchanged.
        paths = tuple(self.mapping.tensors)
        shapes = [tuple(self.mapping.tensors[p]) for p in paths]
        sizes = [int(np.prod(s)) for s in shapes]

        def _pack(d):
            return jnp.concatenate([jnp.asarray(d[p]).reshape(-1) for p in paths])

        def _unpack(flat):
            out, off = {}, 0
            for p, s, sz in zip(paths, sizes, shapes):
                out[p] = jnp.reshape(flat[off:off + s], sz)
                off += s
            return out

        def _loss_packed(flat, rng, obs, act):
            return _loss(_unpack(flat), rng, obs, act)

        grad_packed = jax.jit(jax.value_and_grad(_loss_packed, argnums=0),
                              in_shardings=in_shards,
                              out_shardings=(replicated, replicated))
        val_packed = jax.jit(_loss_packed, in_shardings=in_shards,
                             out_shardings=replicated)

        def loss_and_grad(lora_values, rng, obs, act):
            loss, gflat = grad_packed(_pack(lora_values), rng, obs, act)
            return loss, _unpack(gflat)

        def val_loss(lora_values, rng, obs, act):
            return val_packed(_pack(lora_values), rng, obs, act)

        self.loss_and_grad = loss_and_grad
        self.val_loss = val_loss
        # Exposed for flat-buffer fast paths (train_bridge) and timing probes:
        # single-arg jits + the pack layout.
        self.grad_packed = grad_packed
        self.val_packed = val_packed
        self.lora_flat_sizes = sizes
        self.lora_flat_shapes = shapes

    def domain_loader(self, dataset_root: Path, task: str,
                      episode_ids: set[int] | None, batch_size: int = 8,
                      seed: int | None = None) -> Iterator:
        """Iterator over (obs, actions) jnp batches for one perturbed domain.

        Sets the perturbed-root env var and the trainer module globals before
        constructing the loader: create_data_loader builds its dataset eagerly,
        so _TARGET_TASK/_EPISODE_FILTER are consumed at call time (Task 3 hook).
        ``seed`` overrides the config shuffle seed (per-rank data streams for
        multi-process data parallel; None keeps the config default).
        """
        import openpi.training.data_loader as data_loader
        import openpi.training.sharding as sharding

        os.environ["MLVLA_PERTURBED_DATASET_ROOT"] = str(dataset_root)
        self._tre._TARGET_TASK = task
        self._tre._EPISODE_FILTER = episode_ids
        config = self._tre.get_train_config(
            task_name=f"e2e_{dataset_root.name}", output_dir=Path("/tmp/mlvla_e2e_unused"),
            num_steps=1, phase=1, batch_size=batch_size)
        if seed is not None:
            config = dataclasses.replace(config, seed=seed)
        # Same construction as expert training (train.py main): DATA_AXIS spans
        # batch+fsdp mesh axes; with num_fsdp_devices=1 it shards across devices.
        data_sharding = jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        loader = data_loader.create_data_loader(
            config, sharding=data_sharding, shuffle=True, skip_norm_stats=False)
        return iter(loader)
