#!/usr/bin/env python3
"""Probe v3: is the multi-GPU nan triggered by the sharded batch?

Same zero-LoRA loss as probe2, but obs/act are REPLICATED across devices
(no P(DATA_AXIS) sharding). If finite -> sharding-triggered; if nan ->
multi-device execution itself is broken regardless of data sharding.

Run: CUDA_VISIBLE_DEVICES=4,5 conda run --no-capture-output -n openvla \
        python scripts/e2e_4gpu_probe3.py
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml

import jax
import jax.numpy as jnp

from mlvla.meta.e2e.jax_backend import JAXBackend
from mlvla.meta.e2e.lora_mapping import assemble

if os.environ.get("MLVLA_NO_ACT_CONSTRAINT") == "1":
    import openpi.training.sharding as pi_sharding
    pi_sharding.activation_sharding_constraint = lambda pytree: pytree
    print("[patch] activation_sharding_constraint -> no-op", flush=True)


def site_shape(backend: JAXBackend, key: str, factor: str) -> tuple[int, int]:
    for site in backend.mapping.sites:
        if site.key == key and site.factor == factor:
            return tuple(site.canonical_shape)
    raise KeyError(f"{key}/{factor}")


def main() -> None:
    cfg = yaml.safe_load((REPO / "configs/hypernet_e2e_film_compile_test.yaml").read_text())
    domains = yaml.safe_load((REPO / "configs/domains.yaml").read_text())
    d0 = cfg["train_domains"][0]

    backend = JAXBackend(template_npz=Path(domains[d0]["canonical_npz"]), batch_size=8)
    print(f"devices: {jax.device_count()} mesh: {backend.mesh.devices.shape}", flush=True)

    it = backend.domain_loader(Path(domains[d0]["dataset_root"]), cfg["task_prompt"], None, 8)
    obs, act = next(it)

    module_ab = {k: {"A": np.zeros(site_shape(backend, k, "A"), np.float32),
                     "B": np.zeros(site_shape(backend, k, "B"), np.float32)}
                 for k in backend.mapping.keys}
    tensors = dict(assemble(backend.mapping, module_ab))

    replicated = jax.sharding.NamedSharding(backend.mesh, jax.sharding.PartitionSpec())

    obs_rep = jax.device_put(jax.tree.map(jnp.asarray, obs), replicated)
    act_rep = jax.device_put(jax.tree.map(jnp.asarray, act), replicated)
    lora_rep = jax.device_put(jax.tree.map(jnp.asarray, tensors), replicated)
    key_rep = jax.device_put(jax.random.PRNGKey(7), replicated)

    loss_rep = jax.jit(backend._loss_fn, in_shardings=(replicated,) * 4,
                       out_shardings=replicated)(lora_rep, key_rep, obs_rep, act_rep)
    print(f"[replicated-data] loss={float(loss_rep)!r} finite={bool(np.isfinite(float(loss_rep)))}",
          flush=True)

    import openpi.training.sharding as sharding
    data_shard = jax.sharding.NamedSharding(
        backend.mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    obs_sh = jax.device_put(jax.tree.map(jnp.asarray, obs), data_shard)
    act_sh = jax.device_put(jax.tree.map(jnp.asarray, act), data_shard)
    loss_sh = jax.jit(backend._loss_fn, in_shardings=(replicated, replicated, data_shard, data_shard),
                      out_shardings=replicated)(lora_rep, key_rep, obs_sh, act_sh)
    print(f"[sharded-data]    loss={float(loss_sh)!r} finite={bool(np.isfinite(float(loss_sh)))}",
          flush=True)


if __name__ == "__main__":
    main()
