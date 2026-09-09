#!/usr/bin/env python3
"""Probe v6: localize the sharded-batch nan on the REAL model.

Ladder: [1] SigLIP img only -> [2] embed_prefix (siglip+gemma) -> [3] full
compute_loss (known nan). Everything runs on the backend's real merged
model with default (finite, random-init) LoRA, loader batch, data-sharded.

Run: CUDA_VISIBLE_DEVICES=4,5 conda run --no-capture-output -n openvla \
        python scripts/e2e_4gpu_probe6.py
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
from flax import nnx

from mlvla.meta.e2e.jax_backend import JAXBackend


def main() -> None:
    cfg = yaml.safe_load((REPO / "configs/hypernet_e2e_film_compile_test.yaml").read_text())
    domains = yaml.safe_load((REPO / "configs/domains.yaml").read_text())
    d0 = cfg["train_domains"][0]

    backend = JAXBackend(template_npz=Path(domains[d0]["canonical_npz"]), batch_size=8)
    print(f"devices: {jax.device_count()} mesh: {backend.mesh.devices.shape}", flush=True)

    it = backend.domain_loader(Path(domains[d0]["dataset_root"]), cfg["task_prompt"], None, 8)
    obs, act = next(it)

    model = nnx.merge(backend.graphdef, backend.state)

    import openpi.training.sharding as psh
    data_shard = jax.sharding.NamedSharding(backend.mesh, jax.sharding.PartitionSpec(psh.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(backend.mesh, jax.sharding.PartitionSpec())

    cam = next(iter(obs.images))
    print(f"camera key: {cam!r} images shape: {obs.images[cam].shape}", flush=True)

    img_fn = jax.jit(
        lambda im: model.PaliGemma.img(im, train=False)[0],
        in_shardings=data_shard, out_shardings=data_shard)
    out_img = img_fn(obs.images[cam])
    print(f"[1] siglip img finite={bool(jnp.isfinite(out_img).all())} "
          f"max={float(jnp.abs(out_img).max()):.3e}", flush=True)

    prefix_fn = jax.jit(lambda o: model.embed_prefix(o)[0],
                        in_shardings=data_shard, out_shardings=data_shard)
    out_pre = prefix_fn(obs)
    print(f"[2] embed_prefix finite={bool(jnp.isfinite(out_pre).all())} "
          f"max={float(jnp.abs(out_pre).max()):.3e}", flush=True)

    rng = jax.device_put(jax.random.PRNGKey(7), replicated)
    loss_fn = jax.jit(lambda r, o, a: jnp.mean(model.compute_loss(r, o, a, train=True)),
                      in_shardings=(replicated, data_shard, data_shard),
                      out_shardings=replicated)
    loss = loss_fn(rng, obs, act)
    print(f"[3] full loss={float(loss)!r}", flush=True)


if __name__ == "__main__":
    main()
