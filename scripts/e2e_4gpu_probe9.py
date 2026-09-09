#!/usr/bin/env python3
"""Probe v9: discriminate replicated-args-used vs merge-inside-jit.

[c] replicated lora args USED (tiny penalty term, no nnx.merge)
[d] nnx.merge INSIDE jit with lora from CLOSURE (no lora jit args)

Run: CUDA_VISIBLE_DEVICES=4,5 conda run --no-capture-output -n openvla \
        python scripts/e2e_4gpu_probe9.py
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
from mlvla.meta.e2e.lora_mapping import assemble


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

    import openpi.training.sharding as psh
    data_shard = jax.sharding.NamedSharding(backend.mesh, jax.sharding.PartitionSpec(psh.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(backend.mesh, jax.sharding.PartitionSpec())
    rng = jax.device_put(jax.random.PRNGKey(7), replicated)

    module_ab = {k: {"A": np.zeros(site_shape(backend, k, "A"), np.float32),
                     "B": np.zeros(site_shape(backend, k, "B"), np.float32)}
                 for k in backend.mapping.keys}
    tensors = dict(assemble(backend.mapping, module_ab))
    lora_rep = jax.device_put(jax.tree.map(jnp.asarray, tensors), replicated)

    def replace_lora(lora_values):
        items = []
        for path, var in backend.state.flat_state().items():
            key = tuple(map(str, path))
            if key in lora_values:
                items.append((path, var.replace(
                    value=jnp.asarray(lora_values[key]).astype(var.value.dtype))))
            else:
                items.append((path, var))
        return nnx.merge(backend.graphdef, nnx.State.from_flat_path(items))

    model_fixed = replace_lora(tensors)
    model_fixed.train()

    # [c] replicated lora args USED, no merge
    def loss_c(lora_values, r, o, a):
        penalty = sum(jnp.abs(v).sum(dtype=jnp.float32) for v in jax.tree.leaves(lora_values))
        return jnp.mean(model_fixed.compute_loss(r, o, a, train=True)).astype(jnp.float32) + 0.0 * penalty

    in_sh = (replicated, replicated, data_shard, data_shard)
    lc = float(jax.jit(loss_c, in_shardings=in_sh, out_shardings=replicated)(
        lora_rep, rng, obs, act))
    print(f"[c args-used-no-merge] loss={lc!r} finite={bool(np.isfinite(lc))}", flush=True)

    # [d] merge INSIDE jit, lora via closure constants
    def loss_d(r, o, a):
        m = replace_lora(tensors)
        m.train()
        return jnp.mean(m.compute_loss(r, o, a, train=True)).astype(jnp.float32)

    ld = float(jax.jit(loss_d, in_shardings=(replicated, data_shard, data_shard),
                       out_shardings=replicated)(rng, obs, act))
    print(f"[d merge-inside-const] loss={ld!r} finite={bool(np.isfinite(ld))}", flush=True)


if __name__ == "__main__":
    main()
