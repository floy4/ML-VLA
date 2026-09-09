#!/usr/bin/env python3
"""Probe v8: in-jit lora replacement / replicated-arg bisection.

[a]  replace INSIDE jit, lora as jit arg   (== backend.val_loss, expect nan?)
[a2] replace OUTSIDE jit, lora passed as UNUSED jit arg (replicated-arg control)
[b]  [a] + value_and_grad                  (== backend.loss_and_grad)

Run: CUDA_VISIBLE_DEVICES=4,5 conda run --no-capture-output -n openvla \
        python scripts/e2e_4gpu_probe8.py
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

    # [a] replace inside jit, lora as arg (== backend.val_loss)
    def loss_a(lora_values, r, o, a):
        m = replace_lora(lora_values)
        m.train()
        return jnp.mean(m.compute_loss(r, o, a, train=True)).astype(jnp.float32)

    in_sh = (replicated, replicated, data_shard, data_shard)
    loss_a_jit = jax.jit(loss_a, in_shardings=in_sh, out_shardings=replicated)
    la = float(loss_a_jit(lora_rep, rng, obs, act))
    print(f"[a  replace-inside-arg] loss={la!r} finite={bool(np.isfinite(la))}", flush=True)

    # [a2] replace outside, lora arg unused (replicated-arg control)
    model_fixed = replace_lora(tensors)
    model_fixed.train()

    def loss_a2(_lora_values, r, o, a):
        return jnp.mean(model_fixed.compute_loss(r, o, a, train=True)).astype(jnp.float32)

    la2 = float(jax.jit(loss_a2, in_shardings=in_sh, out_shardings=replicated)(
        lora_rep, rng, obs, act))
    print(f"[a2 replace-outside  ] loss={la2!r} finite={bool(np.isfinite(la2))}", flush=True)

    # [b] [a] + value_and_grad (== backend.loss_and_grad)
    vg = jax.jit(jax.value_and_grad(loss_a, argnums=0),
                 in_shardings=in_sh, out_shardings=(replicated, replicated))
    lv, grads = vg(lora_rep, rng, obs, act)
    gmax = max(float(jnp.abs(g).max()) for g in jax.tree.leaves(grads))
    print(f"[b  vjp-arg          ] loss={float(lv)!r} finite={bool(np.isfinite(float(lv)))} "
          f"grads_finite={bool(np.isfinite(gmax))}", flush=True)


if __name__ == "__main__":
    main()
