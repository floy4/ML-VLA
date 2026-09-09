#!/usr/bin/env python3
"""Probe v10: workaround candidate — single packed lora arg + in-jit merge.

916 small lora arrays concatenated into ONE replicated jit arg, split back
inside the jit, then merged into the model state (the nan-triggering path).
If finite, this is the multi-GPU workaround for the backend.

Run: CUDA_VISIBLE_DEVICES=4,5 conda run --no-capture-output -n openvla \
        python scripts/e2e_4gpu_probe10.py
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

    rng_np = np.random.default_rng(0)
    use_rand = os.environ.get("MLVLA_PROBE10_RAND", "0") == "1"
    if use_rand:
        module_ab = {k: {"A": rng_np.normal(scale=1e-2, size=site_shape(backend, k, "A")).astype(np.float32),
                         "B": rng_np.normal(scale=1e-2, size=site_shape(backend, k, "B")).astype(np.float32)}
                     for k in backend.mapping.keys}
    else:
        module_ab = {k: {"A": np.zeros(site_shape(backend, k, "A"), np.float32),
                         "B": np.zeros(site_shape(backend, k, "B"), np.float32)}
                     for k in backend.mapping.keys}
    tensors = dict(assemble(backend.mapping, module_ab))

    paths = tuple(backend.lora_paths)
    sizes = [int(np.prod(tensors[p].shape)) for p in paths]

    flat_np = np.concatenate([np.asarray(tensors[p]).reshape(-1) for p in paths])
    flat = jax.device_put(jnp.asarray(flat_np), replicated)

    def unpack(flat_arr):
        out, off = {}, 0
        for p, s in zip(paths, sizes):
            out[p] = jnp.reshape(flat_arr[off:off + s], tensors[p].shape)
            off += s
        return out

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

    def loss_packed(flat_lora, r, o, a):
        m = replace_lora(unpack(flat_lora))
        m.train()
        return jnp.mean(m.compute_loss(r, o, a, train=True)).astype(jnp.float32)

    vg = jax.jit(jax.value_and_grad(loss_packed, argnums=0),
                 in_shardings=(replicated, replicated, data_shard, data_shard),
                 out_shardings=(replicated, replicated))
    lv, grads = vg(flat, rng, obs, act)
    gmax = float(jnp.abs(grads).max())
    print(f"[packed-arg vjp] loss={float(lv)!r} finite={bool(np.isfinite(float(lv)))} "
          f"grads_finite={bool(np.isfinite(gmax))} max|g|={gmax:.3e}", flush=True)


if __name__ == "__main__":
    main()
