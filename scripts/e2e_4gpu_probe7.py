#!/usr/bin/env python3
"""Probe v7: isolate which backend._loss ingredient causes the sharded nan.

[1] plain merged model, eval mode            (probe6: finite)
[2] train() mode, no lora replacement
[3] lora replacement (zero), eval mode
[4] replacement + train()  == backend._loss  (probe2: nan)

Run: CUDA_VISIBLE_DEVICES=4,5 conda run --no-capture-output -n openvla \
        python scripts/e2e_4gpu_probe7.py
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

    def run(tag, model, train_mode):
        if train_mode:
            model.train()
        fn = jax.jit(lambda r, o, a: jnp.mean(model.compute_loss(r, o, a, train=True)),
                     in_shardings=(replicated, data_shard, data_shard),
                     out_shardings=replicated)
        loss = float(fn(rng, obs, act))
        print(f"[{tag}] loss={loss!r} finite={bool(np.isfinite(loss))}", flush=True)

    run("1 plain-eval", nnx.merge(backend.graphdef, backend.state), False)
    run("2 train-only", nnx.merge(backend.graphdef, backend.state), True)
    run("3 replace-eval", replace_lora(tensors), False)
    run("4 replace+train", replace_lora(tensors), True)


if __name__ == "__main__":
    main()
