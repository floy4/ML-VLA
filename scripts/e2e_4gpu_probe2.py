#!/usr/bin/env python3
"""Probe v2: per-device param shard finiteness + zero-LoRA loss.

Run (4gpu):  CUDA_VISIBLE_DEVICES=4,5,6,7 conda run --no-capture-output -n openvla python scripts/e2e_4gpu_probe2.py
Run (1gpu):  CUDA_VISIBLE_DEVICES=4   conda run --no-capture-output -n openvla python scripts/e2e_4gpu_probe2.py
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

    # ---- per-device shard finiteness of every state leaf ----
    bad_total = 0
    checked = 0
    for path, var in backend.state.flat_state().items():
        v = getattr(var, "value", var)
        if not hasattr(v, "addressable_shards"):
            continue
        checked += 1
        for shard in v.addressable_shards:
            arr = np.asarray(shard.data)
            if not bool(np.isfinite(arr).all()):
                bad_total += 1
                if bad_total <= 5:
                    print(f"  NON-FINITE: {'/'.join(map(str, path))} on device "
                          f"{shard.device.id}", flush=True)
    print(f"[A] state leaves checked={checked} non-finite shards={bad_total}", flush=True)

    # ---- zero-LoRA loss on one batch ----
    it = backend.domain_loader(Path(domains[d0]["dataset_root"]), cfg["task_prompt"], None, 8)
    obs, act = next(it)
    module_ab = {k: {"A": np.zeros(site_shape(backend, k, "A"), np.float32),
                     "B": np.zeros(site_shape(backend, k, "B"), np.float32)}
                 for k in backend.mapping.keys}
    tensors = {p: jnp.asarray(t) for p, t in assemble(backend.mapping, module_ab).items()}
    loss, grads = backend.loss_and_grad(dict(tensors), jax.random.PRNGKey(7), obs, act)
    print(f"[B] zero-LoRA loss={float(loss)!r}", flush=True)

    # ---- eval-mode (no dropout) forward: same values ----
    vl = float(backend.val_loss(dict(tensors), jax.random.PRNGKey(1), obs, act))
    print(f"[C] val_loss(same train-mode fn)={vl!r}", flush=True)


if __name__ == "__main__":
    main()
