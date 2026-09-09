#!/usr/bin/env python3
"""4-GPU non-finite loss isolation probe.

Layer 1: obs/act leaves finite?
Layer 2: zero-LoRA loss (pure frozen base) finite?
Layer 3: random LoRA (sigma=1e-2, smoke protocol) loss/grads finite?

Run: CUDA_VISIBLE_DEVICES=4,5,6,7 conda run --no-capture-output -n openvla \
        python scripts/e2e_4gpu_probe.py
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
    print(f"mesh: {backend.mesh.devices.shape}", flush=True)
    it = backend.domain_loader(Path(domains[d0]["dataset_root"]), cfg["task_prompt"], None, 8)
    obs, act = next(it)

    ok = all(bool(np.isfinite(np.asarray(x)).all()) for x in jax.tree.leaves((obs, act)))
    print(f"[1] obs/act leaves all finite: {ok}", flush=True)
    for name, x in jax.tree.flatten_with_path((obs, act))[0][:0] or []:
        pass
    bad = [str(p) for p, x in jax.tree_util.tree_flatten_with_path((obs, act))[0]
           if not bool(np.isfinite(np.asarray(x)).all())] if hasattr(jax.tree_util, "tree_flatten_with_path") else []
    # simple per-leaf report
    leaves = []
    jax.tree.map(lambda x: leaves.append(x), (obs, act))
    print(f"    leaves: {len(leaves)}, dtypes: {sorted({str(np.asarray(x).dtype) for x in leaves})}", flush=True)

    rng = np.random.default_rng(0)
    for tag, module_ab in (
        ("zero", {k: {"A": np.zeros(site_shape(backend, k, "A"), np.float32),
                      "B": np.zeros(site_shape(backend, k, "B"), np.float32)}
                  for k in backend.mapping.keys}),
        ("rand1e-2", {k: {"A": rng.normal(scale=1e-2, size=site_shape(backend, k, "A")).astype(np.float32),
                          "B": rng.normal(scale=1e-2, size=site_shape(backend, k, "B")).astype(np.float32)}
                      for k in backend.mapping.keys}),
    ):
        tensors = {p: jnp.asarray(t) for p, t in assemble(backend.mapping, module_ab).items()}
        key = jax.random.PRNGKey(7)
        loss, grads = backend.loss_and_grad(dict(tensors), key, obs, act)
        loss = float(loss)
        gvals = jax.tree.leaves(grads)
        gfinite = all(bool(np.isfinite(np.asarray(g)).all()) for g in gvals)
        gmax = max(float(np.abs(np.asarray(g)).max()) for g in gvals) if gvals else 0.0
        print(f"[2:{tag}] loss={loss!r} finite={np.isfinite(loss)} "
              f"grads_finite={gfinite} max|g|={gmax:.3e}", flush=True)
        vl = float(backend.val_loss(dict(tensors), jax.random.PRNGKey(1), obs, act))
        print(f"[2:{tag}:val] loss={vl!r} finite={np.isfinite(vl)}", flush=True)


if __name__ == "__main__":
    main()
