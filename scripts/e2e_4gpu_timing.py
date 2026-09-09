#!/usr/bin/env python3
"""Timing harness: where does a multi-GPU step lose time vs single-GPU?

Faithful replication of train_bridge's per-step bridge parts, plus flat-buffer
fast-path variants. 10 timed iters each (first 2 warmup discarded).

Run: CUDA_VISIBLE_DEVICES=4,5,6,7 conda run --no-capture-output -n openvla \
        python scripts/e2e_4gpu_timing.py
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml

import jax
import jax.numpy as jnp

import torch

from mlvla.meta.e2e.jax_backend import JAXBackend
from mlvla.meta.e2e.lora_mapping import assemble, disassemble


def site_shape(backend: JAXBackend, key: str, factor: str) -> tuple[int, int]:
    for site in backend.mapping.sites:
        if site.key == key and site.factor == factor:
            return tuple(site.canonical_shape)
    raise KeyError(f"{key}/{factor}")


def main() -> None:
    torch.cuda.set_device(0)
    cfg = yaml.safe_load((REPO / "configs/hypernet_e2e_film_compile_test.yaml").read_text())
    domains = yaml.safe_load((REPO / "configs/domains.yaml").read_text())
    d0 = cfg["train_domains"][0]

    backend = JAXBackend(template_npz=Path(domains[d0]["canonical_npz"]), batch_size=8)
    print(f"devices: {jax.device_count()} mesh: {backend.mesh.devices.shape}", flush=True)

    d0root = Path(domains[d0]["dataset_root"])
    it = backend.domain_loader(d0root, cfg["task_prompt"], None, 8)

    rng = np.random.default_rng(0)
    keys = list(backend.mapping.keys)
    shapes = {k: {"A": site_shape(backend, k, "A"), "B": site_shape(backend, k, "B")}
              for k in keys}

    sizes_a = [int(np.prod(shapes[k]["A"])) for k in keys]
    sizes_b = [int(np.prod(shapes[k]["B"])) for k in keys]

    def mk_ab():
        flat_a = rng.normal(scale=1e-2, size=sum(sizes_a)).astype(np.float32)
        flat_b = rng.normal(scale=1e-2, size=sum(sizes_b)).astype(np.float32)
        vals, ia, ib = {}, 0, 0
        for i, k in enumerate(keys):
            na, nb = sizes_a[i], sizes_b[i]
            vals[k] = {"A": flat_a[ia:ia + na].reshape(shapes[k]["A"]),
                       "B": flat_b[ib:ib + nb].reshape(shapes[k]["B"])}
            ia += na
            ib += nb
        return vals

    paths = tuple(backend.lora_paths)
    path_sizes = backend.lora_flat_sizes
    path_shapes = backend.lora_flat_shapes

    parts: dict[str, list[float]] = {}
    n_iter = 10

    for step in range(n_iter + 2):
        warm = step < 2
        obs, act = next(it)

        a_phys = [torch.tensor(rng.normal(scale=1.0, size=shapes[k]["A"]),
                               dtype=torch.float32, device="cuda:0") for k in keys]
        b_phys = [torch.tensor(rng.normal(scale=1.0, size=shapes[k]["B"]),
                               dtype=torch.float32, device="cuda:0") for k in keys]

        # current bridge path
        t0 = time.time()
        module_ab = mk_ab()
        tensors = {p: jnp.asarray(t) for p, t in assemble(backend.mapping, module_ab).items()}
        if not warm:
            parts["1-asm-jnp916"].append(time.time() - t0)

        t0 = time.time()
        key = jax.random.fold_in(jax.random.PRNGKey(0), step)
        loss, grads = backend.loss_and_grad(dict(tensors), key, obs, act)
        lf = float(loss)
        if not warm:
            parts["2-lossgrad(packed)"].append(time.time() - t0)

        t0 = time.time()
        gmod = disassemble(backend.mapping, {p: np.asarray(g) for p, g in grads.items()})
        flat_ga = np.concatenate([gmod[k]["A"].ravel() for k in keys])
        flat_gb = np.concatenate([gmod[k]["B"].ravel() for k in keys])
        ga = flat_ga  # (torch upload measured separately below)
        if not warm:
            parts["3-grad-d2h-916+disasm"].append(time.time() - t0)

        t0 = time.time()
        ta = torch.from_numpy(flat_ga).to("cuda:0")
        tb = torch.from_numpy(flat_gb).to("cuda:0")
        if not warm:
            parts["4-grad-h2d-torch"].append(time.time() - t0)

        # fast path: flat buffer in, flat grads out
        t0 = time.time()
        asm = assemble(backend.mapping, module_ab)
        flat_np = np.concatenate([np.asarray(asm[p]).reshape(-1) for p in paths])
        if not warm:
            parts["5-asm-flat-np"].append(time.time() - t0)

        t0 = time.time()
        flat_j = jax.device_put(jnp.asarray(flat_np))
        loss2, gflat = backend.grad_packed(flat_j, key, obs, act)
        lf2 = float(loss2)
        if not warm:
            parts["6-grad_packed-direct"].append(time.time() - t0)
        if abs(lf2 - lf) >= 1e-6:
            print(f"[warn] loss mismatch lf={lf:.6f} lf2={lf2:.6f}", flush=True)

        # debug: 5 identical calls -> deterministic? how slow (recompile?)
        flat_direct = np.concatenate([np.asarray(tensors[p]).reshape(-1) for p in paths])
        d_flat = float(np.abs(flat_direct - flat_np).max())
        if step == 2:
            ls = []
            for r in range(5):
                t0 = time.time()
                lr, _ = backend.grad_packed(flat_j, key, obs, act)
                ls.append((round(time.time() - t0, 2), f"{float(lr):.6f}"))
            print(f"[dbg] flat_maxdiff={d_flat:.2e} repeats={ls}", flush=True)
        if abs(lf2 - lf) >= 1e-6:
            print(f"[warn] loss mismatch lf={lf:.6f} lf2={lf2:.6f}", flush=True)

        t0 = time.time()
        gflat_np = np.asarray(gflat)
        goff = 0
        gdict = {}
        for p, s, sh in zip(paths, path_sizes, path_shapes):
            gdict[p] = gflat_np[goff:goff + s].reshape(sh)
            goff += s
        gmod2 = disassemble(backend.mapping, gdict)
        flat_ga2 = np.concatenate([gmod2[k]["A"].ravel() for k in keys])
        if not warm:
            parts["7-grad-flat-d2h+disasm"].append(time.time() - t0)

        assert np.allclose(flat_ga2, flat_ga, rtol=1e-5, atol=1e-8)

    print("\n=== medians (s) over", n_iter, "iters ===")
    for k in sorted(parts):
        v = sorted(parts[k])
        print(f"{k:28s} med={v[len(v)//2]:.3f}  min={v[0]:.3f}  max={v[-1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
