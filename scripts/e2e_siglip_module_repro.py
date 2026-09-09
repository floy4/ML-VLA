#!/usr/bin/env python3
"""Probe v4: ML-VLA SigLIP LoRA module alone, sharded vs replicated batch.

Random init (enabled=False -> pure base path). If the sharded case nans,
the bug is inside this module's composition (conv stem / posemb / scan
blocks / flax fused attention) and can be bisected op-by-op without the
3B model.

Env switches:
  MLVLA_SIGLIP_SCAN=0   -> scan=False (unrolled blocks)
  MLVLA_SIGLIP_F32=1    -> dtype_mm float32

Run: CUDA_VISIBLE_DEVICES=4,5 conda run --no-capture-output -n openvla \
        python scripts/e2e_siglip_module_repro.py
"""
from __future__ import annotations

import os

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from mlvla.jax import siglip_lora


def main() -> None:
    mesh = jax.make_mesh((jax.device_count(), 1), ("batch", "fsdp"))
    data_shard = NamedSharding(mesh, P(("batch", "fsdp")))
    replicated = NamedSharding(mesh, P())
    print(f"devices: {jax.device_count()}", flush=True)

    dtype = jnp.float32 if os.environ.get("MLVLA_SIGLIP_F32") == "1" else jnp.bfloat16
    scan = os.environ.get("MLVLA_SIGLIP_SCAN", "1") == "1"
    print(f"dtype={dtype} scan={scan}", flush=True)

    model = siglip_lora.Module(
        num_classes=1152, variant="So400m/14", pool_type="none",
        scan=scan, dtype_mm=dtype, enabled=False,
    )

    rng = jax.random.PRNGKey(0)
    img = jax.random.normal(rng, (8, 224, 224, 3), dtype=jnp.float32) * 3.0
    params = model.init(jax.random.PRNGKey(1), img, train=False)

    def fwd(p, x):
        return model.apply(p, x, train=False)[1]["pre_logits_2d"]

    out_rep = jax.jit(fwd, in_shardings=(None, replicated))(params, img)
    fin_rep = bool(jnp.isfinite(out_rep).all())
    print(f"[replicated] finite={fin_rep} max|out|={float(jnp.abs(out_rep).max()):.3e}", flush=True)

    x_sh = jax.device_put(img, data_shard)
    out_sh = jax.jit(fwd, in_shardings=(None, data_shard))(params, x_sh)
    fin_sh = bool(jnp.isfinite(out_sh).all())
    print(f"[sharded]    finite={fin_sh} max|out|={float(jnp.abs(out_sh).max()):.3e}", flush=True)
    print(f"[match] same_values={bool(jnp.allclose(out_rep, out_sh, rtol=1e-2, atol=1e-2))}",
          flush=True)


if __name__ == "__main__":
    main()
