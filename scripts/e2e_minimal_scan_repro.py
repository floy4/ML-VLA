#!/usr/bin/env python3
"""Minimal multi-GPU nan repro: SigLIP-style scan + fused attention, bf16.

Mirrors the failing path in mlvla/jax/siglip_lora.py: flax
nn.dot_product_attention (no mask, bf16) inside nn.scan, batch-sharded
across devices (P("B")). So400m/14 shapes: 16 heads, head_dim 72, 256
patches.

Run: CUDA_VISIBLE_DEVICES=4,5 conda run --no-capture-output -n openvla \
        python scripts/e2e_minimal_scan_repro.py
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def main() -> None:
    mesh = Mesh(jax.devices(), ("B",))
    sh = NamedSharding(mesh, P("B"))
    print(f"devices: {jax.device_count()}", flush=True)

    B, H, T, D = 8, 16, 256, 72
    rng = jax.random.PRNGKey(0)

    def mk(dtype):
        qs, ks, vs = jax.random.split(rng, 3)
        q = jax.random.normal(qs, (B, H, T, D), dtype=dtype) * 0.05
        k = jax.random.normal(ks, (B, H, T, D), dtype=dtype) * 0.05
        v = jax.random.normal(vs, (B, H, T, D), dtype=dtype) * 0.05
        return jax.device_put(q, sh), jax.device_put(k, sh), jax.device_put(v, sh)

    def attn(q, k, v):
        return jax.nn.dot_product_attention(q, k, v)

    def scan_attn(q, k, v):
        def body(carry, _):
            q, k, v = carry
            o = attn(q, k, v)
            return (o, k, v), None
        (o, _, _), _ = jax.lax.scan(body, (q, k, v), None, length=12)
        return o

    for dtype, tag in ((jnp.bfloat16, "bf16"), (jnp.float32, "f32")):
        q, k, v = mk(dtype)
        out1 = jax.jit(attn)(q, k, v)
        print(f"[{tag}] jit attn finite: {bool(jnp.isfinite(out1).all())}", flush=True)
        out2 = jax.jit(scan_attn)(q, k, v)
        print(f"[{tag}] scan(attn x12) finite: {bool(jnp.isfinite(out2).all())}", flush=True)


if __name__ == "__main__":
    main()
