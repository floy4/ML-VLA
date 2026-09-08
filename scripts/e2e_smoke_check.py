#!/usr/bin/env python3
"""Gradient parity: bridge (jit value_and_grad on lora pytree) vs reference
(nnx.value_and_grad with DiffState on the same model & values & batch).

Run: CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n openvla \
        python scripts/e2e_smoke_check.py
Accept: eager-vs-bridge < 1e-2; loss |bridge-ref| < 1e-2 (same order of
magnitude); global max|dgrad|/max|ref| < 5e-2; per-tensor cosine > 0.99 on
nonzero tensors; 52/52 tensors matched. Thresholds reflect the bf16 floor of
two independently compiled gradient paths (see reports/2026-09-08-e2e-meta-
network-smoke.md).
"""
from __future__ import annotations

import os

# torch is not used here but XLA preallocation off keeps behaviour identical
# to the training process that this check validates.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import jax
import jax.numpy as jnp
from flax import nnx

from mlvla.meta.e2e.jax_backend import JAXBackend
from mlvla.meta.e2e.lora_mapping import assemble

TEMPLATE = Path("/data2/zhy/models/wizard/experts_perturbed_lagfix/"
                "l1_warm_dim/expert_de2a6ce7/phase1/final/params.canonical.npz")

DATASET_ROOT = Path("/data2/zhy/wizard_perturbed_lagfix_clean/l1_warm_dim")
TASK = "open the middle drawer of the cabinet"


def site_shape(backend: JAXBackend, key: str, factor: str) -> tuple[int, int]:
    """canonical (r,in)/(out,r) for a module factor, read from the mapping."""
    for site in backend.mapping.sites:
        if site.key == key and site.factor == factor:
            return tuple(site.canonical_shape)
    raise KeyError(f"{key}/{factor}")


def main() -> None:
    backend = JAXBackend(template_npz=TEMPLATE, batch_size=8)
    rng = np.random.default_rng(0)
    module_ab = {}
    for key in backend.mapping.keys:
        # Ruling: read r/in/out from each site's canonical_shape (not fixed 16).
        a_shape = site_shape(backend, key, "A")   # (r, in)
        b_shape = site_shape(backend, key, "B")   # (out, r)
        module_ab[key] = {
            "A": rng.normal(scale=1e-2, size=a_shape).astype(np.float32),
            "B": rng.normal(scale=1e-2, size=b_shape).astype(np.float32),
        }
    tensors = {p: jnp.asarray(t) for p, t in assemble(backend.mapping, module_ab).items()}

    it = backend.domain_loader(DATASET_ROOT, TASK, None, 8)
    obs, act = next(it)
    key = jax.random.PRNGKey(7)
    loss_b, grads_b = backend.loss_and_grad(dict(tensors), key, obs, act)

    # reference: same values, same model, same batch, expert-training-style
    # differentiation (nnx.value_and_grad + DiffState(0, All(Param,
    # PathRegex(".*lora.*")))) — the exact call used inside the jitted
    # train_step of expert training. To compare differentiation paths rather
    # than XLA compilation modes, the reference rebuilds the model exactly
    # like the bridge does (base state as closed-over constants, lora values
    # as traced args, merge inside jit): with frozen base params in bf16,
    # eager-vs-jit or constant-vs-argument layouts round differently
    # (~1e-4 loss, ~2e-2 grads — measured, see report), which would mask the
    # plumbing signal the parity check exists for.
    flat_const = backend.state.flat_state()
    lora_dtypes = {str(var.value.dtype) for path, var in flat_const.items()
                   if tuple(map(str, path)) in tensors}
    base_dtypes = {str(var.value.dtype) for path, var in flat_const.items()
                   if tuple(map(str, path)) not in tensors
                   and hasattr(var.value, "dtype")}
    print(f"param dtypes: lora={sorted(lora_dtypes)} base(non-lora)={sorted(base_dtypes)}")

    def ref_loss_fn(m, rng_, obs_, act_):
        m.train()
        # identical to the bridge loss (including the fp32 cast) so the two
        # numbers differ only by the differentiation path, not the formula.
        return jnp.mean(m.compute_loss(rng_, obs_, act_, train=True)).astype(jnp.float32)

    def ref_replace_lora(lora_values):
        items = []
        for path, var in flat_const.items():
            k = tuple(map(str, path))
            if k in lora_values:
                items.append((path, var.replace(
                    value=jnp.asarray(lora_values[k]).astype(var.value.dtype))))
            else:
                items.append((path, var))
        return nnx.merge(backend.graphdef, nnx.State.from_flat_path(items))

    import openpi.shared.nnx_utils as nnx_utils
    diff = nnx.DiffState(0, nnx.All(nnx.Param, nnx_utils.PathRegex(".*lora.*")))

    def ref_step(lora_values):
        m = ref_replace_lora(lora_values)
        return nnx.value_and_grad(ref_loss_fn, argnums=diff)(m, key, obs, act)

    loss_r, grads_r = jax.jit(ref_step)(dict(tensors))

    # Strong plumbing check: an eager forward on an independently rebuilt model
    # (same assembled values, no bridge internals) must match the bridge jit
    # loss to O(bf16 floor); transpose/keying bugs shift the loss at O(1).
    # The floor is batch-dependent (measured 8.8e-7 .. 1.1e-4 on loss ~0.124).
    eager_model = ref_replace_lora({p: np.asarray(t) for p, t in tensors.items()})
    loss_eager = float(ref_loss_fn(eager_model, key, obs, act))
    print(f"eager-forward loss={loss_eager:.6f} vs jit bridge={float(loss_b):.6f} "
          f"|d|={abs(loss_eager - float(loss_b)):.2e}")
    assert abs(loss_eager - float(loss_b)) < 1e-2, "eager/bridge value plumbing mismatch"
    print(f"loss bridge={float(loss_b):.6f} reference={float(loss_r):.6f} "
          f"|d|={abs(float(loss_b)-float(loss_r)):.2e}")

    # Two independently compiled bf16 gradient paths (bridge jit vs DiffState
    # jit) round differently; the spec acceptance is "same order of magnitude".
    # Alignment (cosine) is the plumbing-sensitive metric: mis-wired slots
    # destroy alignment (cos ~ 0) while bf16 rounding keeps cos ~ 1 even when
    # per-tensor rel is ~10% on small-magnitude tensors (e.g. vision tower).
    num = den = 0.0
    matched = total = 0
    rows: list[tuple[float, float, float, float, str]] = []
    ref_flat = {"/".join(map(str, p)): np.asarray(v.value if hasattr(v, "value") else v)
                for p, v in grads_r.flat_state().items()}
    for p, gb in grads_b.items():
        total += 1
        gr = ref_flat.get("/".join(p))
        if gr is None or gr.size == 0:
            continue
        matched += 1
        gb_f = np.asarray(gb, dtype=np.float64).ravel()
        gr_f = gr.astype(np.float64).ravel()
        t_num = float(np.abs(gb_f - gr_f).max())
        t_den = float(np.abs(gr_f).max())
        num = max(num, t_num)
        den = max(den, t_den)
        if t_num == 0.0 and t_den == 0.0:
            continue  # both zero (e.g. lm_head lora unused by the loss): agreement
        cos = float(np.dot(gb_f, gr_f) /
                    (np.linalg.norm(gb_f) * np.linalg.norm(gr_f) + 1e-30))
        rows.append((cos, t_num / t_den if t_den > 0 else 0.0, t_num, t_den, "/".join(p)))
    rows.sort()  # ascending cos: worst alignment first
    print(f"grad tensors: {matched}/{total} matched to reference lora params")
    print(f"max|Dgrad|={num:.3e}  max|ref|={den:.3e}  rel={num/max(den,1e-12):.3e}")
    print("worst per-tensor alignment (top 5):  cos  rel  max|D|  max|ref|")
    for cos, rel, t_num, t_den, name in rows[:5]:
        print(f"  cos={cos:.6f} rel={rel:.2e} |D|={t_num:.2e} |ref|={t_den:.2e}  {name}")
    assert abs(float(loss_b) - float(loss_r)) < 1e-2, "loss mismatch (order of magnitude)"
    assert num / max(den, 1e-12) < 5e-2, "gradient parity failed (order of magnitude)"
    assert all(cos > 0.99 for cos, *_ in rows), "per-tensor gradient misalignment (plumbing bug)"
    assert matched == total, "some bridge grad tensors missing from reference"
    print("PARITY OK")


if __name__ == "__main__":
    main()
