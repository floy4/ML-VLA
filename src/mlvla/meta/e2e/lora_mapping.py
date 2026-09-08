# src/mlvla/meta/e2e/lora_mapping.py
"""Canonical 458-module LoRA <-> JAX nnx lora tensors: build/assemble/disassemble.

Semantics mirror mlvla.adapters.table8_converter exactly: canonical
A [r,in] / B [out,r] are transposed then reshaped into their JAX slot (jax a is
[in...,r] possibly stacked [...,in,r]; jax b is [r,...,out]); scanned tensors
are shared across logical layers and written per (index, split).

``disassemble`` inverts the injection with the same flattenings as the
exporter's ``table8_converter._pair`` (A slots flatten their leading axes and
keep the trailing rank axis; B slots keep the leading rank axis and flatten the
rest).  A plain ``grad.T`` reverses *every* axis and would scramble the N-D
head-grouped slots of the real model (q/kv ``b`` is [r, heads, head_dim],
attn_vec ``a`` is [heads, head_dim, r], vision attention ``b`` likewise); for
2-D slots both flattenings coincide with ``grad.T.reshape(canonical_shape)``.

All dictionary keys are JAX flat-path *tuples* (assemble output, disassemble
input, ``LoRAMapping.tensors``); ``lora_tensor_shapes`` given to
``build_mapping`` may additionally use "/"-joined string keys, which are
normalized on entry.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from mlvla.meta.weights.lora_io import load_canonical


@dataclass(frozen=True)
class ModuleSite:
    key: str                        # canonical key
    factor: str                     # "A" | "B"
    path: tuple[str, ...]           # JAX nnx flat 路径(元组)
    index: int | None               # 层 index(None = 无层维)
    split: int | None               # kv/gating einsum 的 split 维
    canonical_shape: tuple[int, int]  # A: (r,in); B: (out,r)


@dataclass(frozen=True)
class LoRAMapping:
    keys: tuple[str, ...]                      # 458 个 canonical key
    sites: tuple[ModuleSite, ...]              # 916 个写入位
    tensors: dict[tuple[str, ...], tuple[int, ...]]  # jax path -> 完整形状


def _as_path(path: tuple[str, ...] | str) -> tuple[str, ...]:
    return tuple(path.split("/")) if isinstance(path, str) else tuple(path)


def _factor_paths(string_paths: set[str], stem: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    nested, suffixed = f"{stem}/lora_a", f"{stem}_lora_a"
    a_key = nested if nested in string_paths else suffixed
    b_key = a_key.replace("lora_a", "lora_b")
    if a_key not in string_paths or b_key not in string_paths:
        raise KeyError(f"No lora_a/lora_b for stem {stem!r}")
    return tuple(a_key.split("/")), tuple(b_key.split("/"))


def _slot_shape(shape: tuple[int, ...], index: int | None, split: int | None) -> tuple[int, ...]:
    """Shape of the sub-array a module writes into (index/split each drop their axis)."""
    if index is not None:
        shape = shape[1:]
    if split is not None:
        shape = shape[1:]
    return shape


def _validate_slot(module_key: str, factor: str, slot: tuple[int, ...],
                   canonical: tuple[int, ...]) -> None:
    """A transposed canonical factor must reshape exactly into its slot.

    A [r,in] -> A.T [in,r] needs the slot to end with the rank axis; B [out,r]
    -> B.T [r,out] needs the slot to start with it; sizes must match either
    way.  2-D slots therefore have to equal the transposed shape exactly.
    """
    rank = canonical[0] if factor == "A" else canonical[1]
    keeps_rank = bool(slot) and (slot[-1] if factor == "A" else slot[0]) == rank
    if not keeps_rank or int(np.prod(slot)) != int(np.prod(canonical)):
        raise ValueError(
            f"{module_key}: {factor}{tuple(canonical)} transposed does not fit jax slot {slot}")


def build_mapping(template_npz: Path,
                  lora_tensor_shapes: Mapping[tuple[str, ...] | str, tuple[int, ...]]) -> LoRAMapping:
    shapes = {_as_path(path): tuple(shape) for path, shape in lora_tensor_shapes.items()}
    string_paths = {"/".join(path) for path in shapes}
    keys, sites = [], []
    with load_canonical(template_npz) as template:
        # lora_io's CanonicalModule carries key/A/B only; metadata (jax_stem,
        # index, split) lives in the manifest entries exposed by module_specs.
        for module, spec in zip(template.modules(), template.module_specs):
            meta = spec.get("metadata") or {}
            if "jax_stem" not in meta:
                raise ValueError(f"{module.key}: missing jax_stem metadata")
            index, split = meta.get("index"), meta.get("split")
            a_path, b_path = _factor_paths(string_paths, str(meta["jax_stem"]))
            _validate_slot(module.key, "A", _slot_shape(shapes[a_path], index, split),
                           tuple(module.A.shape))
            _validate_slot(module.key, "B", _slot_shape(shapes[b_path], index, split),
                           tuple(module.B.shape))
            keys.append(module.key)
            sites.append(ModuleSite(module.key, "A", a_path, index, split, tuple(module.A.shape)))
            sites.append(ModuleSite(module.key, "B", b_path, index, split, tuple(module.B.shape)))
    return LoRAMapping(tuple(sorted(set(keys))), tuple(sites), shapes)


def _slot(tensor: np.ndarray, site: ModuleSite) -> np.ndarray:
    view = tensor
    if site.index is not None:
        view = view[site.index]
    if site.split is not None:
        view = view[site.split]
    return view


def assemble(mapping: LoRAMapping,
             module_ab: Mapping[str, Mapping[str, np.ndarray]]) -> dict[tuple[str, ...], np.ndarray]:
    """Zero-initialize every lora tensor, then write each canonical factor into its slot."""
    out = {path: np.zeros(shape, dtype=np.float32) for path, shape in mapping.tensors.items()}
    for site in mapping.sites:
        value = np.asarray(module_ab[site.key][site.factor], dtype=np.float32)
        target = _slot(out[site.path], site)
        target[...] = value.T.reshape(target.shape)
    return out


def disassemble(mapping: LoRAMapping,
                tensor_grads: Mapping[tuple[str, ...], np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Inverse of :func:`assemble`: slot gradient -> transposed canonical layout."""
    out: dict[str, dict[str, np.ndarray]] = {key: {} for key in mapping.keys}
    for site in mapping.sites:
        grad = np.asarray(_slot(tensor_grads[site.path], site), dtype=np.float32)
        if site.factor == "A":
            canonical = grad.reshape(-1, grad.shape[-1]).T
        else:
            canonical = grad.reshape(grad.shape[0], -1).T
        out[site.key][site.factor] = canonical.reshape(site.canonical_shape)
    return out
