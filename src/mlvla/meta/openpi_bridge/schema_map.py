"""Map canonical LoRA modules onto OpenPI pi05_base parameter-tree leaves.

The canonical metadata stores a ``jax_stem`` (plus optional ``index`` and
``split``) that was recorded when the WIZARD Orbax LoRA tree was exported.
OpenPI stores the same weights under an isomorphic tree, so a stem resolves
to a leaf by descending the stem path and then picking the array (or the
``w``/``kernel`` entry of the node found).

OpenPI consumes the einsum weights in big_vision layout, e.g. the language
model attention uses ``q_einsum/w[i]`` with shape ``[heads, in, head_dim]``
and the equation ``BTD,NDH->BTNH``.  A LoRA pair that factorizes the logical
``[in, heads*head_dim]`` matrix must therefore be transposed before it can be
added to the stored kernel.  The layout rules below encode exactly those
per-stem flattenings, and every rule is validated against the A/B shapes of
the module before any merge happens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ModuleTarget:
    """Where and how a canonical module merges into the parameter tree."""

    module_key: str
    stem: str
    leaf: tuple[str, ...]
    index: int | None
    split: int | None
    layout: str
    lm_head: bool = False


def _leaf_name(node: Mapping[str, Any]) -> str:
    for name in ("w", "kernel"):
        if name in node:
            return name
    raise KeyError(f"node {dict(node).keys()} has neither 'w' nor 'kernel'")


def _layout_for(stem: str) -> str:
    if stem in ("gemma_expert_lm_head", "paligemma_lm_head"):
        return "lm_head"
    if stem.endswith(("q_einsum", "q_einsum_1")):
        return "heads_in_hd"  # [H, in, hd] -> logical [in, H*hd]
    if stem.endswith(("kv_einsum", "kv_einsum_1")):
        return "one_in_hd"  # [1, in, hd] -> [in, hd]
    if stem.endswith(("attn_vec_einsum", "attn_vec_einsum_1")):
        return "heads_hd_out"  # [H, hd, out] -> [H*hd, out], row-major
    if stem.endswith(("MultiHeadDotProductAttention_0/query", "MultiHeadDotProductAttention_0/key", "MultiHeadDotProductAttention_0/value")):
        return "in_heads_hd"  # vision kernels are stored [in, H, hd] -> [in, H*hd]
    if stem.endswith("MultiHeadDotProductAttention_0/out"):
        return "heads_hd_out"  # [H, hd, out] -> [H*hd, out]
    return "direct"  # 2-D [in, out] kernels


def build_targets(module_specs: Sequence[Mapping[str, Any]]) -> list[ModuleTarget]:
    """Build one :class:`ModuleTarget` per canonical module spec."""

    targets = []
    for spec in module_specs:
        metadata = spec.get("metadata", {})
        stem = metadata.get("jax_stem")
        if not stem:
            raise ValueError(f"module {spec.get('key')} has no jax_stem")
        targets.append(
            ModuleTarget(
                module_key=str(spec["key"]),
                stem=str(stem),
                leaf=tuple(stem.split("/")),
                index=metadata.get("index"),
                split=metadata.get("split"),
                layout=_layout_for(str(stem)),
            )
        )
    return targets


def descend(tree: Any, path: Sequence[str]) -> Any:
    value = tree
    for part in path:
        value = value[part]
    return value


def resolve_kernel(tree: Any, target: ModuleTarget) -> np.ndarray:
    """Return the stored kernel slice addressed by *target*."""

    node = descend(tree, target.leaf)
    kernel = node[_leaf_name(node)] if isinstance(node, Mapping) else node
    kernel = np.asarray(kernel)
    if target.index is not None:
        kernel = kernel[int(target.index)]
    if target.split is not None:
        kernel = kernel[int(target.split)]
    return kernel


def logical_view(kernel: np.ndarray, target: ModuleTarget) -> np.ndarray:
    """Flatten a sliced kernel into its logical ``[in, out]`` matrix."""

    if target.layout == "heads_in_hd":
        heads, dim_in, head_dim = kernel.shape
        return np.ascontiguousarray(kernel.transpose(1, 0, 2)).reshape(dim_in, heads * head_dim)
    if target.layout == "one_in_hd":
        dim_in, head_dim = kernel.shape[-2:]
        return kernel.reshape(dim_in, head_dim)
    if target.layout == "in_heads_hd":  # [in, H, hd] -> [in, H*hd]
        return kernel.reshape(kernel.shape[0], kernel.shape[1] * kernel.shape[2])
    if target.layout == "heads_hd_out":  # [H, hd, out] -> [H*hd, out]
        return kernel.reshape(kernel.shape[0] * kernel.shape[1], kernel.shape[2])
    if target.layout == "direct":
        if kernel.ndim != 2:
            raise ValueError(f"{target.module_key}: expected 2-D kernel, got {kernel.shape}")
        return kernel
    raise ValueError(f"{target.module_key}: lm_head modules have no OpenPI target")


def write_logical(kernel: np.ndarray, logical: np.ndarray, target: ModuleTarget) -> np.ndarray:
    """Inverse of :func:`logical_view`: store a logical matrix back in kernel layout."""

    if kernel.ndim == 2:
        return logical.astype(kernel.dtype)
    if target.layout == "heads_in_hd":
        heads, dim_in, head_dim = kernel.shape
        stacked = logical.reshape(dim_in, heads, head_dim).transpose(1, 0, 2)
        return np.ascontiguousarray(stacked).astype(kernel.dtype)
    if target.layout in ("one_in_hd", "in_heads_hd", "heads_hd_out"):
        return logical.reshape(kernel.shape).astype(kernel.dtype)
    raise ValueError(f"{target.module_key}: unsupported layout {target.layout}")


def check_target(target: ModuleTarget, a_shape: tuple[int, ...], b_shape: tuple[int, ...]) -> tuple[int, int] | None:
    """Validate a target against canonical A/B shapes; returns (dim_in, dim_out) or None."""

    if target.layout == "lm_head":
        return None
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise ValueError(f"{target.module_key}: canonical A/B must be matrices")
    return int(a_shape[1]), int(b_shape[0])
