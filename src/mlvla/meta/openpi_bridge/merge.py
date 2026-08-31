"""Merge canonical LoRA deltas into an OpenPI parameter tree (host, float32)."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from mlvla.meta.openpi_bridge import schema_map
from mlvla.meta.weights.lora_io import CanonicalFile


def merge_canonical(
    tree: Any,
    lora: CanonicalFile,
    *,
    keep_indices: Sequence[int] | None = None,
    compute_dtype: np.dtype | type = np.float32,
) -> tuple[Any, dict[str, Any]]:
    """Return a new parameter tree with ``delta = scale * A^T B^T`` merged in.

    The input tree is left untouched; only leaves that receive a nonzero
    delta are copied, each exactly once regardless of how many LoRA modules
    target it.  ``keep_indices`` restricts the merge to the given canonical
    module indices (used for masked-oracle controls); modules whose B matrix
    is exactly zero are skipped either way.
    """

    targets = schema_map.build_targets(lora.module_specs)
    if keep_indices is not None:
        keep = {int(i) for i in keep_indices}
        missing = keep - set(range(len(targets)))
        if missing:
            raise ValueError(f"keep_indices out of range: {sorted(missing)}")
    else:
        keep = None

    # group the modules by parameter leaf so each leaf is copied once
    groups: dict[tuple[str, ...], list[tuple[int, Any]]] = defaultdict(list)
    skipped_zero = 0
    for index, module in enumerate(lora.modules()):
        target = targets[index]
        if keep is not None and index not in keep:
            continue
        if not module.B.any():
            skipped_zero += 1
            continue
        if target.layout == "lm_head":
            raise ValueError(
                f"{module.key}: nonzero lm_head LoRA has no OpenPI target (tied embeddings)"
            )
        groups[target.leaf].append((index, module))

    merged = tree
    module_stats: dict[str, dict[str, float]] = {}
    for leaf, items in groups.items():
        node = schema_map.descend(merged, leaf)
        if isinstance(node, Mapping):
            name = schema_map._leaf_name(node)
            kernel = np.array(node[name], copy=True)
        else:
            name = None
            kernel = np.array(node, copy=True)

        for index, module in items:
            target = targets[index]
            dim_in, dim_out = schema_map.check_target(target, module.A.shape, module.B.shape)
            slice_view = _slice_view(kernel, target)
            logical = schema_map.logical_view(slice_view, target)
            if logical.shape != (dim_in, dim_out):
                raise ValueError(
                    f"{module.key}: logical kernel {logical.shape} does not match "
                    f"A/B dims ({dim_in}, {dim_out}) for stem {target.stem}"
                )

            delta = module.scale * (
                module.A.astype(compute_dtype).T @ module.B.astype(compute_dtype).T
            )
            base_norm = float(np.linalg.norm(logical.astype(compute_dtype)))
            delta_norm = float(np.linalg.norm(delta))
            new_slice = schema_map.write_logical(
                slice_view, logical.astype(compute_dtype) + delta, target
            )
            _assign_slice(kernel, target, new_slice)

            module_stats[module.key] = {
                "stem": target.stem,
                "index": target.index,
                "split": target.split,
                "delta_fro": delta_norm,
                "base_fro": base_norm,
                "relative_delta": delta_norm / base_norm if base_norm > 0 else float("inf"),
            }

        merged = _replace_leaf(merged, leaf, kernel, name)

    stats = {
        "merged_modules": sum(len(items) for items in groups.values()),
        "skipped_zero_modules": skipped_zero,
        "relative_delta_max": max((s["relative_delta"] for s in module_stats.values()), default=0.0),
        "relative_delta_mean": (
            float(np.mean([s["relative_delta"] for s in module_stats.values()])) if module_stats else 0.0
        ),
        "modules": module_stats,
    }
    return merged, stats


def _slice_view(kernel: np.ndarray, target: schema_map.ModuleTarget) -> np.ndarray:
    view = kernel
    if target.index is not None:
        view = view[int(target.index)]
    if target.split is not None:
        view = view[int(target.split)]
    return view


def _assign_slice(kernel: np.ndarray, target: schema_map.ModuleTarget, new_slice: np.ndarray) -> None:
    if target.index is not None and target.split is not None:
        kernel[target.index, target.split] = new_slice
    elif target.index is not None:
        kernel[target.index] = new_slice
    elif target.split is not None:
        kernel[target.split] = new_slice
    else:
        kernel[...] = new_slice


def _replace_leaf(tree: Any, leaf: tuple[str, ...], kernel: np.ndarray, name: str | None) -> Any:
    """Copy-on-write replacement of one parameter leaf (dicts on the path only)."""

    def rec(value: Any, path: tuple[str, ...]) -> Any:
        if not path:
            if name is None:
                return kernel
            updated = copy.copy(value)
            updated[name] = kernel
            return updated
        head, rest = path[0], path[1:]
        updated = copy.copy(value)
        updated[head] = rec(value[head], rest)
        return updated

    return rec(tree, leaf)


def tree_to_numpy(tree: Any) -> Any:
    """Convert a jax/pytree of arrays to numpy leaves (host, same dtypes)."""

    return _map_leaves(tree, lambda leaf: np.asarray(leaf))


def _map_leaves(tree: Any, fn: Any) -> Any:
    if isinstance(tree, Mapping):
        return {key: _map_leaves(value, fn) for key, value in tree.items()}
    if hasattr(tree, "dtype"):
        return fn(tree)
    return tree
