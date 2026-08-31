from __future__ import annotations

from pathlib import Path

import numpy as np

from mlvla.meta.openpi_bridge import schema_map
from mlvla.meta.openpi_bridge.merge import merge_canonical
from mlvla.meta.weights.lora_io import save_canonical


def _spec(key: str, stem: str, index: int | None = None, split: int | None = None, alpha: float = 16.0) -> dict:
    return {"key": key, "alpha": alpha, "metadata": {"jax_stem": stem, "index": index, "split": split}}


def _make_tree() -> dict:
    rng = np.random.default_rng(7)
    return {
        "PaliGemma": {
            "llm": {
                "layers": {
                    "attn": {
                        "q_einsum": {"w": rng.normal(size=(2, 2, 4, 3))},
                        "kv_einsum": {"w": rng.normal(size=(2, 2, 1, 4, 3))},
                        "attn_vec_einsum": {"w": rng.normal(size=(2, 2, 3, 4))},
                    },
                    "mlp": {
                        "gating_einsum": rng.normal(size=(2, 2, 4, 5)),
                        "linear": rng.normal(size=(2, 5, 4)),
                    },
                }
            },
            "img": {
                "Transformer": {
                    "encoderblock": {
                        "MultiHeadDotProductAttention_0": {
                            "query": {"kernel": rng.normal(size=(2, 4, 2, 3))},
                            "out": {"kernel": rng.normal(size=(2, 2, 3, 4))},
                        }
                    }
                }
            },
        },
        "action_in_proj": {"kernel": rng.normal(size=(4, 6)), "bias": np.zeros(6)},
    }


def _expected_logical(kernel: np.ndarray, layout: str) -> np.ndarray:
    if layout == "heads_in_hd":
        h, dim_in, hd = kernel.shape
        return kernel.transpose(1, 0, 2).reshape(dim_in, h * hd)
    if layout == "one_in_hd":
        return kernel.reshape(kernel.shape[-2], kernel.shape[-1])
    if layout == "in_heads_hd":
        return kernel.reshape(kernel.shape[0], kernel.shape[1] * kernel.shape[2])
    if layout == "heads_hd_out":
        return kernel.reshape(kernel.shape[0] * kernel.shape[1], kernel.shape[2])
    return kernel


CASES = [
    ("llm.q", _spec("llm.q", "PaliGemma/llm/layers/attn/q_einsum", index=1), (2, 4), (6, 2), "heads_in_hd"),
    ("llm.kv", _spec("llm.kv", "PaliGemma/llm/layers/attn/kv_einsum", index=1, split=0), (2, 4), (3, 2), "one_in_hd"),
    (
        "llm.o",
        _spec("llm.o", "PaliGemma/llm/layers/attn/attn_vec_einsum", index=0),
        (2, 6),
        (4, 2),
        "heads_hd_out",
    ),
    (
        "llm.gate",
        _spec("llm.gate", "PaliGemma/llm/layers/mlp/gating_einsum", index=1, split=1),
        (2, 4),
        (5, 2),
        "direct",
    ),
    ("llm.down", _spec("llm.down", "PaliGemma/llm/layers/mlp/linear", index=0), (2, 5), (4, 2), "direct"),
    (
        "vis.q",
        _spec(
            "vis.q",
            "PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query",
            index=1,
        ),
        (2, 4),
        (6, 2),
        "in_heads_hd",
    ),
    (
        "vis.out",
        _spec(
            "vis.out",
            "PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out",
            index=0,
        ),
        (2, 6),
        (4, 2),
        "heads_hd_out",
    ),
    ("policy", _spec("policy", "action_in_proj"), (2, 4), (6, 2), "direct"),
]


def test_logical_view_matches_expected_layouts() -> None:
    tree = _make_tree()
    for name, spec, a_shape, b_shape, layout in CASES:
        target = schema_map.build_targets([spec])[0]
        kernel = schema_map.resolve_kernel(tree, target)
        assert target.layout == layout, name
        logical = schema_map.logical_view(kernel, target)
        assert logical.shape == (a_shape[1], b_shape[0]), name
        assert np.allclose(logical, _expected_logical(kernel, layout)), name


def test_write_logical_round_trips() -> None:
    tree = _make_tree()
    rng = np.random.default_rng(3)
    for name, spec, *_ in CASES:
        target = schema_map.build_targets([spec])[0]
        kernel = schema_map.resolve_kernel(tree, target)
        logical = schema_map.logical_view(kernel, target)
        shuffled = rng.normal(size=logical.shape)
        restored = schema_map.write_logical(kernel, shuffled, target)
        assert np.allclose(schema_map.logical_view(restored, target), shuffled), name
        assert np.allclose(schema_map.write_logical(kernel, logical, target), kernel), name


def test_merge_canonical_adds_delta_and_keeps_other_leaves(tmp_path: Path) -> None:
    tree = _make_tree()
    rng = np.random.default_rng(11)
    specs = [spec for _, spec, *_ in CASES]
    arrays = {}
    for i, (_, spec, a_shape, b_shape, _) in enumerate(CASES):
        arrays[f"module_{i}_A"] = rng.normal(size=a_shape).astype(np.float32)
        arrays[f"module_{i}_B"] = rng.normal(size=b_shape).astype(np.float32)
    path = tmp_path / "params.canonical.npz"
    save_canonical(path, {"modules": specs}, arrays)

    from mlvla.meta.weights.lora_io import load_canonical

    with load_canonical(path) as lora:
        merged, stats = merge_canonical(tree, lora)
        assert stats["merged_modules"] == len(CASES)
        assert stats["skipped_zero_modules"] == 0

        targets = schema_map.build_targets(lora.module_specs)
        for i, (name, spec, *_rest) in enumerate(CASES):
            target = targets[i]
            module = lora.module(i)
            before = schema_map.logical_view(schema_map.resolve_kernel(tree, target), target)
            after = schema_map.logical_view(schema_map.resolve_kernel(merged, target), target)
            expected = before + module.scale * (module.A.T @ module.B.T)
            assert np.allclose(after, expected, atol=1e-4), name
            assert stats["modules"][module.key]["relative_delta"] > 0.0

    # untouched leaves are shared, not copied
    assert merged["action_in_proj"]["bias"] is tree["action_in_proj"]["bias"]
    # and the source tree is unchanged
    assert set(tree.keys()) == {"PaliGemma", "action_in_proj"}


def test_merge_canonical_respects_keep_indices_and_zero_b(tmp_path: Path) -> None:
    tree = _make_tree()
    specs = [CASES[0][1], CASES[1][1]]
    arrays = {
        "module_0_A": np.ones((2, 4), dtype=np.float32),
        "module_0_B": np.zeros((6, 2), dtype=np.float32),
        "module_1_A": np.ones((2, 4), dtype=np.float32),
        "module_1_B": np.ones((3, 2), dtype=np.float32),
    }
    path = tmp_path / "params.canonical.npz"
    save_canonical(path, {"modules": specs}, arrays)

    from mlvla.meta.weights.lora_io import load_canonical

    with load_canonical(path) as lora:
        merged, stats = merge_canonical(tree, lora, keep_indices=[1])
        assert stats["merged_modules"] == 1
        assert stats["skipped_zero_modules"] == 0
        target = schema_map.build_targets(specs)[0]
        assert np.allclose(
            schema_map.resolve_kernel(merged, target), schema_map.resolve_kernel(tree, target)
        )
        # without keep_indices the zero-B module is skipped, not merged
        merged, stats = merge_canonical(tree, lora)
        assert stats["merged_modules"] == 1
        assert stats["skipped_zero_modules"] == 1
