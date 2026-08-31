from __future__ import annotations
import json
import numpy as np
import pytest

from mlvla.meta.weights.lora_io import assert_compatible, load_canonical, save_canonical

EXPERT = "/data2/zhy/models/wizard/experts_perturbed_lagfix/v1_azimuth30/expert_0006744e/phase1/final/params.canonical.npz"

requires_expert = pytest.mark.skipif(
    not __import__("pathlib").Path(EXPERT).exists(), reason="expert npz not mounted"
)


def _synthetic(tmp_path):
    manifest = {
        "base_checkpoint": "pi05_base",
        "modules": [
            {"key": "block_a", "alpha": 16.0},
            {"key": "block_b", "alpha": 16.0},
        ],
    }
    arrays = {
        "module_0_A": np.arange(8, dtype=np.float32).reshape(2, 4),
        "module_0_B": np.ones((6, 2), dtype=np.float32),
        "module_1_A": np.ones((2, 3), dtype=np.float32),
        "module_1_B": np.arange(10, dtype=np.float32).reshape(5, 2),
    }
    path = tmp_path / "synthetic.canonical.npz"
    save_canonical(path, manifest, arrays)
    return path, manifest, arrays


def test_roundtrip(tmp_path):
    path, manifest, arrays = _synthetic(tmp_path)
    with load_canonical(path) as f:
        assert len(f) == 2
        m0 = f.module(0)
        assert m0.key == "block_a" and m0.rank == 2 and m0.alpha == 16.0
        np.testing.assert_array_equal(m0.A, arrays["module_0_A"])
        np.testing.assert_array_equal(m0.B, arrays["module_0_B"])
        keys = [m.key for m in f.modules()]
    assert keys == ["block_a", "block_b"]


def test_assert_compatible(tmp_path):
    p1, _, _ = _synthetic(tmp_path)
    p2, _, _ = _synthetic(tmp_path / "sub")
    with load_canonical(p1) as f1, load_canonical(p2) as f2:
        assert_compatible(f1, f2)  # no raise


@requires_expert
def test_real_expert_loads():
    with load_canonical(EXPERT) as f:
        assert len(f) == 458
        m3 = f.module(3)
        assert m3.A.ndim == 2 and m3.B.ndim == 2 and m3.A.shape[0] == m3.B.shape[1]
        assert m3.A.dtype == np.float32
