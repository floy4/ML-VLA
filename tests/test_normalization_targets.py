from __future__ import annotations
import json
import numpy as np
import torch

from mlvla.meta.weights.lora_io import load_canonical, save_canonical
from mlvla.meta.weights.normalization import compute_rms_scales, scales_to_dict
from mlvla.meta.weights.targets import load_direct_targets, selected_rows


def _write_oracle(tmp_path, name, scale=1.0):
    manifest = {"base_checkpoint": "pi05_base", "modules": [{"key": "m0", "alpha": 16.0}]}
    arrays = {
        "module_0_A": (scale * np.ones((2, 3), dtype=np.float32)),
        "module_0_B": (scale * np.ones((4, 2), dtype=np.float32)),
    }
    path = tmp_path / f"{name}.npz"
    save_canonical(path, manifest, arrays)
    return path


def test_rms_scales(tmp_path):
    p0, p1 = _write_oracle(tmp_path, "d0", 1.0), _write_oracle(tmp_path, "d1", 3.0)
    rows = [{"source_index": 0, "key": "m0"}]
    targets, shapes, modules = load_direct_targets({"d0": p0, "d1": p1}, rows)
    assert shapes == {"m0": {"A": (2, 3), "B": (4, 2)}}
    assert modules == [{"source_index": 0, "key": "m0"}]
    assert set(targets) == {"d0", "d1"}
    assert torch.allclose(targets["d0"]["m0"]["A"], torch.ones(2, 3))
    scales = compute_rms_scales(targets)
    # RMS over both domains: sqrt((1^2*6 + 3^2*6) / 12)
    expected_a = float(np.sqrt((6 * 1 + 6 * 9) / 12))
    assert abs(float(scales["m0"]["A"]) - expected_a) < 1e-5
    d = scales_to_dict(scales)
    assert isinstance(d["m0"]["A"], float) and abs(d["m0"]["A"] - expected_a) < 1e-5


def test_selected_rows(tmp_path):
    rows = [{"source_index": i, "key": f"m{i}", "sensitivity": float(9 - i)} for i in range(9)]
    p = tmp_path / "sel.json"
    p.write_text(json.dumps(rows))
    assert [r["key"] for r in selected_rows(p)] == [f"m{i}" for i in range(9)]
    assert [r["key"] for r in selected_rows(p, 3)] == ["m0", "m1", "m2"]
