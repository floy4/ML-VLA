from __future__ import annotations
from pathlib import Path

import pytest
import yaml

REGISTRY = Path(__file__).resolve().parents[1] / "configs" / "domains.yaml"
requires_data = pytest.mark.skipif(not REGISTRY.exists(), reason="registry not built")


@requires_data
def test_registry_invariants():
    domains = yaml.safe_load(REGISTRY.read_text())
    lagfix = {k: v for k, v in domains.items() if "__" in k}
    level = {k: v for k, v in domains.items() if "__" not in k}
    # 9 perturbed conditions x 10 tasks + the clean oracle domain
    assert len(lagfix) == 91
    conditions = {d["condition"] for d in lagfix.values()}
    assert conditions == {
        "c1_v1l1", "c2_v2l2", "c3_v3l3", "l1_warm_dim", "l2_cool_bright",
        "l3_directional_low", "v1_azimuth30", "v2_azimuth60",
        "v3_elev15_zoom125", "clean",
    }
    for domain_id, d in lagfix.items():
        assert domain_id == f"{d['condition']}__{d['task_hash']}"
        assert Path(d["canonical_npz"]).exists()
        assert 25 <= len(d["episodes"]) <= 50
    # every perturbed condition covers the same 10 task hashes; clean is a single task
    by_cond = {}
    for d in lagfix.values():
        by_cond.setdefault(d["condition"], set()).add(d["task_hash"])
    perturbed = {c: v for c, v in by_cond.items() if c != "clean"}
    assert all(len(v) == 10 for v in perturbed.values())
    assert len(by_cond["clean"]) == 1
    # 12 level multitask domains ({camera,lighting,noise,texture} x L1-3):
    # id == condition (no task hash), full-dataset episode range, oracle npz
    assert len(level) == 12
    assert set(level) == {
        f"{c}_L{l}" for c in ("camera", "lighting", "noise", "texture") for l in "123"
    }
    for domain_id, d in level.items():
        assert d["condition"] == domain_id
        assert Path(d["canonical_npz"]).exists()
        assert d["episodes"] == list(range(len(d["episodes"])))
        assert len(d["episodes"]) >= 50
