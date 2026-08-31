from __future__ import annotations
from pathlib import Path

import pytest
import yaml

REGISTRY = Path(__file__).resolve().parents[1] / "configs" / "domains.yaml"
requires_data = pytest.mark.skipif(not REGISTRY.exists(), reason="registry not built")


@requires_data
def test_registry_invariants():
    domains = yaml.safe_load(REGISTRY.read_text())
    # 9 perturbed conditions x 10 tasks + the clean oracle domain
    assert len(domains) == 91
    conditions = {d["condition"] for d in domains.values()}
    assert len(conditions) == 10
    for domain_id, d in domains.items():
        assert domain_id == f"{d['condition']}__{d['task_hash']}"
        assert Path(d["canonical_npz"]).exists()
        assert 25 <= len(d["episodes"]) <= 50
    # every perturbed condition covers the same 10 task hashes; clean is a single task
    by_cond = {}
    for d in domains.values():
        by_cond.setdefault(d["condition"], set()).add(d["task_hash"])
    perturbed = {c: v for c, v in by_cond.items() if c != "clean"}
    assert all(len(v) == 10 for v in perturbed.values())
    assert len(by_cond["clean"]) == 1
