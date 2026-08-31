from __future__ import annotations
from pathlib import Path
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
requires = pytest.mark.skipif(
    not (ROOT / "configs" / "splits.yaml").exists(), reason="splits not built"
)


@requires
def test_split_invariants():
    splits = yaml.safe_load((ROOT / "configs" / "splits.yaml").read_text())
    domains = yaml.safe_load((ROOT / "configs" / "domains.yaml").read_text())
    counts = {"rung1": (90, 90), "rung2": (72, 18), "rung3": (70, 10), "rung4": (70, 10), "rung5": (60, 30)}
    for name, (n_train, n_test) in counts.items():
        s = splits[name]
        assert len(s["train"]) == n_train and len(s["test"]) == n_test
        if name != "rung1":  # rung1 test==train by design
            assert not set(s["train"]) & set(s["test"])
    # full coverage for rung2/rung5; rung1 test==train; rung3/rung4 exclude c3 entirely
    for name in ("rung2", "rung5"):
        assert set(splits[name]["train"]) | set(splits[name]["test"]) == set(domains), name
    cond = {d: d.split("__")[0] for d in domains}
    # rung3 tests unseen view v3: no v3 or c3 (=v3+l3) condition in training
    assert not any(cond[d] in ("v3_elev15_zoom125", "c3_v3l3") for d in splits["rung3"]["train"])
    assert all(cond[d] == "v3_elev15_zoom125" for d in splits["rung3"]["test"])
    # rung4 tests unseen light l3: no l3 or c3 condition in training
    assert not any(cond[d] in ("l3_directional_low", "c3_v3l3") for d in splits["rung4"]["train"])
    assert all(cond[d] == "l3_directional_low" for d in splits["rung4"]["test"])
    # rung5 trains only single-factor conditions, tests all three combined ones
    assert all(cond[d][0] in "vl" for d in splits["rung5"]["train"])
    assert all(cond[d][0] == "c" for d in splits["rung5"]["test"])
    # rung2 holds out the same 2 task hashes in every condition
    held = {d.split("__")[1] for d in splits["rung2"]["test"]}
    assert held == {"ab3beaf7", "a1d2e98c"}
