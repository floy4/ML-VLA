from __future__ import annotations
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_spec = importlib.util.spec_from_file_location("eval_hypernet_mod", _SCRIPTS / "eval_hypernet.py")
eval_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_mod)


def test_eval_script_help():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "eval_hypernet.py"), "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--checkpoint" in result.stdout


def test_nearest_oracle_over_full_rung_pool():
    """Nearest-oracle must compare each prediction against ALL rung oracles
    (train ∪ test), not just the eval set's own oracles."""
    torch.manual_seed(0)

    def domain_targets(seed):
        g = torch.Generator().manual_seed(seed)
        return {"m0": {"A": torch.randn(2, 4, generator=g), "B": torch.randn(6, 2, generator=g)}}

    targets = {f"d{i}": domain_targets(100 + i) for i in range(3)}  # pool: d0,d1,d2
    scales = {"m0": {"A": 1.0, "B": 1.0}}

    # prediction exactly equals d1's oracle -> nearest among the full pool is d1
    pred_own = {k: {f: t.unsqueeze(0).clone() for f, t in factors.items()}
                for k, factors in targets["d1"].items()}
    hits = eval_mod.nearest_oracle_hits([pred_own], ["d1"], ["d0", "d1", "d2"], targets, scales)
    assert hits == 1

    # prediction closer to d0 (a non-eval pool oracle) than to eval domain d1
    # -> NOT a hit under the full-pool metric (an eval-set-only denominator would
    #    have scored it 1/1)
    pred_wrong = {k: {f: t.unsqueeze(0).clone() for f, t in factors.items()}
                  for k, factors in targets["d0"].items()}
    hits_wrong = eval_mod.nearest_oracle_hits([pred_wrong], ["d1"], ["d0", "d1", "d2"], targets, scales)
    assert hits_wrong == 0
