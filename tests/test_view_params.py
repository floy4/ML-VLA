import math
from pathlib import Path

import pytest
import yaml

from mlvla.meta.view_params import VIEW_PARAMS, VIEW_DIM, append_view_params, view_vector, view_vector_for_domain

from mlvla import paths as _paths

CONDITIONS_YAML = Path(_paths.get("perturbed_root")) / "conditions.yaml"
requires_conditions = pytest.mark.skipif(not CONDITIONS_YAML.exists(), reason="conditions.yaml not generated")
DEF_POS = [0.6586131746834771, 0.0, 1.6103500242372423]
DEF_QUAT = [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]


def _qmul(q, r):
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return [w0*w1-x0*x1-y0*y1-z0*z1, w0*x1+x0*w1+y0*z1-z0*y1,
            w0*y1-x0*z1+y0*w1+z0*x1, w0*z1+x0*y1-y0*x1+z0*w1]


def _qconj(q):
    return [q[0], -q[1], -q[2], -q[3]]


def test_table_basics():
    assert VIEW_DIM == 7
    # 9 perturbed conditions + clean (identity view)
    assert len(VIEW_PARAMS) == 10
    for vec in VIEW_PARAMS.values():
        assert len(vec) == VIEW_DIM
        assert math.isclose(sum(v * v for v in vec[3:]), 1.0, abs_tol=1e-5)


def test_combos_equal_view_components():
    assert VIEW_PARAMS["c1_v1l1"] == VIEW_PARAMS["v1_azimuth30"]
    assert VIEW_PARAMS["c2_v2l2"] == VIEW_PARAMS["v2_azimuth60"]
    assert VIEW_PARAMS["c3_v3l3"] == VIEW_PARAMS["v3_elev15_zoom125"]


def test_lights_are_identity():
    identity = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    for cond in ("l1_warm_dim", "l2_cool_bright", "l3_directional_low"):
        assert VIEW_PARAMS[cond] == identity


def test_views_distinct():
    views = {tuple(VIEW_PARAMS[c]) for c in ("v1_azimuth30", "v2_azimuth60", "v3_elev15_zoom125")}
    assert len(views) == 3


def test_unknown_condition_raises():
    with pytest.raises(ValueError):
        view_vector("v4_nonexistent")


def test_domain_parsing():
    assert view_vector_for_domain("v3_elev15_zoom125__de2a6ce7") == VIEW_PARAMS["v3_elev15_zoom125"]


def test_append_view_params():
    import torch

    feats = torch.zeros(3, 4, 1024)
    out = append_view_params(feats, "v1_azimuth30__de2a6ce7")
    assert out.shape == (3, 4, 1024 + VIEW_DIM)
    expected = torch.tensor(VIEW_PARAMS["v1_azimuth30"]).expand(3, 4, VIEW_DIM)
    assert torch.allclose(out[..., 1024:], expected)


def test_table_matches_conditions_yaml():
    if not CONDITIONS_YAML.exists():
        pytest.skip("conditions.yaml not present")
    conds = yaml.safe_load(CONDITIONS_YAML.read_text())["conditions"]
    identity = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    for name, entry in conds.items():
        cam = (entry.get("view") or {}).get("effective_camera")
        if cam is None:
            expected = identity
        else:
            dpos = [cam["pos"][i] - DEF_POS[i] for i in range(3)]
            qr = _qmul(_qconj(DEF_QUAT), cam["quat"])
            if qr[0] < 0:
                qr = [-v for v in qr]
            expected = dpos + qr
        assert all(abs(a - b) < 1e-5 for a, b in zip(VIEW_PARAMS[name], expected)), name
