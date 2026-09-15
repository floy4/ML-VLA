"""Task-5 pose_feature scheme wiring tests (CPU-only).

(a) 4 schemes' EvidenceBank.view() shape/finiteness over all 10 lagfix
    conditions (8 train + l3/c3 OOD);
(b) regression protection: NO pose_feature block -> view() bit-identical to the
    legacy view_vector_for_domain for every condition;
(c) all conditions (OOD included) have keys in both pose npz caches;
(d) scheme-aware cond [k,4,1024+pose_dim] reaches a real V4 model forward at
    the vggt arm width 3072, and each of the three arm yamls builds a model
    whose forward accepts its own cond width.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO = Path(__file__).resolve().parents[1]
VGGT_CACHE = Path("/data2/zhy/meta_lora_offload/pose_cache/vggt_pose_cache.npz")
RAYMAP_CACHE = Path("/data2/zhy/meta_lora_offload/pose_cache/raymap_pose_cache.npz")
FEATURE_DIR = Path("/home/zhy/vla/meta_lora/outputs/feature_cache_90")

# 10 lagfix conditions: 8 train + 2 OOD (all must resolve under every scheme)
CONDITIONS = [
    "clean", "c1_v1l1", "c2_v2l2", "l1_warm_dim", "l2_cool_bright",
    "v1_azimuth30", "v2_azimuth60", "v3_elev15_zoom125",
    "l3_directional_low", "c3_v3l3",
]

requires_pose_caches = pytest.mark.skipif(
    not (VGGT_CACHE.exists() and RAYMAP_CACHE.exists()),
    reason="pose feature npz caches not present")
requires_feature_dir = pytest.mark.skipif(
    not FEATURE_DIR.exists(), reason="feature_cache_90 not present")


def _make_bank(tmp_path: Path, pose_feature: dict | None = None,
               domains: tuple[str, ...] = ("d1__t",)):
    """Tiny synthetic evidence cache (same layout as test_evidence.py)."""
    from mlvla.meta.e2e.evidence import EvidenceBank
    for domain in domains:
        for split, count in (("train", 4), ("val", 1), ("test", 1)):
            torch.save({
                "features": torch.randn(count, 4, 1024),
                "cls": torch.randn(count, 4, 1024),
                "episode_ids": torch.arange(count),
                "domain": domain, "split": split,
            }, tmp_path / f"{domain}_{split}.pt")
    return EvidenceBank(tmp_path, list(domains), pose_feature=pose_feature)


def _tiny_concat(pose_dim: int):
    """Small synthetic V4 hypernet at an arbitrary cond width."""
    from mlvla.meta.e2e.train_bridge import build_model, model_args_for
    shapes = {
        "m1": {"A": (4, 8), "B": (8, 4)},
        "m2": {"A": (4, 8), "B": (8, 4)},
    }
    hyper = {"dino_dim": 1024, "condition_hidden_dim": 32, "hidden_dim": 64,
             "id_dim": 16, "trunk_layers": 2, "trunk_hidden": 32}
    args = model_args_for("concat", shapes, hyper, pose_dim=pose_dim)
    return args, build_model("concat", shapes, hyper, pose_dim=pose_dim)


# ---------------------------------------------------------------- (a) schemes

@pytest.mark.parametrize("scheme,pose_cfg,dim", [
    ("view7", None, 7),
    ("zero", {"scheme": "zero"}, 7),
    ("vggt", {"scheme": "vggt", "cache": str(VGGT_CACHE), "layer": 23}, 2048),
    ("raymap", {"scheme": "raymap", "cache": str(RAYMAP_CACHE)}, 2048),
])
def test_view_scheme_shapes(tmp_path, scheme, pose_cfg, dim):
    if scheme in ("vggt", "raymap") and not (VGGT_CACHE.exists() and RAYMAP_CACHE.exists()):
        pytest.skip("pose feature npz caches not present")
    bank = _make_bank(tmp_path, pose_cfg)
    for cond in CONDITIONS:
        v = bank.view(f"{cond}__de2a6ce7")
        assert v.shape == (dim,), (cond, tuple(v.shape))
        assert v.dtype == torch.float32
        assert torch.isfinite(v).all(), cond
    if scheme == "zero":
        assert torch.equal(bank.view("clean__de2a6ce7"), torch.zeros(7))
    if scheme == "vggt":
        # camera-moved conditions must carry distinct pose features (the whole
        # point of the scheme); lighting stays ~= clean's, like view7 identity
        vecs = {c: bank.view(f"{c}__de2a6ce7") for c in CONDITIONS}
        for a, b in (("v1_azimuth30", "v2_azimuth60"),
                     ("v1_azimuth30", "v3_elev15_zoom125"),
                     ("v2_azimuth60", "v3_elev15_zoom125")):
            assert not torch.allclose(vecs[a], vecs[b]), (a, b)


# ------------------------------------------------- (b) legacy bit-identity

def test_legacy_view_bit_identical(tmp_path):
    """No pose_feature block -> EXACT legacy view_vector_for_domain output."""
    from mlvla.meta.view_params import pose_vector_from_config, view_vector_for_domain
    bank = _make_bank(tmp_path)  # pose_feature=None
    for cond in CONDITIONS:
        d = f"{cond}__de2a6ce7"
        legacy = torch.tensor(view_vector_for_domain(d), dtype=torch.float32)
        assert torch.equal(bank.view(d), legacy), cond
        # the loader's empty-config branch is the same legacy path
        assert torch.equal(pose_vector_from_config(d, None), legacy), cond


def test_append_view_params_legacy_and_schemes():
    from mlvla.meta.view_params import append_view_params
    feats = torch.zeros(3, 4, 1024)
    legacy = append_view_params(feats, "v1_azimuth30__de2a6ce7")
    assert legacy.shape == (3, 4, 1031)
    # explicit view7 block must be bit-identical to the absent-block path
    out7 = append_view_params(feats, "v1_azimuth30__de2a6ce7", {"scheme": "view7"})
    assert torch.equal(legacy, out7)
    out0 = append_view_params(feats, "v1_azimuth30__de2a6ce7", {"scheme": "zero"})
    assert out0.shape == (3, 4, 1031)
    assert torch.equal(out0[..., 1024:], torch.zeros(3, 4, 7))


@requires_pose_caches
def test_append_view_params_wide_schemes():
    from mlvla.meta.view_params import append_view_params
    feats = torch.zeros(3, 4, 1024)
    for pose_cfg, dim in (
        ({"scheme": "vggt", "cache": str(VGGT_CACHE), "layer": 23}, 2048),
        ({"scheme": "raymap", "cache": str(RAYMAP_CACHE)}, 2048),
    ):
        out = append_view_params(feats, "v1_azimuth30__de2a6ce7", pose_cfg)
        assert out.shape == (3, 4, 1024 + dim), pose_cfg
        assert torch.isfinite(out).all()


# -------------------------------------------------------- (c) cache coverage

@requires_pose_caches
def test_all_conditions_in_both_caches():
    import numpy as np
    for path in (VGGT_CACHE, RAYMAP_CACHE):
        with np.load(path) as z:
            keys = set(z.files)
        for cond in CONDITIONS:
            assert f"cond:{cond}" in keys, (path.name, cond)
        # the OOD conditions explicitly (l3/c3 — never trained on)
        assert "cond:l3_directional_low" in keys
        assert "cond:c3_v3l3" in keys


@requires_pose_caches
def test_loader_matches_independent_numpy():
    import numpy as np
    from mlvla.meta.view_params import pose_vector_for_domain
    with np.load(VGGT_CACHE) as z:
        blob = {k: z[k] for k in z.files if k.startswith("cond:")}
    expected = torch.from_numpy(blob["cond:clean"][23].astype(np.float32).mean(0))
    got = pose_vector_for_domain("clean__de2a6ce7", scheme="vggt",
                                 cache=VGGT_CACHE, layer=23)
    assert got.shape == (2048,)
    assert torch.equal(got, expected)
    with np.load(RAYMAP_CACHE) as z:
        expected_r = torch.from_numpy(z["cond:c1_v1l1"])
    got_r = pose_vector_for_domain("c1_v1l1__de2a6ce7", scheme="raymap",
                                   cache=RAYMAP_CACHE)
    assert got_r.shape == (2048,)
    assert torch.equal(got_r, expected_r)
    # process-level lru_cache: same resolved path -> same dict object
    from mlvla.meta.view_params import _load_pose_cache
    assert _load_pose_cache(str(VGGT_CACHE)) is _load_pose_cache(str(VGGT_CACHE))


# ------------------------------------------- (d) cond -> model forward width

def test_model_args_pose_dim_widths():
    from mlvla.meta.e2e.train_bridge import model_args_for
    hyper = {"dino_dim": 1024, "condition_hidden_dim": 32, "hidden_dim": 64,
             "id_dim": 16, "trunk_layers": 2, "trunk_hidden": 32}
    shapes = {"m1": {"A": (4, 8), "B": (8, 4)}}
    # legacy default (no pose scheme): 1031
    assert model_args_for("concat", shapes, hyper)["condition_dim"] == 1031
    assert model_args_for("concat", shapes, hyper, pose_dim=7)["condition_dim"] == 1031
    # vggt/raymap arm width: 3072
    assert model_args_for("concat", shapes, hyper, pose_dim=2048)["condition_dim"] == 3072


@requires_pose_caches
@requires_feature_dir
def test_vggt_cond_reaches_model_forward():
    """Real evidence blob + real pose cache -> [k,4,3072] cond -> V4 forward.

    Mirrors train_bridge's concat assembly exactly (broadcast pose onto the 4
    frames, V4 frame-means), on CPU.
    """
    from mlvla.meta.e2e.evidence import EvidenceBank
    args, model = _tiny_concat(2048)
    assert args["condition_dim"] == 3072
    bank = EvidenceBank(FEATURE_DIR, ["clean__de2a6ce7"],
                        pose_feature={"scheme": "vggt", "cache": str(VGGT_CACHE),
                                      "layer": 23})
    g = torch.Generator().manual_seed(0)
    x = bank.sample("clean__de2a6ce7", k=8, generator=g)          # [8,4,1024]
    v = bank.view("clean__de2a6ce7")                              # [2048]
    cond = torch.cat([x, v.unsqueeze(0).unsqueeze(1)
                        .expand(x.shape[0], x.shape[1], -1)], dim=-1)
    assert cond.shape == (8, 4, 1024 + 2048)
    model.eval()
    with torch.inference_mode():
        out = model(cond)
    assert set(out) == {"m1", "m2"}
    for ab in out.values():
        assert ab["A"].shape == (8, 4, 8)
        assert ab["B"].shape == (8, 8, 4)
        assert torch.isfinite(ab["A"]).all() and torch.isfinite(ab["B"]).all()


@requires_pose_caches
@pytest.mark.parametrize("yaml_name,expected_dim", [
    ("hypernet_e2e_concat_vggt.yaml", 3072),
    ("hypernet_e2e_concat_raymap.yaml", 3072),
    ("hypernet_e2e_concat_zeropose.yaml", 1031),
])
def test_arm_yamls_working_forward(yaml_name, expected_dim):
    """Each arm yaml: max_steps untouched, scheme resolves, model forward runs
    at the yaml's own cond width with its own pose vector."""
    import yaml
    from mlvla.meta.view_params import pose_dim_for_scheme, pose_vector_from_config
    cfg = yaml.safe_load((REPO / "configs" / yaml_name).read_text())
    assert int(cfg["training"]["max_steps"]) == 60000
    pf = cfg["pose_feature"]
    pose_dim = pose_dim_for_scheme(pf["scheme"])
    args, model = _tiny_concat(pose_dim)
    assert args["condition_dim"] == expected_dim
    v = pose_vector_from_config("clean__de2a6ce7", pf)
    assert v.shape == (pose_dim,) and torch.isfinite(v).all()
    x = torch.randn(2, 4, 1024)
    cond = torch.cat([x, v.view(1, 1, -1).expand(2, 4, -1)], dim=-1)
    model.eval()
    with torch.inference_mode():
        out = model(cond)
    assert set(out) == {"m1", "m2"}
