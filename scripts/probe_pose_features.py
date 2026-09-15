#!/usr/bin/env python3
"""Diagnostic probes on the offline pose-feature caches (VGGT + ray-map).

Step 1 — per-layer linear probe (VGGT layer selection).
  For every VGGT aggregator layer l in 0..23, fit a ridge regression from the
  2048-dim per-episode CAMERA token (frame-branch || global-branch concat) to
  the ground-truth 7-dim view vector [dpos(3), dquat_wxyz(4)] of
  mlvla.meta.view_params (the same computation the e2e training condition
  uses). Pose-bearing conditions: clean, v1_azimuth30, v2_azimuth60,
  v3_elev15_zoom125, c1_v1l1, c2_v2l2, c3_v3l3 — i.e. 4 distinct poses
  (identity, v1, v2, v3) because each c* shares its v* view vector.

  Metrics (deterministic split: per condition, episodes sorted ascending,
  first 21 -> train, last 9 -> test):
    rot_err_deg  mean geodesic angle between predicted and true dquat,
                 theta = 2*arccos(|<q_hat_norm, q_true_norm>|) in degrees.
                 view_params dquat convention: q_default^-1 (x) q_cond,
                 [w, x, y, z], w >= 0; quaternion geodesic distance is
                 sign-invariant so no re-signing is needed.
    trans_err_cm mean ||t_hat - t_true|| (dpos is in meters).
  Secondary: leave-one-condition-out (LOCO) rotation error over the 7
  pose-bearing conditions (whole condition held out: unseen lighting for a
  seen pose, or an unseen pose).

  Layer ruling: minimum primary (test-split) rotation error; layers within
  1 deg of the minimum tie -> deepest (largest) layer index.

Step 1b — lighting invariance spot check.
  Light conditions (l1/l2/l3) have identity ground-truth pose. Episode ids
  are NOT aligned across conditions (per-domain train pools differ), so we
  report full 30x30 cross-pair cosine statistics of the light condition's
  tokens vs clean's tokens per layer (brief threshold: > 0.95), plus a
  stricter variant with the global mean token (mean over all 10 conditions x
  30 episodes at that layer) removed, which strips the large shared
  camera-token component that otherwise dominates the raw cosine.

Step 2 — stability & separability.
  Stability per layer: intra-condition mean pairwise L2 distance across
  episodes vs inter-condition mean pairwise L2 distance; ratio < 0.5 passes
  (gauge noise controlled).
  Separability per layer: 10-condition logistic regression on per-episode
  tokens, 5-fold stratified CV accuracy. Baselines: chance (1/10) and the
  7-dim ground-truth pose vector as the only feature (its ceiling is 0.4:
  pose groups {clean,l1,l2,l3}, {v1,c1}, {v2,c2}, {v3,c3} are
  indistinguishable from pose alone).
  Ray-map comparison: the ray-map cache is deterministic (ONE 2048-dim
  vector per condition, no episode axis), so per-episode CV does not apply.
  We report (a) min pairwise cosine across the 10 condition vectors
  (lighting conditions share the identity ray map by construction) and (b)
  a condition-level leave-one-out ridge pose probe (train on 9 condition
  vectors -> predict the 10th's 7-dim pose, rot/trans error), run on the
  ray-map features and — same protocol — on the VGGT per-condition mean
  tokens for direct comparison.

Usage:
  CUDA_VISIBLE_DEVICES=2 conda run -n openvla python scripts/probe_pose_features.py
(CPU is enough for every step; no GPU is required.)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np

from mlvla.meta.view_params import VIEW_PARAMS

DEFAULT_VGGT = Path("/data2/zhy/meta_lora_offload/pose_cache/vggt_pose_cache.npz")
DEFAULT_RAYMAP = Path("/data2/zhy/meta_lora_offload/pose_cache/raymap_pose_cache.npz")
DEFAULT_OUT = Path("/data2/zhy/meta_lora_offload/pose_cache/probe_tables.json")

N_LAYERS = 24
N_TRAIN_EP = 21          # per condition, of 30
POSE_CONDS = [
    "clean", "v1_azimuth30", "v2_azimuth60", "v3_elev15_zoom125",
    "c1_v1l1", "c2_v2l2", "c3_v3l3",
]
LIGHT_CONDS = ["l1_warm_dim", "l2_cool_bright", "l3_directional_low"]
ALL_CONDS = POSE_CONDS + LIGHT_CONDS
ALPHAS = [1e-2, 1e-1, 1.0, 10.0, 100.0]


# --------------------------------------------------------------------------- #
# ridge probe (closed form, standardized features / targets)
# --------------------------------------------------------------------------- #
def fit_ridge(x_tr: np.ndarray, y_tr: np.ndarray, alpha: float):
    mu_x, sd_x = x_tr.mean(0), x_tr.std(0).clip(1e-8)
    mu_y = y_tr.mean(0)
    xs = (x_tr - mu_x) / sd_x
    ys = y_tr - mu_y
    d = xs.shape[1]
    w = np.linalg.solve(xs.T @ xs + alpha * np.eye(d), xs.T @ ys)
    return mu_x, sd_x, mu_y, w


def apply_ridge(model, x: np.ndarray) -> np.ndarray:
    mu_x, sd_x, mu_y, w = model
    return ((x - mu_x) / sd_x) @ w + mu_y


def pick_alpha(x_tr, y_tr) -> float:
    """Inner fit/val split of the train episodes; select by val MSE on the
    raw 7-dim target."""
    n_fit = 14
    inner = {}
    for a in ALPHAS:
        m = fit_ridge(x_tr[:n_fit], y_tr[:n_fit], a)
        pred = apply_ridge(m, x_tr[n_fit:])
        inner[a] = float(((pred - y_tr[n_fit:]) ** 2).mean())
    return min(inner, key=inner.get)


def rot_err_deg(q_pred: np.ndarray, q_true: np.ndarray) -> float:
    """Geodesic angle (deg) between two [w,x,y,z] quaternions."""
    q_pred = q_pred / max(np.linalg.norm(q_pred), 1e-12)
    q_true = q_true / max(np.linalg.norm(q_true), 1e-12)
    d = abs(float(np.dot(q_pred, q_true)))
    return math.degrees(2.0 * math.acos(min(1.0, d)))


def probe_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    rots, trans = [], []
    for p, t in zip(pred, truth):
        rots.append(rot_err_deg(p[3:7], t[3:7]))
        trans.append(float(np.linalg.norm(p[:3] - t[:3])))
    return {"rot_err_deg": float(np.mean(rots)),
            "rot_err_deg_max": float(np.max(rots)),
            "trans_err_cm": float(np.mean(trans) * 100.0)}


# --------------------------------------------------------------------------- #
# Step 1: per-layer probe
# --------------------------------------------------------------------------- #
def layer_probe(vggt: dict) -> tuple[list[dict], dict]:
    truth = {c: np.asarray(VIEW_PARAMS[c], dtype=np.float64) for c in POSE_CONDS}
    rows = []
    for layer in range(N_LAYERS):
        # [cond, ep, 2048] -> flattened train/test arrays with positional split.
        # X rows are grouped by condition (n consecutive episodes each), so the
        # target block must be np.repeat (condition row held for its n episodes),
        # NOT np.tile (which repeats the whole condition block n times and
        # misaligns pairs — an earlier draft did exactly that and produced
        # trivial-baseline errors).
        per_cond = {c: vggt[f"cond:{c}"][layer].astype(np.float64) for c in POSE_CONDS}
        x_tr = np.concatenate([per_cond[c][:N_TRAIN_EP] for c in POSE_CONDS])
        y_tr = np.repeat(np.stack([truth[c] for c in POSE_CONDS]), N_TRAIN_EP, axis=0)
        x_te = np.concatenate([per_cond[c][N_TRAIN_EP:] for c in POSE_CONDS])
        y_te = np.repeat(np.stack([truth[c] for c in POSE_CONDS]), 30 - N_TRAIN_EP, axis=0)

        alpha = pick_alpha(x_tr, y_tr)
        m = fit_ridge(x_tr, y_tr, alpha)
        test = probe_metrics(apply_ridge(m, x_te), y_te)

        # leave-one-condition-out over the 7 pose-bearing conditions
        loco = []
        for i, held in enumerate(POSE_CONDS):
            idx = [j for j in range(len(POSE_CONDS)) if j != i]
            xl = np.concatenate([per_cond[POSE_CONDS[j]] for j in idx])
            yl = np.repeat(np.stack([truth[POSE_CONDS[j]] for j in idx]), 30, axis=0)
            ml = fit_ridge(xl, yl, alpha)
            loco.append(probe_metrics(apply_ridge(ml, per_cond[held]),
                                      np.tile(truth[held], (30, 1))))
        rows.append({
            "layer": layer, "alpha": alpha, **test,
            "loco_rot_err_deg": float(np.mean([r["rot_err_deg"] for r in loco])),
            "loco_trans_err_cm": float(np.mean([r["trans_err_cm"] for r in loco])),
            "loco_per_cond": {POSE_CONDS[i]: round(locor["rot_err_deg"], 3)
                              for i, locor in enumerate(loco)},
        })
        print(f"[probe] layer {layer:2d} alpha={alpha:>6g} rot={test['rot_err_deg']:7.3f} deg "
              f"(max {test['rot_err_deg_max']:7.3f}) trans={test['trans_err_cm']:6.3f} cm "
              f"loco_rot={rows[-1]['loco_rot_err_deg']:7.3f} deg", flush=True)

    best = min(r["rot_err_deg"] for r in rows)
    tied = [r["layer"] for r in rows if r["rot_err_deg"] <= best + 1.0]
    ruling = max(tied)  # ties within 1 deg -> deepest layer
    print(f"[ruling] best rot_err={best:.3f} deg; layers within 1 deg: {tied} "
          f"-> deepest = layer {ruling}", flush=True)
    return rows, {"min_rot_err_deg": best, "tied_layers": tied, "layer_ruling": ruling}


# --------------------------------------------------------------------------- #
# Step 1b: lighting invariance
# --------------------------------------------------------------------------- #
def lighting_invariance(vggt: dict) -> list[dict]:
    rows = []
    for layer in range(N_LAYERS):
        clean = vggt["cond:clean"][layer].astype(np.float64)
        # global mean token over ALL conditions x episodes (shared component)
        gmean = np.concatenate([vggt[f"cond:{c}"][layer] for c in ALL_CONDS]).mean(0)
        cn = clean / np.linalg.norm(clean, axis=1, keepdims=True)
        row = {"layer": layer}
        for cond in LIGHT_CONDS:
            tok = vggt[f"cond:{cond}"][layer].astype(np.float64)
            tn = tok / np.linalg.norm(tok, axis=1, keepdims=True)
            raw = float((cn @ tn.T).mean())          # mean over 30x30 cross pairs
            cs = (clean - gmean) / np.linalg.norm(clean - gmean, axis=1, keepdims=True)
            ts = (tok - gmean) / np.linalg.norm(tok - gmean, axis=1, keepdims=True)
            strict = float((cs @ ts.T).mean())
            row[cond] = {"raw_cos": round(raw, 6), "resid_cos": round(strict, 6)}
        rows.append(row)
        msg = " ".join(f"{c}={row[c]['raw_cos']:.4f}/{row[c]['resid_cos']:.4f}"
                       for c in LIGHT_CONDS)
        print(f"[light ] layer {layer:2d} raw/resid cos vs clean: {msg}", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# Step 2: stability + separability
# --------------------------------------------------------------------------- #
def pairwise_l2(a: np.ndarray) -> float:
    """Mean pairwise L2 distance within a set of vectors."""
    d = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1)
    n = a.shape[0]
    return float(d.sum() / (n * (n - 1)))  # each unordered pair counted twice


def stability(vggt: dict) -> list[dict]:
    rows = []
    for layer in range(N_LAYERS):
        feats = {c: vggt[f"cond:{c}"][layer].astype(np.float64) for c in ALL_CONDS}
        intra = float(np.mean([pairwise_l2(feats[c]) for c in ALL_CONDS]))
        names = ALL_CONDS
        inter_pairs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = feats[names[i]], feats[names[j]]
                d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
                inter_pairs.append(float(d.mean()))
        inter = float(np.mean(inter_pairs))
        rows.append({"layer": layer, "intra_l2": round(intra, 4),
                     "inter_l2": round(inter, 4), "ratio": round(intra / inter, 4)})
        print(f"[stable] layer {layer:2d} intra={intra:7.3f} inter={inter:7.3f} "
              f"ratio={intra / inter:.4f} {'PASS' if intra / inter < 0.5 else 'FAIL'}",
              flush=True)
    return rows


def separability_vggt(vggt: dict) -> tuple[list[dict], dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    conds = sorted(ALL_CONDS)
    x_all = {c: vggt[f"cond:{c}"].astype(np.float64) for c in conds}

    # baselines (condition-level, deterministic): chance + 7-dim truth pose
    y = np.array([conds.index(c) for c in ALL_CONDS for _ in range(30)])
    pose_x = np.stack([np.asarray(VIEW_PARAMS[c], dtype=np.float64)
                       for c in ALL_CONDS for _ in range(30)])
    base = {"chance": 1.0 / len(conds)}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    # raw (unstandardized) CAMERA tokens share a large mean component that
    # wrecks lbfgs conditioning -> minutes per fit. Standardize inside the CV
    # pipeline (no leakage) and parallelize the 5 folds.
    def clf():
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=1000))
    base["pose7_dim_truth"] = float(cross_val_score(
        clf(), pose_x, y, cv=cv, n_jobs=5).mean())
    print(f"[separ] baselines: chance={base['chance']:.3f} "
          f"pose7={base['pose7_dim_truth']:.3f} (ceiling 0.4: pose groups share vectors)",
          flush=True)

    rows = []
    for layer in range(N_LAYERS):
        x = np.concatenate([x_all[c][layer] for c in ALL_CONDS])
        acc = float(cross_val_score(clf(), x, y, cv=cv, n_jobs=5).mean())
        rows.append({"layer": layer, "cv_acc": round(acc, 4)})
        print(f"[separ] layer {layer:2d} 10-cond logistic CV acc={acc:.4f}", flush=True)
    return rows, base


def raymap_probe(raymap: dict, vggt: dict) -> dict:
    """Condition-level comparison: ray-map [2048] vs VGGT per-condition mean
    tokens, identical LOO ridge pose-probe protocol over the 10 conditions."""
    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    names = sorted(ALL_CONDS)
    rm = {c: raymap[f"cond:{c}"].astype(np.float64) for c in names}
    pair = {a: {b: cos(rm[a], rm[b]) for b in names} for a in names}
    offdiag = [pair[a][b] for a in names for b in names if a != b]

    y = np.stack([np.asarray(VIEW_PARAMS[c], dtype=np.float64) for c in names])

    def loo_probe(x: np.ndarray, label: str) -> dict:
        rots, trans = [], []
        for i in range(len(names)):
            idx = [j for j in range(len(names)) if j != i]
            # fixed alpha=1: with n=9 training samples the grid selection is noise
            xs, ys = x[idx], y[idx]
            m = fit_ridge(xs, ys, 1.0)
            pred = apply_ridge(m, x[i:i + 1])[0]
            rots.append(rot_err_deg(pred[3:7], y[i, 3:7]))
            trans.append(float(np.linalg.norm(pred[:3] - y[i, :3])) * 100.0)
        out = {"loo_rot_err_deg": round(float(np.mean(rots)), 3),
               "loo_trans_err_cm": round(float(np.mean(trans)), 3),
               "loo_rot_per_cond": {names[i]: round(r, 3) for i, r in enumerate(rots)}}
        print(f"[raymap] {label} LOO pose probe: rot={out['loo_rot_err_deg']} deg "
              f"trans={out['loo_trans_err_cm']} cm", flush=True)
        return out

    out = {
        "min_pairwise_cos": round(min(offdiag), 6),
        "mean_pairwise_cos": round(float(np.mean(offdiag)), 6),
        "note": ("clean/l1/l2/l3 share the identity ray map exactly and each c* "
                 "shares its v* map; cos ~1.0 within these groups is by "
                 "construction (pose-only channel, no lighting information)"),
        "pairwise_cosine": {a: {b: round(v, 4) for b, v in row.items()}
                            for a, row in pair.items()},
        "raymap": loo_probe(np.stack([rm[c] for c in names]), "raymap"),
    }
    # VGGT per-condition mean tokens at every layer, same protocol
    per_layer = {}
    for layer in range(N_LAYERS):
        x = np.stack([vggt[f"cond:{c}"][layer].astype(np.float64).mean(0) for c in names])
        per_layer[layer] = loo_probe(x, f"vggt L{layer}")
    out["vggt_cond_mean_loo"] = per_layer
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vggt-cache", type=Path, default=DEFAULT_VGGT)
    parser.add_argument("--raymap-cache", type=Path, default=DEFAULT_RAYMAP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    vggt = dict(np.load(args.vggt_cache, allow_pickle=False))
    raymap = dict(np.load(args.raymap_cache, allow_pickle=False))
    assert set(f"cond:{c}" for c in ALL_CONDS) <= set(vggt), "vggt cache missing condition"

    meta = json.loads(str(vggt["meta"]))
    assert meta["n_layers"] == N_LAYERS and meta["token_dim"] == 2048

    probe_rows, ruling = layer_probe(vggt)
    light_rows = lighting_invariance(vggt)
    stab_rows = stability(vggt)
    sep_rows, sep_base = separability_vggt(vggt)
    rm = raymap_probe(raymap, vggt)

    # trivial baseline: always predict the identity pose (clean's truth)
    ident = np.asarray(VIEW_PARAMS["clean"], dtype=np.float64)
    triv_rot = [rot_err_deg(ident[3:7], np.asarray(VIEW_PARAMS[c][3:7]))
                for c in POSE_CONDS]
    triv_trans = [float(np.linalg.norm(ident[:3] - np.asarray(VIEW_PARAMS[c][:3])))
                  for c in POSE_CONDS]
    trivial = {"always_identity_rot_err_deg": round(float(np.mean(triv_rot)), 3),
               "always_identity_trans_err_cm": round(float(np.mean(triv_trans)) * 100.0, 3)}
    print(f"[base  ] trivial always-identity: rot={trivial['always_identity_rot_err_deg']} deg "
          f"trans={trivial['always_identity_trans_err_cm']} cm (mean over the 7 pose conds)",
          flush=True)

    payload = {
        "vggt_cache": str(args.vggt_cache),
        "raymap_cache": str(args.raymap_cache),
        "split": {"n_train_ep": N_TRAIN_EP, "n_test_ep": 30 - N_TRAIN_EP,
                  "pose_conds": POSE_CONDS, "note": "positional per-condition split"},
        "step1_layer_probe": probe_rows,
        "step1_trivial_baseline": trivial,
        "step1_ruling": ruling,
        "step1b_light_invariance": light_rows,
        "step2_stability": stab_rows,
        "step2_separability": sep_rows,
        "step2_separability_baselines": sep_base,
        "step2_raymap": rm,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}", flush=True)
    print(f"LAYER RULING: layer={ruling['layer_ruling']} "
          f"rot_err={ruling['min_rot_err_deg']:.3f} "
          f"(rule: min rotation error; ties within 1 deg -> deepest layer)",
          flush=True)


if __name__ == "__main__":
    main()
