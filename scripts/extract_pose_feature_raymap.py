#!/usr/bin/env python3
"""Render Plücker ray maps per perturbed condition and encode them with DINOv3.

The ray map for a condition is rendered as if produced by the condition's
effective camera, but every ray is *expressed in the DEFAULT camera frame*
(transform T = T_def^-1 @ T_cond). A ray map expressed in the camera's own
frame carries no pose information (any camera looking at any scene yields the
same own-frame grid), so the default-frame representation is what turns pose
changes into image changes.

Per condition we build two 3-channel maps at 518x518:
  dir_map    ray directions d (unit vectors) in the default frame,
             mapped from [-1,1] to [0,1].
  moment_map Plücker moments m = d x o, where o is the ray origin (the
             condition camera center) in the default frame. Normalized by a
             single GLOBAL constant max|m| over all conditions (recorded in
             meta; preserves cross-condition translation magnitude), then
             clipped to [-1,1] and mapped to [0,1].

Maps are quantized to uint8 and encoded with the exact DINOv3 path used by
scripts/cache_domain_features.py (mlvla.meta.dinov3_encoder.encode_frames:
224x224 bilinear resize, ImageNet normalization, cls token). The per-condition
feature is the concatenation [dir_map cls (1024) || moment_map cls (1024)].

Camera conventions (verified against LIBERO-plus generators):
  - conditions.yaml quats are [w, x, y, z] cam-to-world (MuJoCo set_camera).
  - MuJoCo/OpenGL camera frame: +x right, +y up, camera looks along -z.
  - Pixel (u, v): u increases rightward, v (image row) increases downward, so
    the own-frame ray is normalize([(u-cx)/fx, (cy-v)/fy, -1]).
  - Intrinsics from MuJoCo: cam_fovy of the "agentview" camera in the scene
    XML loaded with mujoco.MjModel (45 deg for the tabletop scenes); for the
    square renders here the horizontal/vertical distinction is moot.

Output npz: keys "cond:<name>" -> float32 [2048]; key "meta" -> JSON string
with K, fov provenance, moment normalization constant, per-condition
effective==default verification and check results.
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

DEFAULT_CONDITIONS_YAML = Path("/data2/zhy/wizard_perturbed_lagfix_clean/conditions.yaml")
DEFAULT_OUTPUT = Path("/data2/zhy/meta_lora_offload/pose_cache/raymap_pose_cache.npz")
CAMERA_NAME = "agentview"
PIVOT = np.array([0.0, 0.0, 0.8])  # LIBERO-plus rotate_around_y / scale pivot


# --------------------------------------------------------------------------- #
# Camera math (pure numpy, no scipy dependency)
# --------------------------------------------------------------------------- #
def quat_wxyz_to_R(q) -> np.ndarray:
    """[w, x, y, z] quaternion -> 3x3 rotation matrix (cam-to-world)."""
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def R_to_angle_deg(R: np.ndarray) -> float:
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def rot_z(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(deg: float) -> np.ndarray:
    # LIBERO-plus rotate_around_y uses Rotation.from_rotvec(radians(-degrees)*[0,1,0]).
    t = math.radians(-deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


# --------------------------------------------------------------------------- #
# Intrinsics
# --------------------------------------------------------------------------- #
def read_fov_from_mujoco(scene_xml: Path):
    """Load the scene XML in MuJoCo and read cam_fovy of CAMERA_NAME.

    Returns (fov_deg, provenance string) or (None, error string).
    """
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(scene_xml))
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
        if cid < 0:
            return None, f"camera '{CAMERA_NAME}' not found in {scene_xml}"
        fov = float(model.cam_fovy[cid]) if hasattr(model, "cam_fovy") else float(model.cam_fov[cid])
        return fov, (
            f"mujoco {mujoco.__version__} MjModel.from_xml_path({scene_xml}).cam_fovy"
            f"[{CAMERA_NAME}]"
        )
    except Exception as exc:  # noqa: BLE001 - fall back loudly
        return None, f"mujoco read failed ({type(exc).__name__}: {exc})"


def build_k(fov_deg: float, res: int):
    """Normalized-focal K for a square res x res image (brief's Step 1 formula)."""
    f = 0.5 / math.tan(math.radians(fov_deg) / 2.0)  # normalized focal length
    fpx = f * res
    c = (res - 1) / 2.0
    K = np.array([[fpx, 0.0, c], [0.0, fpx, c], [0.0, 0.0, 1.0]])
    return K, fpx, c


# --------------------------------------------------------------------------- #
# Ray rendering
# --------------------------------------------------------------------------- #
def pixel_ray_grid_camera_frame(fpx: float, res: int) -> np.ndarray:
    """Own-frame unit ray directions, [H, W, 3] float64, row 0 = image top.

    OpenGL/MuJoCo camera (x right, y up, looks along -z):
      d(u, v) = normalize([(u - cx)/fx, (cy - v)/fy, -1]).
    """
    lin = np.arange(res, dtype=np.float64)
    uu, vv = np.meshgrid(lin, lin)  # xy indexing: [row(v), col(u)]
    cx = cy = (res - 1) / 2.0
    d = np.stack([(uu - cx) / fpx, (cy - vv) / fpx, -np.ones_like(uu)], axis=-1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def analytic_grid(fpx: float, res: int) -> np.ndarray:
    """Independent re-derivation of the own-frame grid (identity-check oracle).

    Deliberately constructed differently from pixel_ray_grid_camera_frame
    (mgrid + unnormalized [x, y, -1] then scaled) so a typo in one path breaks
    the check. The grid is u/v-symmetric, so the check is complemented by
    corner sign assertions in check_corner_signs().
    """
    vv, uu = np.mgrid[0:res, 0:res].astype(np.float64)
    cx = cy = (res - 1) / 2.0
    raw = np.stack([(uu - cx) / fpx, -(vv - cy) / fpx, -np.ones_like(uu)], axis=-1)
    return raw / np.linalg.norm(raw, axis=-1, keepdims=True)


def render_condition(default_cam, effective_cam, fpx: float, res: int):
    """Rays of the effective camera expressed in the DEFAULT camera frame.

    Returns (dir_map [H,W,3] unit vectors, origin o [3], moment [H,W,3]).
    """
    R_def = quat_wxyz_to_R(default_cam["quat"])
    t_def = np.asarray(default_cam["pos"], dtype=np.float64)
    R_cond = quat_wxyz_to_R(effective_cam["quat"])
    t_cond = np.asarray(effective_cam["pos"], dtype=np.float64)

    # T = T_def^-1 @ T_cond  (condition-camera frame -> default-camera frame)
    R_t = R_def.T @ R_cond
    o = R_def.T @ (t_cond - t_def)

    d_cam = pixel_ray_grid_camera_frame(fpx, res)  # [H,W,3]
    d = d_cam @ R_t.T  # rotate each ray into the default frame
    m = np.cross(d, np.broadcast_to(o, d.shape))  # Plücker moment
    return d, o, m


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def check_corner_signs(dir_map: np.ndarray):
    """Chirality guards the symmetric-grid identity check cannot catch.

    Top-left pixel (row 0, col 0): left of center => d_x < 0; top of image with
    y pointing up => d_y > 0; forward is -z => d_z < 0.
    """
    tl = dir_map[0, 0]
    br = dir_map[-1, -1]
    assert tl[0] < 0 and tl[1] > 0 and tl[2] < 0, f"top-left ray {tl}"
    assert br[0] > 0 and br[1] < 0 and br[2] < 0, f"bottom-right ray {br}"


def generator_crosscheck(conds: dict, default_cam: dict):
    """Re-derive v1/v2/v3 effective cameras from the default camera using the
    LIBERO-plus formulas (rotate_around_y -> rotate_around_z -> scale, pivots
    at z=0.8) and compare against the yaml values. Validates the [w,x,y,z]
    convention and the world-frame composition order against ground truth.
    """
    results = {}
    t_def = np.asarray(default_cam["pos"], dtype=np.float64)
    R_def = quat_wxyz_to_R(default_cam["quat"])
    for name, spec in (
        ("v1_azimuth30", dict(horizon=30, vertical=0, scale=1.0)),
        ("v2_azimuth60", dict(horizon=60, vertical=0, scale=1.0)),
        ("v3_elev15_zoom125", dict(horizon=0, vertical=15, scale=1.25)),
    ):
        view = conds[name]["view"]
        R, t = R_def, t_def
        if spec["vertical"] != 0:
            Rv, tv = rot_y(spec["vertical"]), PIVOT
            R = Rv @ R
            t = tv + Rv @ (t - tv)
        if spec["horizon"] != 0:
            R = rot_z(spec["horizon"]) @ R
            t = rot_z(spec["horizon"]) @ t
        if spec["scale"] != 1.0:
            t = PIVOT + spec["scale"] * (t - PIVOT)  # quat unchanged
        eff = conds[name]["view"]["effective_camera"]
        R_yaml = quat_wxyz_to_R(eff["quat"])
        t_yaml = np.asarray(eff["pos"], dtype=np.float64)
        R_err = float(np.abs(R - R_yaml).max())
        t_err = float(np.abs(t - t_yaml).max())
        # also confirm the view params recorded in yaml match the spec used here
        assert view["horizon"] == spec["horizon"] and view["vertical"] == spec["vertical"], name
        assert abs(view["scale_factor"] - spec["scale"]) < 1e-12, name
        assert R_err < 1e-8 and t_err < 1e-8, f"{name}: R_err={R_err}, t_err={t_err}"
        results[name] = {"rot_err": R_err, "pos_err": t_err}
    # combined conditions share the view component's effective camera
    for cname, vname in (("c1_v1l1", "v1_azimuth30"), ("c2_v2l2", "v2_azimuth60"), ("c3_v3l3", "v3_elev15_zoom125")):
        if cname not in conds:
            continue
        a = conds[cname]["view"]["effective_camera"]
        b = conds[vname]["view"]["effective_camera"]
        err = float(
            max(
                np.abs(np.asarray(a["pos"]) - np.asarray(b["pos"])).max(),
                np.abs(quat_wxyz_to_R(a["quat"]) - quat_wxyz_to_R(b["quat"])).max(),
            )
        )
        assert err < 1e-12, f"{cname} vs {vname} effective camera mismatch: {err}"
        results[f"{cname}=={vname}"] = {"err": err}
    return results


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--conditions-yaml", type=Path, default=DEFAULT_CONDITIONS_YAML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dinov3", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--dump-maps-dir", type=Path, default=None, help="optional dir for debug PNGs")
    parser.add_argument(
        "--fov",
        type=float,
        default=None,
        help="override fov (deg); use when running in an env without mujoco, "
        "carrying the value read authoritatively via the default path elsewhere",
    )
    parser.add_argument("--fov-source", type=str, default=None, help="provenance string for --fov")
    args = parser.parse_args()

    import yaml

    if args.dinov3 is None:
        args.dinov3 = __import__("mlvla.paths", fromlist=["PATHS"]).PATHS["dinov3_model"]

    record = yaml.safe_load(args.conditions_yaml.read_text())
    conds: dict = record["conditions"]

    # ---- shared default camera ------------------------------------------------
    default_cams = [
        c["view"]["default_camera"] for c in conds.values() if c.get("view")
    ]
    assert default_cams, "no view block with default_camera found"
    ref = default_cams[0]
    for dc in default_cams[1:]:
        assert np.allclose(dc["pos"], ref["pos"]) and np.allclose(dc["quat"], ref["quat"]), (
            "default cameras differ across conditions"
        )
    default_cam = ref

    # ---- Step 1: intrinsics from MuJoCo --------------------------------------
    scene_xml = next(
        (Path(c["light"]["scene_xml"]) for c in conds.values() if c.get("light")),
        None,
    )
    fov, fov_source = (None, "no light scene_xml in conditions.yaml")
    if args.fov is not None:
        fov, fov_source = args.fov, args.fov_source or "CLI override"
    elif scene_xml is not None and scene_xml.exists():
        fov, fov_source = read_fov_from_mujoco(scene_xml)
    fov_fallback = fov is None
    if fov_fallback:
        fov = 45.0
        fov_source = "FALLBACK (mujoco read failed): " + fov_source + " -> MuJoCo default 45 deg"
        print(f"WARNING: {fov_source}", flush=True)
    K, fpx, center = build_k(fov, args.resolution)
    print(f"fov={fov} deg ({'FALLBACK' if fov_fallback else 'from MuJoCo'}) fpx={fpx:.6f} c={center}", flush=True)

    # ---- convention cross-check vs LIBERO-plus generator ---------------------
    cross = generator_crosscheck(conds, default_cam)
    print(f"generator cross-check ok: {json.dumps(cross)}", flush=True)

    # ---- Step 2: render all conditions ---------------------------------------
    res = args.resolution
    analytic = analytic_grid(fpx, res)
    renders, cond_meta = {}, {}
    for name, spec in conds.items():
        if spec.get("view"):
            effective = spec["view"]["effective_camera"]
            eff_source = "yaml view.effective_camera"
            eff_is_default = False
        else:
            # type light / none: no view perturbation -> env leaves the agentview
            # camera at the default pose => effective == default.
            effective = default_cam
            eff_source = "default fallback (no view perturbation)"
            eff_is_default = True
        d, o, m = render_condition(default_cam, effective, fpx, res)
        renders[name] = (d, m)
        cond_meta[name] = {
            "type": spec["type"],
            "effective_source": eff_source,
            "effective_equals_default": eff_is_default,
            "T_translation_norm": float(np.linalg.norm(o)),
            "T_rotation_deg": R_to_angle_deg(quat_wxyz_to_R(effective["quat"]).T @ quat_wxyz_to_R(default_cam["quat"])),
        }
        print(
            f"{name}: |o|={cond_meta[name]['T_translation_norm']:.6f} "
            f"rot={cond_meta[name]['T_rotation_deg']:.3f} deg eff_is_default={eff_is_default}",
            flush=True,
        )

    # identity check: every condition with effective == default must reproduce
    # the analytic own-frame grid exactly; its moment must vanish.
    identity = {}
    for name, cm in cond_meta.items():
        if not cm["effective_equals_default"]:
            continue
        d, m = renders[name]
        check_corner_signs(d)
        dir_err = float(np.abs(d - analytic).max())
        mom_err = float(np.abs(m).max())
        assert dir_err < 1e-5, f"{name}: identity dir err {dir_err} >= 1e-5"
        assert mom_err < 1e-9, f"{name}: identity moment err {mom_err} >= 1e-9"
        identity[name] = {"max_dir_err": dir_err, "max_moment_err": mom_err}
    assert "clean" in identity, "clean condition missing from identity check"
    print(f"identity check ok: {json.dumps(identity)}", flush=True)

    # ---- moment normalization (single global constant) ------------------------
    global_m = max(float(np.abs(m).max()) for _, m in renders.values())
    assert global_m > 0, "global moment max is zero (no view-perturbed condition?)"

    def to_u8(arr01: np.ndarray) -> np.ndarray:
        return np.clip(np.round(arr01 * 255.0), 0, 255).astype(np.uint8)

    dir_imgs, mom_imgs = {}, {}
    for name, (d, m) in renders.items():
        dir01 = (d + 1.0) / 2.0
        mom01 = (np.clip(m / global_m, -1.0, 1.0) + 1.0) / 2.0
        dir_imgs[name] = to_u8(dir01)
        mom_imgs[name] = to_u8(mom01)
        assert np.isfinite(d).all() and np.isfinite(m).all(), name

    if args.dump_maps_dir:
        from PIL import Image

        args.dump_maps_dir.mkdir(parents=True, exist_ok=True)
        for name in renders:
            Image.fromarray(dir_imgs[name]).save(args.dump_maps_dir / f"{name}_dir.png")
            Image.fromarray(mom_imgs[name]).save(args.dump_maps_dir / f"{name}_moment.png")
        print(f"dumped debug maps to {args.dump_maps_dir}", flush=True)

    # ---- Step 3: DINOv3 encoding (identical path to cache_domain_features) ---
    from mlvla.meta.dinov3_encoder import encode_frames, load_dinov3

    model, hidden, num_register, _ = load_dinov3(args.dinov3, device=args.device)
    names = sorted(renders)
    frames = [dir_imgs[n] for n in names] + [mom_imgs[n] for n in names]
    out = encode_frames(
        model, frames, device=args.device, num_register_tokens=num_register, hidden_size=hidden
    )
    cls = out["cls"]  # [2N, 1024] float32
    feats = {n: np.concatenate([cls[i], cls[len(names) + i]]).astype(np.float32) for i, n in enumerate(names)}
    for n, f in feats.items():
        assert f.shape == (2 * hidden,) and np.isfinite(f).all(), n

    # ---- Step 4: checks -------------------------------------------------------
    def cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    cos_clean_v1 = cos(feats["clean"], feats["v1_azimuth30"])
    cos_v1_v2 = cos(feats["v1_azimuth30"], feats["v2_azimuth60"])
    i = {n: k for k, n in enumerate(names)}

    def chan_cos(chan: str, a: str, b: str) -> float:
        off = 0 if chan == "dir" else len(names)
        return cos(cls[off + i[a]], cls[off + i[b]])

    dir_cos = {
        "clean~v1": chan_cos("dir", "clean", "v1_azimuth30"),
        "clean~v2": chan_cos("dir", "clean", "v2_azimuth60"),
        "v1~v2": chan_cos("dir", "v1_azimuth30", "v2_azimuth60"),
    }
    mom_cos = {
        "clean~v1": chan_cos("mom", "clean", "v1_azimuth30"),
        "v1~v2": chan_cos("mom", "v1_azimuth30", "v2_azimuth60"),
    }
    print(f"concat: cosine(clean,v1)={cos_clean_v1:.6f}  cosine(v1,v2)={cos_v1_v2:.6f}", flush=True)
    print(f"  dir-channel: {dir_cos}", flush=True)
    print(f"  moment-channel: {mom_cos}", flush=True)
    # Brief's monotonicity spot check ("光线图" = the ray/dir map): bigger
    # viewpoint change -> farther DINO features.
    monotonic_ok = dir_cos["v1~v2"] < dir_cos["clean~v1"]
    assert monotonic_ok, (
        f"dir-map monotonicity check failed: cos(v1,v2)={dir_cos['v1~v2']} "
        f">= cos(clean,v1)={dir_cos['clean~v1']}"
    )
    # The moment channel is exactly flat (zero moment) for every
    # effective==default condition, so clean-vs-v1 there measures "flat vs
    # structured" (a category discontinuity), not pose magnitude; it is
    # recorded, not asserted.
    moment_note = (
        "moment-channel clean~v1 compares a by-construction flat map (zero "
        "moment: identical camera) against a structured map, so its cosine is "
        "a flat-vs-structured discontinuity rather than pose distance"
    )

    # full pairwise matrix for the report
    pairwise = {
        a: {b: round(cos(feats[a], feats[b]), 6) for b in names} for a in names
    }

    # ---- Step 3b: save --------------------------------------------------------
    meta = {
        "script": str(Path(__file__).resolve()),
        "conditions_yaml": str(args.conditions_yaml),
        "render": {
            "resolution": res,
            "camera": CAMERA_NAME,
            "fov_deg": fov,
            "fov_is_fallback": fov_fallback,
            "fov_source": fov_source,
            "K": K.tolist(),
            "focal_px": fpx,
            "principal_point": [center, center],
            "principal_point_convention": (
                f"(res-1)/2 = {center} (pixel-center convention; the task brief "
                "wrote 259.5 = res/2 — a half-pixel principal-point offset that is "
                "shared across all conditions and so does not affect condition "
                "separability)"
            ),
            "quat_convention": "[w,x,y,z] cam-to-world (MuJoCo set_camera)",
            "ray_convention": "OpenGL/MuJoCo frame: x right, y up, -z forward; row 0 = image top; d=normalize([(u-cx)/fx,(cy-v)/fy,-1])",
            "frame": "rays expressed in DEFAULT camera frame via T = T_def^-1 @ T_cond; moment = cross(d, o)",
        },
        "moment_normalization": {
            "type": "global_constant_all_conditions",
            "max_abs_moment": global_m,
            "mapping": "clip(m/max, -1, 1) -> [0,1] -> uint8",
        },
        "features": {
            "layout": "dir_map cls (1024) || moment_map cls (1024)",
            "dim": 2 * hidden,
            "encoder": args.dinov3,
            "encoder_path": "mlvla.meta.dinov3_encoder.encode_frames (identical to scripts/cache_domain_features.py: 224 bilinear resize, ImageNet norm, cls token, bfloat16)",
        },
        "conditions": cond_meta,
        "effective_equals_default_verified": {
            n: cm["effective_equals_default"] for n, cm in cond_meta.items()
        },
        "checks": {
            "identity_max_dir_err": {n: v["max_dir_err"] for n, v in identity.items()},
            "identity_max_moment_err": {n: v["max_moment_err"] for n, v in identity.items()},
            "generator_crosscheck": cross,
            "cosine_clean_v1_concat": cos_clean_v1,
            "cosine_v1_v2_concat": cos_v1_v2,
            "monotonicity_ok": monotonic_ok,
            "monotonicity_channel": "dir_map_cls (the ray map, per brief '光线图')",
            "dir_channel_cosine": dir_cos,
            "moment_channel_cosine": mom_cos,
            "moment_channel_note": moment_note,
            "all_finite": True,
            "n_conditions": len(names),
        },
        "pairwise_cosine": pairwise,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"cond:{n}": feats[n] for n in names}
    payload["meta"] = np.asarray(json.dumps(meta, indent=2))
    np.savez(args.output, **payload)

    # reload and verify schema
    got = np.load(args.output, allow_pickle=False)
    assert set(got.files) == {f"cond:{n}" for n in names} | {"meta"}, got.files
    for n in names:
        v = got[f"cond:{n}"]
        assert v.shape == (2 * hidden,) and v.dtype == np.float32 and np.isfinite(v).all(), n
    json.loads(str(got["meta"]))
    print(f"wrote {args.output} with {len(names)} conditions; schema verified on reload", flush=True)
    print(f"torch device used: {args.device} (CUDA_VISIBLE_DEVICES pinning is external)", flush=True)


if __name__ == "__main__":
    main()
