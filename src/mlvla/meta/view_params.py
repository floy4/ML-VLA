"""Camera-pose view parameters per perturbation condition (metadata lookup, no regression).

Vectors are [dpos(3), dquat(4)] relative to the default tabletop camera, from the
effective_camera records in /data2/zhy/wizard_perturbed/conditions.yaml; dquat is
q_default^-1 (x) q_cond with the w >= 0 sign convention. Cross-checked by tests.

The ``pose_feature`` scheme layer on top lets the conditioning vector come from
learned pose caches instead of the 7-dim extrinsics table:

- ``view7``  (default, legacy): VIEW_PARAMS table — bit-identical to pre-scheme code
- ``zero``:  7 zeros (pose-ablation control; same cond width as view7)
- ``vggt``:  VGGT CAMERA-token cache, ``blob["cond:<cond>"][layer].mean(0)`` [2048];
             layer ruling: 23 (task-5 brief, from the task-3 probe)
- ``raymap``: DINOv3-encoded Plücker ray-map cache, ``blob["cond:<cond>"]`` [2048]

Both caches also carry entries for lighting/noise conditions (camera unmoved, so
the feature is naturally ~clean — the same identity semantics as VIEW_PARAMS).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

VIEW_PARAMS: dict[str, list[float]] = {
    "v1_azimuth30": [-0.088237, 0.329307, 0.0, 0.965926, 0.0, 0.201361, 0.162607],
    "v2_azimuth60": [-0.329307, 0.570376, 0.0, 0.866025, 0.0, 0.388999, 0.314133],
    "v3_elev15_zoom125": [-0.125566, 0.0, 0.38115, 0.991445, -0.130526, 0.0, 0.0],
    "l1_warm_dim": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "l2_cool_bright": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "l3_directional_low": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "c1_v1l1": [-0.088237, 0.329307, 0.0, 0.965926, 0.0, 0.201361, 0.162607],
    "c2_v2l2": [-0.329307, 0.570376, 0.0, 0.866025, 0.0, 0.388999, 0.314133],
    "c3_v3l3": [-0.125566, 0.0, 0.38115, 0.991445, -0.130526, 0.0, 0.0],
    "clean": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    # Level multitask domains (max-CHR). Camera vectors are the mean per-episode
    # relative pose pooled over the level's episodes across all 4 suites
    # (per-episode extrinsics from libero_plus_camera_annotated episodes
    # parquet; per-suite default pose = mean of camera_is_identity episodes;
    # same dquat convention as above). Non-camera perturbation families get the
    # identity vector, matching the lagfix lighting/noise conditions.
    "camera_L1": [0.010432, -0.000874, 0.016641, 0.999992, -0.0001, 0.00318, 0.002522],
    "camera_L2": [0.114822, 0.000868, 0.141893, 0.999328, -0.036538, 0.001796, 0.002154],
    "camera_L3": [-0.152705, 0.010396, 0.156541, 0.99796, -0.063544, 0.00483, 0.003791],
    "lighting_L1": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "lighting_L2": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "lighting_L3": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "noise_L1": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "noise_L2": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "noise_L3": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "texture_L1": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "texture_L2": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "texture_L3": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    # add_object level domains (Qwen3.5-9B delta banding): no view change,
    # identity like the other non-camera families.
    "add_object_L1": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "add_object_L2": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "add_object_L3": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
}
VIEW_DIM = 7


def view_vector(condition: str) -> list[float]:
    try:
        return VIEW_PARAMS[condition]
    except KeyError:
        raise ValueError(f"unknown condition {condition!r}; known: {sorted(VIEW_PARAMS)}") from None


def view_vector_for_domain(domain: str) -> list[float]:
    return view_vector(domain.split("__", 1)[0])


def append_view_params(features: torch.Tensor, domain: str,
                       pose_feature: dict | None = None) -> torch.Tensor:
    """Concatenate the domain's pose vector onto per-episode features [..., feature_dim].

    ``pose_feature`` is the config block ``{scheme, cache, layer}``; when absent
    (or None) this is the legacy view7 path, bit-identical to the pre-scheme
    implementation.
    """
    pose = pose_vector_from_config(domain, pose_feature)
    vec = pose.to(dtype=features.dtype, device=features.device)
    return torch.cat([features, vec.expand(*features.shape[:-1], vec.shape[-1])], dim=-1)


# ---------------------------------------------------------------------------
# pose_feature schemes (view7 / zero / vggt / raymap)
# ---------------------------------------------------------------------------

VGGT_POSE_DIM = 2048  # CAMERA token = frame-token ‖ global-token concat
DEFAULT_VGGT_LAYER = 23  # task-3 layer ruling


def pose_dim_for_scheme(scheme: str) -> int:
    """Pose-vector width per scheme: view7/zero -> 7, vggt/raymap -> 2048."""
    if scheme in ("view7", "zero"):
        return VIEW_DIM
    if scheme in ("vggt", "raymap"):
        return VGGT_POSE_DIM
    raise ValueError(f"unknown pose scheme {scheme!r}; known: view7, zero, vggt, raymap")


@lru_cache(maxsize=None)
def _load_pose_cache(cache: str) -> dict[str, np.ndarray]:
    """Process-level cache of one pose npz blob (keyed by resolved path string).

    npz members are materialized eagerly so the returned dict outlives the file
    handle. The blobs are plain arrays (no pickle).
    """
    with np.load(cache) as z:
        return {k: z[k] for k in z.files}


def pose_vector_for_domain(domain: str, scheme: str = "view7",
                           cache: str | Path | None = None,
                           layer: int = DEFAULT_VGGT_LAYER) -> torch.Tensor:
    """[pose_dim] float32 pose vector for a domain, under the given scheme.

    ``domain`` may carry a ``__<hash>`` suffix (stripped, as in
    view_vector_for_domain). view7 is the legacy table lookup; zero returns 7
    zeros; vggt/raymap read ``cond:<condition>`` from the npz cache
    (vggt takes the mean over the cache's episode axis at ``layer``).
    """
    cond = domain.split("__", 1)[0]
    if scheme == "view7":
        return torch.tensor(view_vector(cond), dtype=torch.float32)
    if scheme == "zero":
        return torch.zeros(VIEW_DIM, dtype=torch.float32)
    if cache is None:
        raise ValueError(f"pose scheme {scheme!r} requires a cache path")
    blob = _load_pose_cache(str(Path(cache).resolve()))
    key = f"cond:{cond}"
    if key not in blob:
        raise KeyError(
            f"{key} missing from pose cache {cache}; available: "
            f"{sorted(k for k in blob if k.startswith('cond:'))}")
    if scheme == "vggt":
        arr = blob[key][int(layer)].mean(0)          # [n_episodes, 2048] -> [2048]
    elif scheme == "raymap":
        arr = blob[key]                              # [2048]
    else:
        raise ValueError(f"unknown pose scheme {scheme!r}; known: view7, zero, vggt, raymap")
    return torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float32)


def pose_vector_from_config(domain: str, pose_feature: dict | None) -> torch.Tensor:
    """Dispatch on the config block ``pose_feature: {scheme, cache, layer}``.

    None/empty block -> legacy view7 (regression-protection invariant: output is
    bit-identical to ``view_vector_for_domain``).
    """
    if not pose_feature:
        return torch.tensor(view_vector_for_domain(domain), dtype=torch.float32)
    return pose_vector_for_domain(
        domain,
        scheme=pose_feature.get("scheme", "view7"),
        cache=pose_feature.get("cache"),
        layer=int(pose_feature.get("layer", DEFAULT_VGGT_LAYER)),
    )
