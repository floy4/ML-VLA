"""Camera-pose view parameters per perturbation condition (metadata lookup, no regression).

Vectors are [dpos(3), dquat(4)] relative to the default tabletop camera, from the
effective_camera records in /data2/zhy/wizard_perturbed/conditions.yaml; dquat is
q_default^-1 (x) q_cond with the w >= 0 sign convention. Cross-checked by tests.
"""
from __future__ import annotations

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
}
VIEW_DIM = 7


def view_vector(condition: str) -> list[float]:
    try:
        return VIEW_PARAMS[condition]
    except KeyError:
        raise ValueError(f"unknown condition {condition!r}; known: {sorted(VIEW_PARAMS)}") from None


def view_vector_for_domain(domain: str) -> list[float]:
    return view_vector(domain.split("__", 1)[0])


def append_view_params(features: torch.Tensor, domain: str) -> torch.Tensor:
    """Concatenate the domain's view vector onto per-episode features [..., feature_dim]."""
    vec = torch.tensor(view_vector_for_domain(domain), dtype=features.dtype, device=features.device)
    return torch.cat([features, vec.expand(*features.shape[:-1], VIEW_DIM)], dim=-1)
