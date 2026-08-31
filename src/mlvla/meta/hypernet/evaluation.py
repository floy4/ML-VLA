from __future__ import annotations

from collections.abc import Mapping

import torch

from .losses import delta_metrics


@torch.inference_mode()
def evaluate_direct_ab(
    predictions: Mapping[str, Mapping[str, torch.Tensor]],
    targets: Mapping[str, Mapping[str, torch.Tensor]],
    scales: Mapping[str, Mapping[str, float]],
) -> list[dict[str, float | str]]:
    rows = []
    for key, factors in predictions.items():
        pred_A = factors["A"] * scales[key]["A"]
        pred_B = factors["B"] * scales[key]["B"]
        target_A, target_B = targets[key]["A"], targets[key]["B"]
        if target_A.ndim == 2:
            target_A = target_A.expand(pred_A.shape[0], -1, -1)
            target_B = target_B.expand(pred_B.shape[0], -1, -1)
        relative, cosine = delta_metrics(pred_A, pred_B, target_A, target_B)
        A_error = torch.linalg.vector_norm(pred_A - target_A, dim=(-2, -1)) / torch.linalg.vector_norm(
            target_A, dim=(-2, -1)
        ).clamp_min(1e-12)
        B_error = torch.linalg.vector_norm(pred_B - target_B, dim=(-2, -1)) / torch.linalg.vector_norm(
            target_B, dim=(-2, -1)
        ).clamp_min(1e-12)
        rows.append(
            {
                "key": key,
                "relative_A_error": float(A_error.mean()),
                "relative_B_error": float(B_error.mean()),
                "relative_delta_error": float(relative.mean()),
                "delta_cosine": float(cosine.mean()),
            }
        )
    return rows
