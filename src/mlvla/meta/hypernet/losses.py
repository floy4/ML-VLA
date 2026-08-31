"""Factor and full-update losses that avoid materializing large ΔW matrices."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.nn import functional as F


def _delta_squared_norm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    aa = A @ A.transpose(-1, -2)
    bb = B.transpose(-1, -2) @ B
    return (aa * bb.transpose(-1, -2)).sum(dim=(-2, -1)).clamp_min(0)


def _delta_inner(
    A1: torch.Tensor, B1: torch.Tensor, A2: torch.Tensor, B2: torch.Tensor
) -> torch.Tensor:
    return torch.einsum(
        "...ij,...ji->...",
        B1.transpose(-1, -2) @ B2,
        A2 @ A1.transpose(-1, -2),
    )


def delta_metrics(
    pred_A: torch.Tensor,
    pred_B: torch.Tensor,
    target_A: torch.Tensor,
    target_B: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred2 = _delta_squared_norm(pred_A, pred_B)
    target2 = _delta_squared_norm(target_A, target_B)
    cross = _delta_inner(pred_A, pred_B, target_A, target_B)
    relative = ((pred2 + target2 - 2 * cross).clamp_min(0) / target2.clamp_min(eps)).sqrt()
    cosine = cross / (pred2 * target2).clamp_min(eps).sqrt()
    return relative, cosine


def reconstruction_loss(
    predictions: Mapping[str, Mapping[str, torch.Tensor]],
    targets: Mapping[str, Mapping[str, torch.Tensor]],
    scales: Mapping[str, Mapping[str, float]],
    delta_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ab_loss = next(iter(predictions.values()))["A"].new_zeros(())
    delta_loss = ab_loss.clone()
    for key, factors in predictions.items():
        ab_loss = ab_loss + F.mse_loss(factors["A"], targets[key]["A"] / scales[key]["A"])
        ab_loss = ab_loss + F.mse_loss(factors["B"], targets[key]["B"] / scales[key]["B"])
        if delta_weight:
            pred_A = factors["A"] * scales[key]["A"]
            pred_B = factors["B"] * scales[key]["B"]
            relative, _ = delta_metrics(pred_A, pred_B, targets[key]["A"], targets[key]["B"])
            delta_loss = delta_loss + relative.square().mean()
    ab_loss = ab_loss / (2 * len(predictions))
    if delta_weight:
        delta_loss = delta_loss / len(predictions)
    total = ab_loss + delta_weight * delta_loss
    return total, {"ab": ab_loss.detach(), "delta": delta_loss.detach()}
