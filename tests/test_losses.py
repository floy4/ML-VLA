from __future__ import annotations

import torch

from mlvla.meta.hypernet.losses import delta_metrics


def test_delta_metrics_match_dense() -> None:
    torch.manual_seed(0)
    pred_A = torch.randn(2, 3, 7)
    pred_B = torch.randn(2, 5, 3)
    target_A = torch.randn(2, 3, 7)
    target_B = torch.randn(2, 5, 3)
    relative, cosine = delta_metrics(pred_A, pred_B, target_A, target_B)
    pred = pred_B @ pred_A
    target = target_B @ target_A
    dense_relative = torch.linalg.vector_norm(pred - target, dim=(-2, -1)) / torch.linalg.vector_norm(
        target, dim=(-2, -1)
    )
    dense_cosine = torch.nn.functional.cosine_similarity(pred.flatten(1), target.flatten(1))
    assert torch.allclose(relative, dense_relative, atol=1e-5)
    assert torch.allclose(cosine, dense_cosine, atol=1e-5)
