"""
Loss functions for T2L-style meta-network training.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss: MSE between generated LoRA and oracle expert LoRA.

    Supports optional z-score normalization to handle scale differences across modules.

    Args:
        pred_z_score: If True, normalize predictions and targets using pre-computed mean/std
        mean_recon_target: Dict mapping module_key -> {"A": [L, r, in], "B": [L, out, r]}
        std_recon_target: Dict mapping module_key -> {"A": [L, r, in], "B": [L, out, r]}
        lambda_recon: Weight for reconstruction loss
    """

    def __init__(
        self,
        pred_z_score: bool = False,
        mean_recon_target: dict[str, dict[str, torch.Tensor]] | None = None,
        std_recon_target: dict[str, dict[str, torch.Tensor]] | None = None,
        lambda_recon: float = 1.0,
    ):
        super().__init__()
        self.pred_z_score = pred_z_score
        self.lambda_recon = lambda_recon

        # Register mean/std as buffers (non-trainable)
        if pred_z_score:
            assert mean_recon_target is not None and std_recon_target is not None
            self.register_buffer("mean_A", mean_recon_target["A"])
            self.register_buffer("mean_B", mean_recon_target["B"])
            self.register_buffer("std_A", std_recon_target["A"])
            self.register_buffer("std_B", std_recon_target["B"])
        else:
            self.mean_A = self.mean_B = self.std_A = self.std_B = None

    def forward(
        self,
        pred_A: torch.Tensor,
        pred_B: torch.Tensor,
        target_A: torch.Tensor,
        target_B: torch.Tensor,
        module_key: str | None = None,
        layer_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute reconstruction loss.

        Args:
            pred_A: [B, r, in] predicted LoRA A
            pred_B: [B, out, r] predicted LoRA B
            target_A: [B, r, in] or [r, in] oracle LoRA A
            target_B: [B, out, r] or [out, r] oracle LoRA B
            module_key: Module group key (for z-score lookup)
            layer_indices: [B] layer indices (for z-score lookup)

        Returns:
            Dict with keys: "loss", "l1_A", "l1_B", "unnorm_err"
        """
        # Expand target if needed
        if target_A.dim() == 2:
            target_A = target_A.unsqueeze(0).expand_as(pred_A)
        if target_B.dim() == 2:
            target_B = target_B.unsqueeze(0).expand_as(pred_B)

        if self.pred_z_score and module_key is not None and layer_indices is not None:
            # Z-score normalization
            mean_A = self.mean_A[module_key][layer_indices]
            mean_B = self.mean_B[module_key][layer_indices]
            std_A = self.std_A[module_key][layer_indices]
            std_B = self.std_B[module_key][layer_indices]

            # Normalize predictions
            pred_A_norm = (pred_A - mean_A) / (std_A + 1e-10)
            pred_B_norm = (pred_B - mean_B) / (std_B + 1e-10)

            # Normalize targets
            target_A_norm = (target_A - mean_A) / (std_A + 1e-10)
            target_B_norm = (target_B - mean_B) / (std_B + 1e-10)

            # Compute loss on normalized values
            l1_A = F.l1_loss(pred_A_norm, target_A_norm)
            l1_B = F.l1_loss(pred_B_norm, target_B_norm)

            # Compute unnormalized error for logging
            with torch.no_grad():
                pred_A_unnorm = pred_A.detach() * (std_A + 1e-10) + mean_A
                pred_B_unnorm = pred_B.detach() * (std_B + 1e-10) + mean_B
                unnorm_err = (F.l1_loss(pred_A_unnorm, target_A) + F.l1_loss(pred_B_unnorm, target_B)).item() / 2
        else:
            # Direct MSE loss
            l1_A = F.mse_loss(pred_A, target_A)
            l1_B = F.mse_loss(pred_B, target_B)
            unnorm_err = (l1_A.item() + l1_B.item()) / 2

        loss = (l1_A + l1_B) / 2 * self.lambda_recon

        return {
            "loss": loss,
            "l1_A": l1_A,
            "l1_B": l1_B,
            "unnorm_err": torch.tensor(unnorm_err),
        }


class CombinedLoss(nn.Module):
    """
    Combined loss for end-to-end training: action loss + reconstruction loss.

    Args:
        lambda_recon: Weight for reconstruction loss
        lambda_scale: Weight for scale regularization (optional)
    """

    def __init__(
        self,
        lambda_recon: float = 0.1,
        lambda_scale: float = 0.0,
    ):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_scale = lambda_scale

    def forward(
        self,
        action_loss: torch.Tensor,
        recon_loss: torch.Tensor,
        pred_std: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            action_loss: Action prediction loss (from JAX backend)
            recon_loss: Reconstruction loss (from ReconstructionLoss)
            pred_std: Predicted LoRA standard deviation (for scale regularization)

        Returns:
            Dict with keys: "total_loss", "action_loss", "recon_loss", "scale_loss"
        """
        total = action_loss

        if self.lambda_recon > 0:
            total = total + self.lambda_recon * recon_loss

        scale_loss = torch.zeros(1, device=action_loss.device)
        if self.lambda_scale > 0 and pred_std is not None:
            # Penalize extreme scales
            scale_loss = (pred_std ** 2).mean()
            total = total + self.lambda_scale * scale_loss

        return {
            "total_loss": total,
            "action_loss": action_loss,
            "recon_loss": recon_loss,
            "scale_loss": scale_loss,
        }


def compute_cosine_similarity(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between predicted and target LoRA weights.

    Args:
        pred: [B, ...] predicted weights
        target: [B, ...] or [...] target weights

    Returns:
        [B] cosine similarity per sample
    """
    if target.dim() < pred.dim():
        target = target.unsqueeze(0).expand_as(pred)

    pred_flat = pred.flatten(start_dim=1)
    target_flat = target.flatten(start_dim=1)

    cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=-1)
    return cos_sim
