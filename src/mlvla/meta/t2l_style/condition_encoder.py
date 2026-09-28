"""
Two-branch condition encoder for T2L-style meta-network.
Encodes semantic (DINO) and geometric (pose/extrinsics) evidence into task embedding.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoBranchConditionEncoder(nn.Module):
    """
    Encodes evidence into task embedding for T2L-style generator.

    Architecture:
    - Semantic branch: DINO features (1024-d) → MLP → task_emb_size
    - Geometric branch: pose/extrinsics (7-d or 2048-d) → MLP → task_emb_size
    - Fusion: LayerNorm + Addition (enables zero-shot composition)

    Inputs:
        semantic: [B, K, 4, 1024] DINO patch-mean features (K evidence episodes, 4 frames each)
                  or [B, K, 1024] if pre-aggregated over frames
        geometric: [B, K, D_geo] pose vectors (D_geo=7 for view7, 2048 for vggt/raymap)
                  or [B, D_geo] if single pose per sample

    Output:
        [B, task_emb_size] condition vector
    """

    def __init__(
        self,
        dino_dim: int = 1024,
        geo_dim: int = 7,
        hidden_dim: int = 256,
        task_emb_size: int = 256,
        fusion: str = "additive",  # "additive" or "concat"
        dropout: float = 0.05,
    ):
        super().__init__()
        self.task_emb_size = task_emb_size
        self.fusion = fusion

        # Semantic branch: project DINO features to full task_emb_size
        self.semantic_proj = nn.Sequential(
            nn.Linear(dino_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, task_emb_size),
        )

        # Geometric branch: project pose to full task_emb_size
        self.geometric_proj = nn.Sequential(
            nn.Linear(geo_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, task_emb_size),
        )

        # Final fusion layer norm
        self.output_ln = nn.LayerNorm(task_emb_size)

        # Optional fusion MLP for concat mode
        if fusion == "concat":
            self.fusion_mlp = nn.Sequential(
                nn.Linear(task_emb_size * 2, task_emb_size),
                nn.GELU(),
                nn.LayerNorm(task_emb_size),
            )
        else:
            self.fusion_mlp = None

    def forward(
        self,
        semantic: torch.Tensor,
        geometric: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            semantic: [B, K, 4, D] or [B, K, D] or [B, D] DINO features
            geometric: [B, K, D_geo] or [B, D_geo] pose vectors

        Returns:
            [B, task_emb_size] condition embedding
        """
        # Aggregate over evidence frames
        if semantic.dim() == 4:  # [B, K, 4, D]
            semantic = semantic.mean(dim=[1, 2])
        elif semantic.dim() == 3:
            semantic = semantic.mean(dim=1)

        if geometric.dim() == 3:
            geometric = geometric.mean(dim=1)

        # Project each branch to full task_emb_size
        e_sem = self.semantic_proj(semantic)  # [B, task_emb_size]
        e_geo = self.geometric_proj(geometric)  # [B, task_emb_size]

        # Fusion
        if self.fusion == "additive":
            # Add with layer norm enables compositionality
            e_sem = F.layer_norm(e_sem, e_sem.shape[-1:])
            e_geo = F.layer_norm(e_geo, e_geo.shape[-1:])
            condition = e_sem + e_geo
        else:  # concat
            condition = torch.cat([e_sem, e_geo], dim=-1)
            condition = self.fusion_mlp(condition)

        return self.output_ln(condition)