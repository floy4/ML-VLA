"""FiLM-conditioned DirectAB hypernetwork: DINO path modulated by a dedicated camera-pose encoder."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from mlvla.meta.hypernet.direct_ab import ModuleABHead


class FiLMABHyperNetwork(nn.Module):
    """DINO encoder + independent pose encoder fused via FiLM, then shared trunk + A/B heads.

    Identity-at-init: gamma head is zero-weight / one-bias, beta head zero/zero, so the
    untrained model computes exactly the pose-free function.
    """

    def __init__(
        self,
        dino_dim: int,
        view_dim: int,
        module_shapes: Mapping[str, Mapping[str, tuple[int, int]]],
        condition_hidden_dim: int = 256,
        hidden_dim: int = 512,
        head_bottleneck: int = 32,
    ) -> None:
        super().__init__()
        self.module_shapes = {
            key: {factor: tuple(shape) for factor, shape in shapes.items()}
            for key, shapes in module_shapes.items()
        }
        self.dino_encoder = nn.Sequential(
            nn.Linear(dino_dim, condition_hidden_dim),
            nn.GELU(),
            nn.Linear(condition_hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.pose_encoder = nn.Sequential(nn.Linear(view_dim, hidden_dim), nn.GELU())
        self.gamma_out = nn.Linear(hidden_dim, hidden_dim)
        self.beta_out = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.gamma_out.weight)
        nn.init.ones_(self.gamma_out.bias)
        nn.init.zeros_(self.beta_out.weight)
        nn.init.zeros_(self.beta_out.bias)
        self.trunk = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.heads = nn.ModuleDict(
            {
                self.safe_name(key): ModuleABHead(
                    hidden_dim,
                    self.module_shapes[key]["A"],
                    self.module_shapes[key]["B"],
                    head_bottleneck,
                )
                for key in self.module_shapes
            }
        )

    @staticmethod
    def safe_name(key: str) -> str:
        return key.replace(".", "__")

    def forward(self, dino: torch.Tensor, view: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        if dino.ndim == 3:
            dino = dino.mean(dim=1)
        phi = self.dino_encoder(dino)
        z = self.pose_encoder(view)
        hidden = self.trunk(self.gamma_out(z) * phi + self.beta_out(z))
        return {key: self.heads[self.safe_name(key)](hidden) for key in self.module_shapes}
