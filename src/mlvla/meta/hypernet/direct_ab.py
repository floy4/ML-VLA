"""Shared decoder for one-hot V0 and DINO V1 direct A/B reconstruction."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


class ModuleABHead(nn.Module):
    def __init__(
        self, hidden_dim: int, A_shape: tuple[int, int], B_shape: tuple[int, int], bottleneck: int
    ) -> None:
        super().__init__()
        self.A_shape = A_shape
        self.B_shape = B_shape
        self.adapter = nn.Sequential(nn.Linear(hidden_dim, bottleneck), nn.GELU())
        self.A_out = nn.Linear(bottleneck, A_shape[0] * A_shape[1])
        self.B_out = nn.Linear(bottleneck, B_shape[0] * B_shape[1])

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        value = self.adapter(hidden)
        return {
            "A": self.A_out(value).reshape(hidden.shape[0], *self.A_shape),
            "B": self.B_out(value).reshape(hidden.shape[0], *self.B_shape),
        }


class DirectABHyperNetwork(nn.Module):
    """Condition encoder + shared trunk + module-specific direct A/B heads."""

    def __init__(
        self,
        condition_dim: int,
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
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, condition_hidden_dim),
            nn.GELU(),
            nn.Linear(condition_hidden_dim, hidden_dim),
            nn.GELU(),
        )
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

    def forward(self, condition: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        if condition.ndim == 3:
            condition = condition.mean(dim=1)
        hidden = self.trunk(self.condition_encoder(condition))
        return {key: self.heads[self.safe_name(key)](hidden) for key in self.module_shapes}
