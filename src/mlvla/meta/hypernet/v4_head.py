"""v4 head: T2L-style direct output heads (no dictionary factorization).

Same identity-conditioned trunk as v3, but the per-group coefficient heads +
shared dictionaries (a bilinear C·D factorization that SGD only fit to cosine
0.584 despite proven span capacity) are replaced by direct linear heads that
output each shape group's full A/B factors. The A/B switch vector is dropped
because A and B now have separate heads.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


def group_key(shapes: Mapping[str, tuple[int, int]]) -> str:
    (r, in_dim), (out_dim, _) = shapes["A"], shapes["B"]
    return f"A{r}x{in_dim}_B{out_dim}x{r}"


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


class DirectHeadDecoder(nn.Module):
    """Identity-conditioned trunk + per-shape-group direct linear output heads."""

    def __init__(
        self,
        module_shapes: Mapping[str, Mapping[str, tuple[int, int]]],
        task_dim: int = 512,
        hidden_dim: int = 512,
        id_dim: int = 256,
        trunk_layers: int = 6,
        trunk_hidden: int = 1024,
    ) -> None:
        super().__init__()
        self.module_shapes = {
            key: {factor: tuple(shape) for factor, shape in shapes.items()}
            for key, shapes in module_shapes.items()
        }
        self.module_list = sorted(self.module_shapes)
        self.module_index = {key: i for i, key in enumerate(self.module_list)}
        groups: dict[str, list[str]] = {}
        for key in self.module_list:
            groups.setdefault(group_key(self.module_shapes[key]), []).append(key)
        self._group_names = sorted(groups)
        self._group_modules = [groups[g] for g in self._group_names]
        self._group_idx = [torch.tensor([self.module_index[m] for m in members])
                           for members in self._group_modules]

        n_modules = len(self.module_list)
        self.id_emb = nn.Embedding(n_modules, id_dim)
        nn.init.normal_(self.id_emb.weight)
        trunk_in = task_dim + id_dim
        self.trunk = nn.ModuleList(ResidualBlock(trunk_in, trunk_hidden)
                                   for _ in range(trunk_layers))
        self.out_norm = nn.LayerNorm(trunk_in)
        self.out_proj = nn.Linear(trunk_in, hidden_dim)

        self.head_A = nn.ModuleDict()
        self.head_B = nn.ModuleDict()
        for gi, gname in enumerate(self._group_names):
            A_shape = self.module_shapes[self._group_modules[gi][0]]["A"]
            B_shape = self.module_shapes[self._group_modules[gi][0]]["B"]
            safe = f"g{gi}"
            self.head_A[safe] = nn.Linear(hidden_dim, A_shape[0] * A_shape[1])
            self.head_B[safe] = nn.Linear(hidden_dim, B_shape[0] * B_shape[1])

    def forward(self, task: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        batch = task.shape[0]
        n = len(self.module_list)
        ids = torch.arange(n, device=task.device)
        h = torch.cat([
            task[:, None, :].expand(batch, n, task.shape[-1]),
            self.id_emb(ids)[None].expand(batch, n, -1),
        ], dim=-1)
        for block in self.trunk:
            h = block(h)
        h = F.gelu(self.out_proj(self.out_norm(h)))              # [B, n, hidden]

        out: dict[str, dict[str, torch.Tensor]] = {}
        for gi, members in enumerate(self._group_modules):
            safe = f"g{gi}"
            idx = self._group_idx[gi].to(task.device)
            r, in_dim = self.module_shapes[members[0]]["A"]
            out_dim, _ = self.module_shapes[members[0]]["B"]
            a = self.head_A[safe](h[:, idx]).view(batch, len(members), r, in_dim)
            b = self.head_B[safe](h[:, idx]).view(batch, len(members), r, out_dim).transpose(-1, -2)
            for j, key in enumerate(members):
                out[key] = {"A": a[:, j], "B": b[:, j]}
        return out


class V4DirectABHyperNetwork(nn.Module):
    """Direct-head variant: condition encoder + DirectHeadDecoder."""

    def __init__(
        self,
        condition_dim: int,
        module_shapes: Mapping[str, Mapping[str, tuple[int, int]]],
        condition_hidden_dim: int = 256,
        hidden_dim: int = 512,
        id_dim: int = 256,
        trunk_layers: int = 6,
        trunk_hidden: int = 1024,
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
        self.decoder = DirectHeadDecoder(
            self.module_shapes, task_dim=hidden_dim, hidden_dim=hidden_dim,
            id_dim=id_dim, trunk_layers=trunk_layers, trunk_hidden=trunk_hidden,
        )

    def forward(self, condition: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        if condition.ndim == 3:
            condition = condition.mean(dim=1)
        return self.decoder(self.condition_encoder(condition))
