"""Shared-head hypernetworks in the spirit of text-to-lora's ``shared_AB_head``.

Instead of 458 independent ``ModuleABHead`` output heads (~1.5B params), every
module shares ONE coefficient head. Per-module learned embeddings (the T2L
``out_emb`` analog) differentiate modules and the A/B factors, and per-shape
learned output dictionaries decode the coefficients into full A/B tensors:

    C = coeff_head(pre_mlp(h + E[i]))          # [B, n_modules_in_shape, r, d]
    A_i = C_i @ dict_A_shape                   # [B, r, in]
    B_i = (C'_i @ dict_B_shape).T              # [B, out, r]

~40M params for the 458-module fullvw4 selection vs 1528M for per-module heads.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


class SharedDictDecoder(nn.Module):
    """One shared coefficient head + per-shape learned output dictionaries."""

    def __init__(
        self,
        module_shapes: Mapping[str, Mapping[str, tuple[int, int]]],
        hidden_dim: int = 512,
        dict_dim: int = 64,
    ) -> None:
        super().__init__()
        self.module_shapes = {
            key: {factor: tuple(shape) for factor, shape in shapes.items()}
            for key, shapes in module_shapes.items()
        }
        self.dict_dim = dict_dim
        self.module_list = sorted(self.module_shapes)
        self.module_index = {key: i for i, key in enumerate(self.module_list)}
        groups: dict[tuple[tuple[int, int], tuple[int, int]], list[str]] = {}
        for key in self.module_list:
            shapes = self.module_shapes[key]
            groups.setdefault((shapes["A"], shapes["B"]), []).append(key)
        self.group_shapes = list(groups)
        self._group_modules = [members for members in groups.values()]
        self.max_rank = max(A_shape[0] for A_shape, _ in self.group_shapes)

        n_modules = len(self.module_list)
        self.E_A = nn.Embedding(n_modules, hidden_dim)
        self.E_B = nn.Embedding(n_modules, hidden_dim)
        self.pre_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.coeff_head = nn.Linear(hidden_dim, self.max_rank * dict_dim)
        self.dict_A = nn.ParameterDict()
        self.dict_B = nn.ParameterDict()
        for gi, (A_shape, B_shape) in enumerate(self.group_shapes):
            scale = dict_dim ** -0.5
            self.dict_A[f"g{gi}"] = nn.Parameter(torch.randn(dict_dim, A_shape[1]) * scale)
            self.dict_B[f"g{gi}"] = nn.Parameter(torch.randn(dict_dim, B_shape[0]) * scale)

    def forward(self, hidden: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        batch = hidden.shape[0]
        out: dict[str, dict[str, torch.Tensor]] = {}
        for gi, (A_shape, B_shape) in enumerate(self.group_shapes):
            members = self._group_modules[gi]
            idx = torch.tensor([self.module_index[m] for m in members], device=hidden.device)
            n_g = len(members)
            rank_a, in_dim = A_shape
            out_dim, rank_b = B_shape
            inp_a = hidden.unsqueeze(1) + self.E_A(idx).unsqueeze(0)
            coeff_a = self.coeff_head(self.pre_mlp(inp_a))[
                ..., : rank_a * self.dict_dim
            ].view(batch, n_g, rank_a, self.dict_dim)
            a = coeff_a @ self.dict_A[f"g{gi}"]
            inp_b = hidden.unsqueeze(1) + self.E_B(idx).unsqueeze(0)
            rank_b_eff = min(rank_b, self.max_rank)
            coeff_b = self.coeff_head(self.pre_mlp(inp_b))[
                ..., : rank_b_eff * self.dict_dim
            ].view(batch, n_g, rank_b_eff, self.dict_dim)
            b = (coeff_b @ self.dict_B[f"g{gi}"]).transpose(-1, -2)
            for j, key in enumerate(members):
                out[key] = {"A": a[:, j], "B": b[:, j]}
        return out


class SharedDirectABHyperNetwork(nn.Module):
    """Concat variant: condition encoder + shared trunk + SharedDictDecoder."""

    def __init__(
        self,
        condition_dim: int,
        module_shapes: Mapping[str, Mapping[str, tuple[int, int]]],
        condition_hidden_dim: int = 256,
        hidden_dim: int = 512,
        head_bottleneck: int = 32,
        dict_dim: int = 64,
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
        self.decoder = SharedDictDecoder(self.module_shapes, hidden_dim, dict_dim)

    def forward(self, condition: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        if condition.ndim == 3:
            condition = condition.mean(dim=1)
        hidden = self.trunk(self.condition_encoder(condition))
        return self.decoder(hidden)


class SharedFiLMABHyperNetwork(nn.Module):
    """FiLM variant: DINO path modulated by a pose encoder + SharedDictDecoder.

    Identity-at-init gamma/beta so the untrained model computes the pose-free
    function (mirrors FiLMABHyperNetwork).
    """

    def __init__(
        self,
        dino_dim: int,
        view_dim: int,
        module_shapes: Mapping[str, Mapping[str, tuple[int, int]]],
        condition_hidden_dim: int = 256,
        hidden_dim: int = 512,
        head_bottleneck: int = 32,
        dict_dim: int = 64,
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
        self.decoder = SharedDictDecoder(self.module_shapes, hidden_dim, dict_dim)

    def forward(self, dino: torch.Tensor, view: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        if dino.ndim == 3:
            dino = dino.mean(dim=1)
        phi = self.dino_encoder(dino)
        z = self.pose_encoder(view)
        hidden = self.trunk(self.gamma_out(z) * phi + self.beta_out(z))
        return self.decoder(hidden)
