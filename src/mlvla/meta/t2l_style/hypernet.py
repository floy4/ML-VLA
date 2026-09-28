"""
T2L-style hypernetwork for π0.5 LoRA generation.
Based on Text-to-LoRA (Sakana AI) architecture with layer embeddings and shared trunk.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal


class MLPResidualBlock(nn.Module):
    """Pre-norm residual MLP block (T2L-style)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        pre_layer_norm: bool = True,
        post_dropout: float = 0.05,
    ):
        super().__init__()
        layers = []
        if pre_layer_norm:
            layers.append(nn.LayerNorm(input_size))
        layers += [
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(post_dropout),
            nn.Linear(hidden_size, output_size),
            nn.SiLU(),
        ]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class T2LStyleHyperNet(nn.Module):
    """
    Hypernetwork generating LoRA A/B for π0.5 VLA policy.

    Architecture (based on T2L HyperModulator):
    - Layer depth embedding: nn.Embedding(max_layers, depth_emb_size)
    - Layer type embedding: nn.Embedding(n_module_types, type_emb_size)
    - Mixer MLP: projects concatenated [task_emb + depth_emb + type_emb]
    - Residual trunk: 2× MLPResidualBlocks (pre-norm, SiLU, dropout 0.05)
    - Projection MLP: to head_in_size
    - Heads: nn.ModuleDict of Linear heads, one per module type (17 groups)

    Args:
        module_shapes: Dict mapping group_key -> {"A": (r, in_features), "B": (out_features, r)}
                       Example: {"A16x512_B512x16": {"A": (16, 512), "B": (512, 16)}}
        max_layers: Maximum number of layers per module type (e.g., 27 for π0.5)
        task_emb_size: Dimension of input task embedding (from TwoBranchConditionEncoder)
        depth_emb_size: Dimension of layer depth embedding
        type_emb_size: Dimension of layer type embedding
        trunk_hidden: Hidden dimension in trunk MLPs
        head_in_size: Input dimension to output heads
        lora_rank: LoRA rank (r)
        shared_AB_head: If True, use single head + per-module learned offset (40M variant)
        dropout: Dropout rate

    Param count estimate (with head_in_size=128, max_layers=27):
    - Embeddings: (27 depth + 17 type) × 64 ≈ 3K (negligible)
    - Mixer: (256+64+64)×4 → 1536 → 384 linear: ~0.6M
    - Trunk: 2× residual blocks: ~1.2M
    - Heads: 17 groups × (128 × (r×(in+out))) ≈ 17 × 128 × (16×~2000) ≈ 70M

    Total: ~72M (vs current 713M), 10× reduction.
    Aligns with T2L-L (55M) configuration.
    """

    def __init__(
        self,
        module_shapes: dict[str, dict[str, tuple[int, int]]],
        max_layers: int,
        task_emb_size: int = 256,
        depth_emb_size: int = 64,
        type_emb_size: int = 64,
        trunk_hidden: int = 512,
        head_in_size: int = 128,
        lora_rank: int = 16,
        shared_AB_head: bool = False,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.max_layers = max_layers
        self.task_emb_size = task_emb_size
        self.lora_rank = lora_rank
        self.shared_AB_head = shared_AB_head

        # Parse module shapes to extract in/out features per group
        # group_key format: "A{r}x{in}_B{out}x{r}"
        self.module_groups = {}  # group_key -> {"in": in_features, "out": out_features}
        for group_key, shapes in module_shapes.items():
            r_a, in_feat = shapes["A"]
            out_feat, r_b = shapes["B"]
            assert r_a == r_b == lora_rank, f"Rank mismatch in {group_key}"
            self.module_groups[group_key] = {"in": in_feat, "out": out_feat}

        self.target_modules = list(self.module_groups.keys())
        self.module_to_int = {m: i for i, m in enumerate(self.target_modules)}

        # Extract in/out features for each module
        self.in_features = {m: self.module_groups[m]["in"] for m in self.target_modules}
        self.out_features = {m: self.module_groups[m]["out"] for m in self.target_modules}

        # Layer embeddings
        self.layer_depth_encoder = nn.Sequential(
            nn.Embedding(max_layers, depth_emb_size),
            nn.LayerNorm(depth_emb_size),
        )
        self.layer_type_encoder = nn.Sequential(
            nn.Embedding(len(self.target_modules), type_emb_size),
            nn.LayerNorm(type_emb_size),
        )

        # Task encoder (simple MLP, replaces T2L's TaskEncoder)
        encoded_task_emb_size = task_emb_size // 2
        self.task_encoder = nn.Sequential(
            nn.Linear(task_emb_size, encoded_task_emb_size),
            nn.LayerNorm(encoded_task_emb_size),
        )

        # Mixer: concatenate [task_emb + depth_emb + type_emb]
        mlp_inp_size = encoded_task_emb_size + depth_emb_size + type_emb_size
        self.mixer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(mlp_inp_size, mlp_inp_size * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_inp_size * 4, mlp_inp_size),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Residual trunk (mlp1 + mlp2, shared by both paths)
        self.mlp1 = MLPResidualBlock(
            mlp_inp_size,
            mlp_inp_size * 4,
            mlp_inp_size,
            pre_layer_norm=True,
            post_dropout=dropout,
        )
        self.mlp2 = MLPResidualBlock(
            mlp_inp_size,
            mlp_inp_size * 4,
            mlp_inp_size,
            pre_layer_norm=True,
            post_dropout=dropout,
        )

        # Projection to head input size
        self.mlp3 = nn.Sequential(
            nn.LayerNorm(mlp_inp_size),
            nn.Linear(mlp_inp_size, mlp_inp_size * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_inp_size * 4, head_in_size),
            nn.SiLU(),
        )

        # Output heads
        if shared_AB_head:
            # Per-group learned embeddings for A/B distinction.
            # Added at the trunk level (mlp_inp_size) after mlp2, before mlp3.
            # Zero-initialized so training starts close to the shared baseline.
            self.out_emb = nn.ParameterDict(
                (
                    m,
                    nn.ParameterDict(
                        dict(
                            A=nn.Parameter(torch.zeros(mlp_inp_size)),
                            B=nn.Parameter(torch.zeros(mlp_inp_size)),
                        )
                    ),
                )
                for m in self.target_modules
            )

        # Create heads
        heads = []
        for module in self.target_modules:
            in_feat = self.in_features[module]
            out_feat = self.out_features[module]

            if not shared_AB_head:
                # Each head outputs flattened A and B
                output_size = lora_rank * (in_feat + out_feat)
            else:
                # Shared head outputs max(r * in, r * out)
                output_size = lora_rank * max(in_feat, out_feat)

            layer = nn.Linear(head_in_size, output_size, bias=False)
            heads.append((module, layer))

        self.heads = nn.ModuleDict(heads)

        # Pre-compute split shapes for efficient unpacking
        self.split_shapes = {}
        for module in self.target_modules:
            r = lora_rank
            in_feat = self.in_features[module]
            out_feat = self.out_features[module]
            self.split_shapes[module] = (r * in_feat, r * out_feat)

        # Store scaling factor (can be overridden externally)
        self.scaling = 1.0

    def forward(
        self,
        task_emb: torch.Tensor,
        layer_indices: torch.Tensor,
        layer_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate LoRA A and B for a specific layer type.

        Args:
            task_emb: [B, task_emb_size] condition embedding (from TwoBranchConditionEncoder)
            layer_indices: [B] layer depth indices
            layer_type: Module group key (e.g., "A16x512_B512x16")

        Returns:
            A: [B, r, in_features] LoRA A matrix
            B: [B, out_features, r] LoRA B matrix (transposed)
        """
        bs = len(layer_indices)
        assert layer_type in self.target_modules, f"Unknown layer type: {layer_type}"

        # Get embeddings
        depth_emb = self.layer_depth_encoder(layer_indices)  # [B, depth_emb_size]
        type_idx = torch.tensor(
            [self.module_to_int[layer_type]],
            device=layer_indices.device,
        ).expand(bs)
        type_emb = self.layer_type_encoder(type_idx)  # [B, type_emb_size]

        # Encode task embedding
        encoded_task_emb = self.task_encoder(task_emb)  # [B, encoded_task_emb_size]

        # Concatenate and mix
        cat_emb = torch.cat([encoded_task_emb, depth_emb, type_emb], dim=-1)
        mlp_inp = self.mixer(cat_emb)

        # Residual trunk — mlp1 only; mlp2 is applied per-path below so the
        # shared head can inject out_emb between mlp1 and mlp2 without the
        # trunk's mlp2 call double-processing the signal.
        mlp_out = self.mlp1(mlp_inp)

        # Get head
        head = self.heads[layer_type]

        if not self.shared_AB_head:
            # Non-shared: mlp2 then mlp3 then head
            head_out = head(self.mlp3(self.mlp2(mlp_out)))
            A_flat, B_flat = torch.split(head_out, self.split_shapes[layer_type], dim=-1)

            # Reshape to matrix form
            A = A_flat.reshape(bs, self.lora_rank, self.in_features[layer_type])
            B = B_flat.reshape(bs, self.lora_rank, self.out_features[layer_type]).transpose(-1, -2)
        else:
            # Shared head with per-group A/B embeddings.
            # out_emb is added at the trunk level (after mlp1, before mlp2)
            # so A and B see different signals through the remaining layers.
            # out_emb is zero-initialized so training starts from the shared baseline.
            splitted_out = []
            for out_emb_key, num_features in zip(
                ["A", "B"],
                [self.in_features[layer_type], self.out_features[layer_type]],
            ):
                out_emb = self.out_emb[layer_type][out_emb_key]
                head_in = self.mlp3(self.mlp2(mlp_out + out_emb))
                head_out = head(head_in)
                head_out = head_out.view(bs, self.lora_rank, -1)
                head_out = head_out[..., :num_features]
                head_out = head_out.reshape(bs, self.lora_rank * num_features)
                splitted_out.append(head_out)

            A_flat, B_flat = splitted_out
            A = A_flat.reshape(bs, self.lora_rank, self.in_features[layer_type])
            B = B_flat.reshape(bs, self.lora_rank, self.out_features[layer_type]).transpose(-1, -2)

        return A, B

    def get_delta_weights(
        self,
        layer_indices: torch.Tensor,
        layer_type: str,
        task_emb: torch.Tensor,
        factorized: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Generate LoRA delta weights (A, B) or full deltaW = B @ A.

        Args:
            layer_indices: [B] layer depth indices
            layer_type: Module group key
            task_emb: [B, task_emb_size] condition embedding
            factorized: If True, return (A, B); else return deltaW = B @ A

        Returns:
            If factorized: (A, B) where A: [B, r, in], B: [B, out, r]
            Else: deltaW: [B, out, in]
        """
        A, B = self.forward(task_emb, layer_indices, layer_type)

        if factorized:
            return A, B
        else:
            deltaW = torch.bmm(B, A)
            return deltaW

    def gen_lora(
        self,
        layer_indices: torch.Tensor,
        task_emb: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Generate LoRA for all target modules (inference mode).

        Args:
            layer_indices: [L] layer indices (single task, batch dim = layers)
            task_emb: [1, task_emb_size] single task embedding

        Returns:
            Dict mapping module_name -> {"lora_A": [L, r, in], "lora_B": [L, out, r]}
        """
        assert task_emb.shape[0] == 1, "Only one task at a time"

        lora_dict = {}
        for target_module in self.target_modules:
            A, B = self.get_delta_weights(
                layer_indices,
                target_module,
                task_emb.expand(len(layer_indices), -1),
                factorized=True,
            )
            lora_dict[target_module] = {
                "lora_A": A.cpu().contiguous(),
                "lora_B": B.cpu().contiguous(),
            }

        return lora_dict
