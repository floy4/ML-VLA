# src/mlvla/meta/e2e/evidence.py
"""Per-domain cached DINOv3 evidence (reservoir frames) + view params."""
from __future__ import annotations

from pathlib import Path

import torch

from mlvla.meta.view_params import pose_vector_from_config, view_vector_for_domain


class EvidenceBank:
    def __init__(self, feature_dir: Path, domains: list[str], pooling: str = "patch",
                 pose_feature: dict | None = None) -> None:
        self.feature_dir = Path(feature_dir)
        self.domains = list(domains)
        self.key = "features" if pooling == "patch" else "cls"
        # Optional pose_feature config block {scheme, cache, layer}: when absent,
        # view() keeps the legacy view_vector_for_domain path bit-identical.
        self.pose_feature = dict(pose_feature) if pose_feature else None
        self._blobs = {
            (domain, split): torch.load(self.feature_dir / f"{domain}_{split}.pt",
                                        map_location="cpu")
            for domain in domains for split in ("train", "val", "test")
        }

    def sample(self, domain: str, k: int = 8,
               generator: torch.Generator | None = None) -> torch.Tensor:
        feats = self._blobs[(domain, "train")][self.key]
        k = min(k, feats.shape[0])
        idx = torch.randint(feats.shape[0], (k,), generator=generator)
        return feats[idx]

    def domain_mean(self, domain: str, split: str = "val") -> torch.Tensor:
        return self._blobs[(domain, split)][self.key]

    def split_episode_ids(self, domain: str, split: str) -> list[int]:
        return [int(i) for i in self._blobs[(domain, split)]["episode_ids"].tolist()]

    def view(self, domain: str) -> torch.Tensor:
        """[pose_dim] float32 pose vector for the domain, dispatched by config.

        No pose_feature block -> legacy view7 (bit-identical regression path);
        with one -> the scheme's vector (view7/zero: 7, vggt/raymap: 2048).
        """
        if self.pose_feature:
            return pose_vector_from_config(domain, self.pose_feature)
        try:
            vec = view_vector_for_domain(domain)
        except ValueError:
            raise KeyError(
                f"unknown domain {domain!r}: no view params for condition "
                f"{domain.split('__', 1)[0]!r} (see mlvla/meta/view_params.py VIEW_PARAMS)"
            ) from None
        return torch.tensor(vec, dtype=torch.float32)
