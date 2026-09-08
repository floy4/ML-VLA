# src/mlvla/meta/e2e/evidence.py
"""Per-domain cached DINOv3 evidence (reservoir frames) + view params."""
from __future__ import annotations

from pathlib import Path

import torch

from mlvla.meta.view_params import view_vector_for_domain

# Identity camera pose (no perturbation), used when the domain's condition part
# is not in VIEW_PARAMS (e.g. synthetic domain names in tests). Real runs use
# known conditions and never hit this branch.
_IDENTITY_VIEW = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


class EvidenceBank:
    def __init__(self, feature_dir: Path, domains: list[str], pooling: str = "patch") -> None:
        self.feature_dir = Path(feature_dir)
        self.domains = list(domains)
        self.key = "features" if pooling == "patch" else "cls"
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
        try:
            vec = view_vector_for_domain(domain)
        except ValueError:
            vec = _IDENTITY_VIEW
        return torch.tensor(vec, dtype=torch.float32)
