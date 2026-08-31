"""Framework-independent, lossless representation of a LoRA adapter.

The paper's tokenizer assumes the usual PyTorch/PEFT convention: ``A`` has
shape ``[rank, in_features]`` and ``B`` has shape ``[out_features, rank]``.
Keeping that convention at the project boundary prevents accidental A/B
transposes when checkpoints originate in JAX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class LoRAModule:
    key: str
    A: np.ndarray
    B: np.ndarray
    alpha: float = 16.0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        a = np.asarray(self.A)
        b = np.asarray(self.B)
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"{self.key}: A and B must be matrices, got {a.shape}, {b.shape}")
        if a.shape[0] != b.shape[1]:
            raise ValueError(f"{self.key}: incompatible LoRA rank: A={a.shape}, B={b.shape}")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"{self.key}: adapter contains NaN or infinity")
        object.__setattr__(self, "A", a)
        object.__setattr__(self, "B", b)

    @property
    def rank(self) -> int:
        return int(self.A.shape[0])

    @property
    def delta(self) -> np.ndarray:
        return (self.B @ self.A) * (self.alpha / self.rank)


@dataclass(frozen=True)
class CanonicalLoRAAdapter:
    modules: Mapping[str, LoRAModule]
    base_checkpoint: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        modules = dict(self.modules)
        if not modules:
            raise ValueError("A canonical adapter must contain at least one module")
        if set(modules) != {m.key for m in modules.values()}:
            raise ValueError("Module mapping keys must equal LoRAModule.key")
        object.__setattr__(self, "modules", modules)

    def validate(self, *, rank: int = 16, alpha: float = 16.0) -> None:
        for module in self.modules.values():
            if module.rank != rank:
                raise ValueError(f"{module.key}: rank {module.rank}, expected {rank}")
            if not np.isclose(module.alpha, alpha):
                raise ValueError(f"{module.key}: alpha {module.alpha}, expected {alpha}")

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for key in sorted(self.modules):
            module = self.modules[key]
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(module.A).view(np.uint8))
            digest.update(np.ascontiguousarray(module.B).view(np.uint8))
            digest.update(str(module.alpha).encode())
        return digest.hexdigest()

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        arrays: dict[str, np.ndarray] = {}
        manifest = {"base_checkpoint": self.base_checkpoint, "metadata": dict(self.metadata), "modules": []}
        for index, key in enumerate(sorted(self.modules)):
            module = self.modules[key]
            arrays[f"module_{index}_A"] = module.A
            arrays[f"module_{index}_B"] = module.B
            manifest["modules"].append({"key": key, "alpha": module.alpha, "metadata": dict(module.metadata)})
        arrays["manifest_json"] = np.asarray(json.dumps(manifest))
        np.savez_compressed(path, **arrays)

    @classmethod
    def load_npz(cls, path: str | Path) -> "CanonicalLoRAAdapter":
        with np.load(path, allow_pickle=False) as data:
            manifest = json.loads(str(data["manifest_json"]))
            modules = {}
            for index, item in enumerate(manifest["modules"]):
                key = item["key"]
                modules[key] = LoRAModule(
                    key, data[f"module_{index}_A"].copy(), data[f"module_{index}_B"].copy(),
                    float(item["alpha"]), item.get("metadata", {}),
                )
        return cls(modules, manifest.get("base_checkpoint"), manifest.get("metadata", {}))
