"""Canonical LoRA npz I/O for the wizard params.canonical.npz schema."""
from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CanonicalModule:
    index: int
    key: str
    rank: int
    alpha: float
    A: np.ndarray  # [rank, in]
    B: np.ndarray  # [out, rank]

    @property
    def scale(self) -> float:
        return self.alpha / self.rank


class CanonicalFile:
    def __init__(self, path, data, manifest):
        self.path = Path(path)
        self._data = data
        self.manifest = manifest
        self._entries = manifest.get("modules", [])

    def __len__(self):
        return len(self._entries)

    def module(self, index: int) -> CanonicalModule:
        item = self._entries[index]
        A = np.asarray(self._data[f"module_{index}_A"])
        B = np.asarray(self._data[f"module_{index}_B"])
        return CanonicalModule(
            index=index,
            key=str(item["key"]),
            rank=int(A.shape[0]),
            alpha=float(item.get("alpha", 16.0)),
            A=A,
            B=B,
        )

    def modules(self):
        for index in range(len(self)):
            yield self.module(index)

    @property
    def module_specs(self):
        """Manifest module entries (key + metadata.jax_stem), for schema_map.build_targets."""
        return self._entries

    def close(self):
        self._data.close()


@contextmanager
def load_canonical(path):
    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest_json"]))
        yield CanonicalFile(path, data, manifest)


def save_canonical(path, manifest, arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, manifest_json=np.asarray(json.dumps(manifest)), **arrays)


def assert_compatible(*files: CanonicalFile) -> None:
    reference = files[0]
    for other in files[1:]:
        if len(other) != len(reference):
            raise ValueError(f"{other.path}: {len(other)} modules, expected {len(reference)}")
        for ref, oth in zip(reference.modules(), other.modules()):
            if ref.key != oth.key or ref.A.shape != oth.A.shape or ref.B.shape != oth.B.shape:
                raise ValueError(
                    f"module {ref.index} mismatch between {reference.path} and {other.path}: "
                    f"{ref.key}{ref.A.shape}{ref.B.shape} vs {oth.key}{oth.A.shape}{oth.B.shape}"
                )
