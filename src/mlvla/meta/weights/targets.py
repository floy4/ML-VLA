"""Load selected oracle A/B targets for N domains."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import torch

from .lora_io import load_canonical


def selected_rows(path, max_modules=None):
    rows = json.loads(Path(path).read_text())
    if max_modules is not None:
        rows = rows[: int(max_modules)]
    return rows


def load_direct_targets(oracle_paths: Mapping[str, Path], rows):
    domain_targets = {}
    shapes: dict[str, dict[str, tuple[int, ...]]] = {}
    modules = []
    for domain_index, (domain, path) in enumerate(oracle_paths.items()):
        with load_canonical(path) as f:
            domain_modules = {}
            for row in rows:
                module = f.module(int(row["source_index"]))
                domain_modules[module.key] = {
                    "A": torch.from_numpy(module.A.copy()).float(),
                    "B": torch.from_numpy(module.B.copy()).float(),
                }
                if domain_index == 0:
                    shapes[module.key] = {"A": tuple(module.A.shape), "B": tuple(module.B.shape)}
                    modules.append({"source_index": module.index, "key": module.key})
            domain_targets[domain] = domain_modules
    return domain_targets, shapes, modules
