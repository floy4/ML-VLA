"""Central path configuration.

All machine-specific absolute paths live in configs/paths.yaml (override with
the MLVLA_PATHS env var). Repo-relative values in the yaml are resolved
against the repository root.
"""
from __future__ import annotations

import os
import pathlib

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load() -> dict[str, str]:
    cfg_path = os.environ.get("MLVLA_PATHS") or (REPO_ROOT / "configs" / "paths.yaml")
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    resolved = {}
    for key, value in raw.items():
        p = pathlib.Path(os.path.expanduser(str(value)))
        if not p.is_absolute():
            p = REPO_ROOT / p
        resolved[key] = str(p)
    return resolved


PATHS: dict[str, str] = _load()


def get(key: str) -> str:
    return PATHS[key]


def add_to_sys_path() -> None:
    """Make external repos (openpi, LIBERO-plus) importable."""
    import sys

    for key in ("openpi_root", "libero_plus_root"):
        root = pathlib.Path(PATHS[key])
        for sub in ("src",):
            candidate = root / sub
            if candidate.is_dir() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))


def set_derived_env() -> None:
    """Env vars the training/eval stack expects, defaulted from paths.yaml."""
    mapping = {
        "HF_LEROBOT_HOME": "lerobot_home",
        "HF_HOME": "hf_cache",
        "HF_DATASETS_CACHE": "hf_datasets_cache",
        "TMPDIR": "tmpdir",
    }
    for env_key, path_key in mapping.items():
        os.environ.setdefault(env_key, PATHS[path_key])
