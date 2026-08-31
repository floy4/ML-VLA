#!/usr/bin/env python3
"""Build configs/domains.yaml: the 90 (condition, task) verification domains."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import yaml

CONDITIONS = [
    "v1_azimuth30", "v2_azimuth60", "v3_elev15_zoom125",
    "l1_warm_dim", "l2_cool_bright", "l3_directional_low",
    "c1_v1l1", "c2_v2l2", "c3_v3l3",
]
from mlvla import paths as _paths

EXPERT_ROOT = Path(_paths.get("campaign_expert_root"))
DATA_ROOT = Path(_paths.get("campaign_data_root"))


def episode_metadata(condition: str) -> list[dict]:
    pattern = DATA_ROOT / condition / "meta" / "episodes" / "chunk-000"
    rows = []
    for path in sorted(pattern.glob("file-*.parquet")):
        table = pq.read_table(path)
        rows.extend(table.to_pylist())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "configs" / "domains.yaml")
    args = parser.parse_args()

    domains = {}
    for condition in CONDITIONS:
        tasks_table = pq.read_table(DATA_ROOT / condition / "meta" / "tasks.parquet").to_pydict()
        prompt_by_index = dict(zip(tasks_table["task_index"], tasks_table["task"]))
        episodes_by_task: dict[str, list[int]] = {}
        for row in episode_metadata(condition):
            prompt = prompt_by_index[int(row["task_index"])]
            episodes_by_task.setdefault(prompt, []).append(int(row["episode_index"]))

        for task_index, prompt in sorted(prompt_by_index.items()):
            digest = hashlib.md5(prompt.encode()).hexdigest()[:8]
            canonical = EXPERT_ROOT / condition / f"expert_{digest}" / "phase1" / "final" / "params.canonical.npz"
            if not canonical.exists():
                raise FileNotFoundError(f"missing expert weights: {canonical}")
            episodes = sorted(episodes_by_task.get(prompt, []))
            if not episodes:
                raise ValueError(f"{condition}: no episodes for task {prompt!r}")
            domains[f"{condition}__{digest}"] = {
                "condition": condition,
                "task_hash": digest,
                "prompt": prompt,
                "canonical_npz": str(canonical),
                "dataset_root": str(DATA_ROOT / condition),
                "episodes": episodes,
            }

    if len(domains) != 90:
        raise ValueError(f"expected 90 domains, built {len(domains)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(domains, sort_keys=True))
    total = sum(len(d["episodes"]) for d in domains.values())
    print(f"wrote {args.output}: {len(domains)} domains, {total} episodes total")


if __name__ == "__main__":
    main()
