"""Post-generation sanity checks on the perturbed replay datasets."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import pyarrow.parquet as pq
import yaml

from mlvla import paths as _paths


def run(root: pathlib.Path) -> int:
    conds = yaml.safe_load((root / "conditions.yaml").read_text())["conditions"]
    fails = []

    def check(cond, msg):
        flag = "PASS" if cond else "FAIL"
        if not cond:
            fails.append(msg)
        print(f"{flag}: {msg}")

    check(len(conds) == 9, f"conditions.yaml has 9 entries (got {len(conds)})")

    for cid in conds:
        cond_dir = root / cid
        info_path = cond_dir / "meta" / "info.json"
        if not info_path.exists():
            check(False, f"{cid}/meta/info.json exists")
            continue
        info = json.loads(info_path.read_text())
        check(info["total_episodes"] >= 400,
              f"{cid}: total_episodes >= 400 (got {info['total_episodes']}; source libero_goal caps at ~43/task)")
        check(info["fps"] == 10, f"{cid}: fps == 10")
        check(info["total_tasks"] == 10, f"{cid}: total_tasks == 10")

        data_path = cond_dir / "data" / "chunk-000" / "file-000.parquet"
        if data_path.exists():
            # Column-restricted read: full-table reads fail on the nested
            # observation-image columns (pyarrow ArrowNotImplementedError).
            t = pq.read_table(data_path, columns=["actions", "state"])
            actions = t.column("actions").to_pylist()
            check(len(actions) > 0, f"{cid}: data parquet non-empty ({len(actions)} rows)")
            if actions:
                check(min(len(a) for a in actions) == 7, f"{cid}: actions are 7-dim")
                check(all(-2 < v < 2 for a in actions[:100] for v in a),
                      f"{cid}: actions in plausible range [-2, 2]")
            states = t.column("state").to_pylist()
            if states:
                check(min(len(s) for s in states) == 8, f"{cid}: states are 8-dim")

    for cid in ["l1_warm_dim", "l2_cool_bright", "l3_directional_low",
                "c1_v1l1", "c2_v2l2", "c3_v3l3"]:
        check((root / cid / "scene.xml").exists(), f"{cid}/scene.xml exists")

    print()
    if fails:
        print(f"{len(fails)} CHECKS FAILED:")
        for f in fails:
            print(f"  X {f}")
        return 1
    print("All data checks passed.")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=_paths.get("perturbed_root"),
                   help="perturbed dataset root (default: paths.yaml perturbed_root)")
    args = p.parse_args(argv)
    sys.exit(run(pathlib.Path(args.root)))


if __name__ == "__main__":
    main()
