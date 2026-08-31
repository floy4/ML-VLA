"""Filter failed replay demos from the perturbed condition datasets.

Reads a replay-audit failure list (per-task failed init ranks, valid across
conditions since view/light perturbations do not change physics — produced
by comparing replay_success.json across conditions), then writes cleaned
LeRobot dataset copies to <dst_root>/<cond>. Originals are kept untouched.

Per condition it also asserts the per-episode length fingerprint matches the
reference (v1_azimuth30) dataset, guaranteeing the audit transfers.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import pyarrow.parquet as pq
import pyarrow as pa

from mlvla import paths as _paths

CONDS = ["v1_azimuth30", "v2_azimuth60", "v3_elev15_zoom125",
         "l1_warm_dim", "l2_cool_bright", "l3_directional_low",
         "c1_v1l1", "c2_v2l2", "c3_v3l3"]


def filter_condition(src: pathlib.Path, dst: pathlib.Path,
                     failed_by_task_idx: dict[int, set],
                     desc_by_idx: dict[int, str],
                     v1_fingerprint: list[tuple[int, int]]) -> None:
    if (dst / "data" / "chunk-000" / "file-000.parquet").exists():
        print(f"[{src.name}] already filtered, skip")
        return

    ep_t = pq.read_table(src / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    rows = sorted(zip(ep_t.column("episode_index").to_pylist(),
                      ep_t.column("task_index").to_pylist(),
                      ep_t.column("length").to_pylist()))
    fingerprint = [(t, l) for _, t, l in rows]
    assert fingerprint == v1_fingerprint, f"{src.name}: episode fingerprint differs from v1 — audit does not transfer!"

    # rank within task by episode_index order
    counters = {}
    drop = set()
    for e, t, l in rows:
        k = counters.get(t, 0)
        counters[t] = k + 1
        if k in failed_by_task_idx[t]:
            drop.add(e)
    keep = [e for e, t, l in rows if e not in drop]
    remap = {old: new for new, old in enumerate(keep)}

    # stream data parquet, filter + remap episode_index + rebuild global index
    dst_data = dst / "data" / "chunk-000"
    dst_data.mkdir(parents=True)
    src_pf = pq.ParquetFile(src / "data" / "chunk-000" / "file-000.parquet")
    schema = src_pf.schema_arrow
    writer = None
    g_idx = 0
    n_kept_rows = 0
    for batch in src_pf.iter_batches(batch_size=2048):
        tbl = pa.Table.from_batches([batch])
        eps = tbl.column("episode_index").to_pylist()
        mask = [e not in drop for e in eps]
        sub = tbl.filter(pa.array(mask))
        if sub.num_rows == 0:
            continue
        new_eps = pa.array([remap[e] for e in sub.column("episode_index").to_pylist()],
                           type=pa.int64())
        new_idx = pa.array(range(g_idx, g_idx + sub.num_rows), type=pa.int64())
        cols = {name: sub.column(name) for name in sub.column_names}
        cols["episode_index"] = new_eps
        cols["index"] = new_idx
        out = pa.table(cols, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(dst_data / "file-000.parquet", schema)
        writer.write_table(out)
        g_idx += sub.num_rows
        n_kept_rows += sub.num_rows
    if writer:
        writer.close()

    # episodes meta (remapped, kept only)
    kept_rows = [(remap[e], t, l) for e, t, l in rows if e not in drop]
    ep_out = pa.table({
        "episode_index": pa.array([r[0] for r in kept_rows], pa.int64()),
        "tasks": pa.array([[desc_by_idx[r[1]]] for r in kept_rows], pa.list_(pa.string())),
        "length": pa.array([r[2] for r in kept_rows], pa.int64()),
        "task_index": pa.array([r[1] for r in kept_rows], pa.int64()),
    }, schema=ep_t.schema)
    dst_meta = dst / "meta" / "episodes" / "chunk-000"
    dst_meta.mkdir(parents=True)
    pq.write_table(ep_out, dst_meta / "file-000.parquet")
    shutil.copy(src / "meta" / "tasks.parquet", dst / "meta" / "tasks.parquet")

    # info.json
    info = json.loads((src / "meta" / "info.json").read_text())
    info["total_episodes"] = len(kept_rows)
    info["total_frames"] = n_kept_rows
    info["splits"]["train"] = f"0:{len(kept_rows)}"
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    print(f"[{src.name}] kept {len(kept_rows)}/{len(rows)} eps, {n_kept_rows} frames "
          f"(dropped {len(drop)} eps)", flush=True)


def run(src_root: pathlib.Path, dst_root: pathlib.Path, audit_path: pathlib.Path) -> None:
    audit = json.loads(audit_path.read_text())

    # build task_index -> failed ranks map from v1 metadata + audit (by desc)
    v1_tasks = pq.read_table(src_root / "v1_azimuth30" / "meta" / "tasks.parquet")
    desc_by_idx = dict(zip(v1_tasks.column("task_index").to_pylist(), v1_tasks.column("task").to_pylist()))
    failed_by_task_idx = {}
    for stem, info in audit.items():
        desc = info["desc"]
        tidx = next(i for i, d in desc_by_idx.items() if d == desc)
        failed_by_task_idx[tidx] = set(info["failed_inits"])

    # v1 episodes: reference fingerprint (task, length) per episode in index order
    v1_ep = pq.read_table(src_root / "v1_azimuth30" / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    v1_rows = sorted(zip(v1_ep.column("episode_index").to_pylist(),
                         v1_ep.column("task_index").to_pylist(),
                         v1_ep.column("length").to_pylist()))
    v1_fingerprint = [(t, l) for _, t, l in v1_rows]

    for cond in CONDS:
        filter_condition(src_root / cond, dst_root / cond,
                         failed_by_task_idx, desc_by_idx, v1_fingerprint)

    print("\ndone. cleaned datasets at", dst_root)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src_root", default=_paths.get("perturbed_root"))
    p.add_argument("--dst_root", default=None,
                   help="cleaned output root (default: <src_root>_clean)")
    p.add_argument("--audit", default="/tmp/replay_audit_all.json",
                   help="replay audit JSON (per-task failed init ranks)")
    args = p.parse_args(argv)
    src_root = pathlib.Path(args.src_root)
    dst_root = pathlib.Path(args.dst_root) if args.dst_root else src_root.parent / (src_root.name + "_clean")
    run(src_root, dst_root, pathlib.Path(args.audit))


if __name__ == "__main__":
    main()
