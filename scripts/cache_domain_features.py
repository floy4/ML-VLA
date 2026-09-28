#!/usr/bin/env python3
"""Cache 4 reservoir DINOv3 frames per episode for each of the 90 domains."""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def reservoir_frames(parquet_paths: list[Path], episodes: set[int], frames: int, seed: int):
    import pyarrow.parquet as pq

    rng = random.Random(seed)
    reservoirs: dict[int, list[bytes]] = {}
    counts: dict[int, int] = {}
    wanted = ["episode_index", "image"]
    for parquet_path in parquet_paths:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=256, columns=wanted):
            for row in batch.to_pylist():
                episode = int(row["episode_index"])
                if episode not in episodes:
                    continue
                raw = row["image"].get("bytes")
                if not raw:
                    continue
                counts[episode] = counts.get(episode, 0) + 1
                bucket = reservoirs.setdefault(episode, [])
                if len(bucket) < frames:
                    bucket.append(raw)
                else:
                    replacement = rng.randrange(counts[episode])
                    if replacement < frames:
                        bucket[replacement] = raw
    return [(episode, values) for episode, values in sorted(reservoirs.items()) if len(values) == frames]


def first_frames(parquet_paths: list[Path], episodes: set[int]):
    """First row per episode in file order = frame 0 of that episode."""
    import pyarrow.parquet as pq

    firsts: dict[int, bytes] = {}
    wanted = ["episode_index", "image"]
    for parquet_path in parquet_paths:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=256, columns=wanted):
            for row in batch.to_pylist():
                episode = int(row["episode_index"])
                if episode not in episodes or episode in firsts:
                    continue
                raw = row["image"].get("bytes")
                if raw:
                    firsts[episode] = raw
    return sorted(firsts.items())


def split_indices(count: int, seed: int):
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    train_end = int(count * 0.70)
    val_end = train_end + int(count * 0.15)
    return {"train": indices[:train_end], "val": indices[train_end:val_end], "test": indices[val_end:]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", type=Path, default=REPO_ROOT / "configs" / "domains.yaml")
    parser.add_argument("--dinov3", type=str, default=__import__("mlvla.paths", fromlist=["PATHS"]).PATHS["dinov3_model"])
    parser.add_argument("--output-dir", type=Path, default=Path(__import__("mlvla.paths", fromlist=["PATHS"]).PATHS["feature_cache_dir"]))
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--frame-mode", choices=["reservoir", "first"], default="reservoir")
    parser.add_argument("--manifest-name", default="manifest.json",
                        help="per-shard manifest filename (parallel shards then merge)")
    parser.add_argument("--split-registry", default=None,
                        help="comma list whose sorted order defines domain_index "
                             "for split seeds; default = the --only set (original "
                             "build semantics)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--only", type=str, default=None, help="comma-separated domain ids (debug)")
    args = parser.parse_args()

    import numpy as np
    import torch
    import yaml
    from PIL import Image

    from mlvla.meta.dinov3_encoder import encode_frames, load_dinov3

    domains = yaml.safe_load(args.domains.read_text())
    if args.only:
        wanted = set(args.only.split(","))
        domains = {k: v for k, v in domains.items() if k in wanted}
    # Split seeds use seed + domain_index over the sorted split-registry (the
    # original level cache enumerated its 17-domain --only set; pass that same
    # list here to reproduce its train/val/test splits exactly).
    # Registry order is taken verbatim when given: the original level cache was
    # built in two passes (camera/lighting/noise/texture 12-block, then
    # add_object/clean 4-block), each condition-major — domain_index must
    # reproduce those enumerations, not a sorted merge.
    registry = args.split_registry.split(",") if args.split_registry else sorted(domains)
    model, hidden, num_register, _ = load_dinov3(args.dinov3, device="cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": args.seed, "frames": args.frames, "frame_mode": args.frame_mode,
                "model": args.dinov3, "domains": {}}

    for domain_index, domain_id in enumerate(registry):
        if domain_id not in domains:
            continue
        spec = domains[domain_id]
        parquets = sorted((Path(spec["dataset_root"]) / "data").glob("chunk-*/file-*.parquet"))
        if args.frame_mode == "first":
            args.frames = 1
            samples = [(ep, [raw]) for ep, raw in first_frames(parquets, set(spec["episodes"]))]
        else:
            samples = reservoir_frames(parquets, set(spec["episodes"]), args.frames, args.seed + domain_index)
        if not samples:
            raise RuntimeError(f"{domain_id}: reservoir produced no complete episodes")
        frames_np = [np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")) for _, values in samples for raw in values]
        feats, cls = [], []
        for start in range(0, len(frames_np), args.batch_size):
            out = encode_frames(model, frames_np[start : start + args.batch_size], device="cuda",
                                num_register_tokens=num_register, hidden_size=hidden)
            feats.append(out["patch"].mean(axis=1))  # [n, 1024] meta-lora pooling
            cls.append(out["cls"])
        patch_mean = np.concatenate(feats).reshape(len(samples), args.frames, -1)
        cls_all = np.concatenate(cls).reshape(len(samples), args.frames, -1)
        splits = split_indices(len(samples), args.seed + domain_index)
        manifest["domains"][domain_id] = {}
        for split, indices in splits.items():
            output = args.output_dir / f"{domain_id}_{split}.pt"
            torch.save(
                {
                    "features": torch.from_numpy(patch_mean[indices].copy()),
                    "cls": torch.from_numpy(cls_all[indices].copy()),
                    "episode_ids": torch.tensor([samples[i][0] for i in indices]),
                    "domain": domain_id,
                    "split": split,
                },
                output,
            )
            manifest["domains"][domain_id][split] = {"episodes": len(indices)}
        print(f"{domain_id}: {len(samples)} episodes cached", flush=True)

    (args.output_dir / args.manifest_name).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.output_dir}/{args.manifest_name} with {len(manifest['domains'])} domains")


if __name__ == "__main__":
    main()
