#!/usr/bin/env python3
"""First-frame feature cache for lighting_L3__deep with splits preserved from
the original reservoir cache (its 90/21/21 split was derived from lighting_L3's
membership rather than split_indices, so copy episode->split verbatim)."""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT / "scripts", REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cache_domain_features import first_frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", type=Path, default=REPO_ROOT / "configs" / "domains.yaml")
    ap.add_argument("--dinov3", type=str, default=__import__("mlvla.paths", fromlist=["PATHS"]).PATHS["dinov3_model"])
    ap.add_argument("--output-dir", type=Path, default=Path("/data2/zhy/meta_lora_offload/feature_cache_level_firstframe"))
    ap.add_argument("--old-cache", type=Path, default=Path("/home/zhy/vla/meta_lora/outputs/feature_cache_level"))
    ap.add_argument("--domain", default="lighting_L3__deep")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    import numpy as np
    import torch
    import yaml
    from PIL import Image

    from mlvla.meta.dinov3_encoder import encode_frames, load_dinov3

    spec = yaml.safe_load(args.domains.read_text())[args.domain]
    parquets = sorted((Path(spec["dataset_root"]) / "data").glob("chunk-*/file-*.parquet"))
    samples = first_frames(parquets, set(spec["episodes"]))
    by_ep = dict(samples)

    want = {}
    for split in ["train", "val", "test"]:
        blob = torch.load(args.old_cache / f"{args.domain}_{split}.pt", map_location="cpu", weights_only=False)
        want[split] = blob["episode_ids"].tolist()
    missing = set().union(*(set(v) for v in want.values())) - set(by_ep)
    assert not missing, f"{len(missing)} episodes missing from parquet, e.g. {sorted(missing)[:5]}"

    model, hidden, num_register, _ = load_dinov3(args.dinov3, device="cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"domain": args.domain, "splits": {}}
    for split, eps in want.items():
        frames_np = [np.asarray(Image.open(io.BytesIO(by_ep[ep])).convert("RGB")) for ep in eps]
        feats, cls = [], []
        for start in range(0, len(frames_np), args.batch_size):
            out = encode_frames(model, frames_np[start:start + args.batch_size], device="cuda",
                                num_register_tokens=num_register, hidden_size=hidden)
            feats.append(out["patch"].mean(axis=1))
            cls.append(out["cls"])
        torch.save(
            {
                "features": torch.from_numpy(np.concatenate(feats).reshape(len(eps), 1, -1).copy()),
                "cls": torch.from_numpy(np.concatenate(cls).reshape(len(eps), 1, -1).copy()),
                "episode_ids": torch.tensor(eps),
                "domain": args.domain,
                "split": split,
            },
            args.output_dir / f"{args.domain}_{split}.pt",
        )
        manifest["splits"][split] = {"episodes": len(eps)}
        print(f"{args.domain}/{split}: {len(eps)} episodes saved", flush=True)

    (args.output_dir / "manifest_deep.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
