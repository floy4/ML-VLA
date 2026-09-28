#!/usr/bin/env python3
"""Probe: does instance-first-frame DINO evidence change the generated LoRA?

Compares, per domain, (a) the export-time condition (cache test-split mean,
per-episode forward then averaged — export semantics) against (b) the DINO
patch-mean feature of each rendered test-instance first frame, measuring
rel-L2 of the flat predicted LoRA (all selected modules' A+B concatenated).
Reference scale: cross-domain rel-L2 among the cache conditions (the known
condition-collapse band, ~1e-4..1e-3).

Run in an env with transformers>=4.53 (DINOv3), e.g. `simpler`.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

SRC = pathlib.Path("/home/zhy/vla/ML-VLA/src")
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlvla.meta.dinov3_encoder import encode_frames, load_dinov3
from mlvla.meta.view_params import append_view_params

CKPT = "/data2/zhy/meta_lora_offload/hypernet/e2e_concat_level_mt_expand/step20000.pt"
DINOV3 = "/data2/zhy/models/dinov3-vitl16-pretrain-lvd1689m"
CACHE = pathlib.Path("/home/zhy/vla/meta_lora/outputs/feature_cache_level")
DEVICE = "cuda:0"


def flat_lora(pred: dict, keys: list[str]) -> torch.Tensor:
    """Sum over the batch axis first, then flatten — chunks of unequal size
    accumulate correctly and equal-size chunks can't silently elementwise-add."""
    parts = []
    for k in keys:
        parts.append(pred[k]["A"].sum(0).reshape(-1))
        parts.append(pred[k]["B"].sum(0).reshape(-1))
    return torch.cat(parts)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=pathlib.Path, required=True,
                        help="NPZ from probe_render_firstframes.py")
    parser.add_argument("--checkpoint", default=CKPT)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    assert ckpt.get("evidence") == "dino_view_v4", ckpt.get("evidence")
    from mlvla.meta.hypernet import V4DirectABHyperNetwork
    model = V4DirectABHyperNetwork(**ckpt["model_args"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(DEVICE)
    pose_cfg = ckpt.get("pose_feature")
    keys = [row["key"] for row in ckpt["modules"]]

    dino, hidden, num_register, _ = load_dinov3(DINOV3, device=DEVICE)
    dino = dino.to(torch.float32)

    # (a) cache conditions per domain (export semantics: per-episode forward, mean)
    cache_vec = {}
    domains = sorted({r["job"] for r in json.loads(str(np.load(args.frames)["meta"]))})
    for domain in domains:
        blob = torch.load(CACHE / f"{domain}_test.pt", map_location="cpu")
        ev = blob["features"].to(DEVICE)  # [N,4,1024]
        n = ev.shape[0]
        acc = None
        for s in range(0, n, 32):
            cond = append_view_params(ev[s:s + 32], domain, pose_cfg)
            pred = model(cond)
            v = flat_lora(pred, keys)
            acc = v if acc is None else acc + v
        cache_vec[domain] = acc / n
        print(f"[cache] {domain}: n={n} episodes, flat |v|={float(cache_vec[domain].norm()):.2f}")

    # (b) instance first-frame conditions
    frames_npz = np.load(args.frames)
    rows = json.loads(str(frames_npz["meta"]))
    inst_vec = {}
    for r in rows:
        img = frames_npz[r["key"]]  # uint8 HWC, orientation already matched
        out = encode_frames(dino, [img], device=DEVICE,
                            num_register_tokens=num_register, hidden_size=hidden)
        feat = torch.from_numpy(out["patch"].mean(axis=1)).to(DEVICE)  # [1,1024]
        cond = append_view_params(feat, r["job"], pose_cfg)  # [1,1,1031]
        inst_vec[r["key"]] = flat_lora(model(cond), keys)

    def rel(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a - b).norm() / a.norm())

    # report
    report = {"checkpoint": args.checkpoint, "domains": {}}
    print("\n=== instance-first-frame vs cache-mean (rel-L2, per instance) ===")
    for domain in domains:
        vals = [rel(cache_vec[domain], inst_vec[k]) for k in inst_vec if k.startswith(domain + "|")]
        report["domains"][domain] = {
            "vs_cache": vals,
            "vs_cache_mean": float(np.mean(vals)),
            "vs_cache_max": float(np.max(vals)),
        }
        per = ", ".join(f"{v:.5f}" for v in vals)
        print(f"{domain}: mean={np.mean(vals):.5f} max={np.max(vals):.5f}  [{per}]")

    print("\n=== cross-instance spread within domain (pairwise rel-L2) ===")
    for domain in domains:
        ks = [k for k in inst_vec if k.startswith(domain + "|")]
        pairs = [rel(inst_vec[a], inst_vec[b]) for i, a in enumerate(ks) for b in ks[i + 1:]]
        if pairs:
            report["domains"][domain]["cross_instance_max"] = float(np.max(pairs))
            print(f"{domain}: max pairwise={np.max(pairs):.5f} (n={len(ks)} instances)")

    print("\n=== reference: cross-domain cache conditions (collapse band) ===")
    for i, a in enumerate(domains):
        for b in domains[i + 1:]:
            r = rel(cache_vec[a], cache_vec[b])
            print(f"{a} vs {b}: {r:.5f}")
            report[f"xdomain_{a}_vs_{b}"] = r

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n")
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
