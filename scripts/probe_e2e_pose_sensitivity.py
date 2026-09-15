#!/usr/bin/env python3
"""Weight-cosine pose sensitivity of a trained e2e_concat hypernet checkpoint.

Loads step60000.pt, rebuilds V4DirectABHyperNetwork from the checkpoint's own
``model_args`` (the exact constructor kwargs training saved), and generates
LoRA weights from val evidence (EvidenceBank.domain_mean(d, "val")) under
three view-vector variants per train domain:

  (1) original 7-dim cond vector  (bank.view(d), = view_params.view_vector)
  (2) zeroed 7-dim cond vector
  (3) swapped to another train domain's 7-dim cond vector

The comparison vector per condition is the flattened concatenation of every
module's generated A and B tensors (mean over the k evidence rows first —
the same reduction training deploys; scale factors are per-key constants and
cosine is reported both raw and scale-weighted).

Readings:
  cos(1,2) > 0.999  -> the pose channel is effectively ignored by the model.
  cos(1,3) high     -> cross-condition insensitivity (swapping the pose of a
                       DIFFERENT camera condition changes nothing).
Per-key cosine distributions (mean/median/p5/min over the 456 modules) show
whether sensitivity hides in a few modules.

Condition layout mirrors train_bridge's concat branch exactly:
  cond = cat([evidence(k,4,1024), view.expand(k,4,7)], dim=-1)  -> [k,4,1031]
(the model frame-means internally).

Usage:
  CUDA_VISIBLE_DEVICES=2 conda run -n openvla python scripts/probe_e2e_pose_sensitivity.py \
      --checkpoint /data2/zhy/meta_lora_offload/hypernet/e2e_concat/step60000.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch
import yaml

from mlvla.meta.e2e.evidence import EvidenceBank
from mlvla.meta.hypernet import V4DirectABHyperNetwork

DEFAULT_CKPT = Path("/data2/zhy/meta_lora_offload/hypernet/e2e_concat/step60000.pt")
DEFAULT_CONFIG = REPO_ROOT / "configs" / "hypernet_e2e_concat.yaml"
DEFAULT_OUT = Path("/data2/zhy/meta_lora_offload/pose_cache/e2e_pose_weightcos.json")
VIEW_DIM = 7


def flat_weights(model: V4DirectABHyperNetwork, ev: torch.Tensor,
                 view: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Forward one view variant; return (global flat vector, per-key flat
    vectors). Mean over the k evidence rows first (training's deployment
    reduction)."""
    k, n_frames, dino = ev.shape
    cond = torch.cat([ev, view.view(1, 1, VIEW_DIM).expand(k, n_frames, VIEW_DIM)], dim=-1)
    with torch.inference_mode():
        pred = model(cond)
    keys = model.decoder.module_list
    per_key, chunks = [], []
    for key in keys:
        a = pred[key]["A"].mean(0).reshape(-1)
        b = pred[key]["B"].mean(0).reshape(-1)
        per_key.append(torch.cat([a, b]))
        chunks.append(per_key[-1])
    return torch.cat(chunks), per_key


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def per_key_stats(fa: list[torch.Tensor], fb: list[torch.Tensor]) -> dict:
    cs = torch.tensor([cos(a, b) for a, b in zip(fa, fb)])
    q = torch.quantile(cs, torch.tensor([0.05, 0.5]))
    return {"mean": round(float(cs.mean()), 6), "median": round(float(q[1]), 6),
            "p5": round(float(q[0]), 6), "min": round(float(cs.min()), 6)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    train_domains = list(cfg["train_domains"])

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert ck["variant"] == "concat", f"expected concat checkpoint, got {ck['variant']}"
    assert list(ck["domains"]) == train_domains, "checkpoint/config domain mismatch"
    model = V4DirectABHyperNetwork(**ck["model_args"])
    model.load_state_dict(ck["state_dict"])
    model.to(args.device).eval()
    keys = model.decoder.module_list

    # per-key scales (physical LoRA = pred * scale); cosine weighting variant
    scales = ck["scales"]
    sizes_a = [int(np.prod(ck["model_args"]["module_shapes"][k]["A"])) for k in keys]
    sizes_b = [int(np.prod(ck["model_args"]["module_shapes"][k]["B"])) for k in keys]
    key_scale = torch.cat([torch.cat([
        torch.full((na,), scales[k]["A"], device=args.device),
        torch.full((nb,), scales[k]["B"], device=args.device)])
        for k, na, nb in zip(keys, sizes_a, sizes_b)])

    bank = EvidenceBank(Path(cfg["feature_cache_dir"]), train_domains)

    results = {}
    for d in train_domains:
        ev = bank.domain_mean(d, "val").to(args.device)
        v_orig = bank.view(d).to(args.device)
        v_zero = torch.zeros(VIEW_DIM, device=args.device)

        f1, pk1 = flat_weights(model, ev, v_orig)
        f2, pk2 = flat_weights(model, ev, v_zero)

        entry = {
            "n_val_episodes": int(ev.shape[0]),
            "cos_1_2_orig_vs_zero": round(cos(f1, f2), 6),
            "cos_1_2_scaled": round(cos(f1 * key_scale, f2 * key_scale), 6),
            "per_key_cos_1_2": per_key_stats(pk1, pk2),
            "cos_1_3_swapped": {},
            "cos_1_3_scaled": {},
        }
        for other in train_domains:
            if other == d:
                continue
            f3, _ = flat_weights(model, ev, bank.view(other).to(args.device))
            entry["cos_1_3_swapped"][other.split("__")[0]] = round(cos(f1, f3), 6)
            entry["cos_1_3_scaled"][other.split("__")[0]] = round(
                cos(f1 * key_scale, f3 * key_scale), 6)

        vals = list(entry["cos_1_3_swapped"].values())
        entry["cos_1_3_min"] = min(vals)
        entry["cos_1_3_mean"] = round(float(np.mean(vals)), 6)
        results[d.split("__")[0]] = entry
        print(f"{d.split('__')[0]:<22s} cos(1,2)={entry['cos_1_2_orig_vs_zero']:.6f} "
              f"(scaled {entry['cos_1_2_scaled']:.6f}, per-key mean "
              f"{entry['per_key_cos_1_2']['mean']:.6f} min "
              f"{entry['per_key_cos_1_2']['min']:.6f}) | cos(1,3) min="
              f"{entry['cos_1_3_min']:.6f} mean={entry['cos_1_3_mean']:.6f}", flush=True)

    mean12 = float(np.mean([r["cos_1_2_orig_vs_zero"] for r in results.values()]))
    mean13 = float(np.mean([r["cos_1_3_mean"] for r in results.values()]))
    verdict_ignored = mean12 > 0.999
    summary = {
        "checkpoint": str(args.checkpoint),
        "mean_cos_1_2": round(mean12, 6),
        "mean_cos_1_3": round(mean13, 6),
        "pose_channel_ignored(cos12>0.999)": verdict_ignored,
        "protocol": ("val evidence domain_mean(d,'val'); flat = concat over "
                     f"{len(keys)} modules of mean-over-k generated A||B; raw "
                     "cosine primary, scale-weighted secondary"),
    }
    print(json.dumps(summary, indent=2), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "per_domain": results}, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
