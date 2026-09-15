#!/usr/bin/env python3
"""Offline VGGT CAMERA-token pose feature extraction per perturbed condition.

For each perturbed-camera condition we run ONE VGGT forward over
[clean_reference_frame, ep0_frame0, ..., epN_frame0] where the episode pool is
exactly the train-split episode_ids of the DINOv3 feature cache
(EvidenceBank.split_episode_ids(domain, "train")). Sharing a single forward
between the clean reference frame and all condition frames pins every
condition to the same gauge; VGGT additionally anchors frame 0 (the clean
reference) with its dedicated "first frame" camera token.

Saved npz schema (per condition <name>, no domain hash suffix):
  cond:<name>        float32 [n_layers=24, n_cond_frames, token_dim=2048]
                     CAMERA-token (token index 0) hidden states of every
                     alternating-attention layer pair, EXCLUDING the reference
                     frame. Channel layout: [0:1024] frame-branch output,
                     [1024:2048] global-branch output (torch.cat dim=-1).
  episode_ids:<name> int64   [n_cond_frames] (sorted ascending)
  frame_idx:<name>   int64   [n_cond_frames] (0 = first frame of episode)
  meta               json string (model, layer definition, ref provenance,
                     episode-pool source incl. any fallback decisions)

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n openvla python scripts/extract_pose_feature_vggt.py \
      --conditions clean,v1_azimuth30 --out /tmp/vggt_smoke.npz
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# 8 training conditions (all __de2a6ce7 domains) + 2 held-out OOD conditions
# (splits_view_full.yaml test split). The clean reference frame also comes from
# the clean__de2a6ce7 train split.
DEFAULT_CONDITIONS = [
    "c1_v1l1",
    "c2_v2l2",
    "l1_warm_dim",
    "l2_cool_bright",
    "v1_azimuth30",
    "v2_azimuth60",
    "v3_elev15_zoom125",
    "clean",
    "l3_directional_low",
    "c3_v3l3",
]
REF_CONDITION = "clean"
VGGT_COMMIT = "a288dd0f14786c93483e45524328726ab7b1b4ce"  # facebookresearch/vggt main 2026-05-18


def first_frames_from_parquet(dataset_root: Path, episode_ids: list[int]) -> dict[int, bytes]:
    """Decode frame 0 of every requested episode from a lerobot v3.0 dataset."""
    import pyarrow.parquet as pq

    wanted = set(episode_ids)
    found: dict[int, tuple[int, bytes]] = {}  # episode -> (frame_index, jpeg bytes)
    parquets = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"no parquet files under {dataset_root}/data")
    for parquet_path in parquets:
        parquet = pq.ParquetFile(parquet_path)
        columns = [c for c in ("episode_index", "frame_index", "image") if c in parquet.schema_arrow.names]
        if "episode_index" not in columns or "image" not in columns:
            raise RuntimeError(f"unexpected parquet schema {parquet.schema_arrow.names} in {parquet_path}")
        for batch in parquet.iter_batches(batch_size=256, columns=columns):
            for row in batch.to_pylist():
                episode = int(row["episode_index"])
                if episode not in wanted:
                    continue
                raw = row["image"].get("bytes") if isinstance(row["image"], dict) else None
                if not raw:
                    continue
                frame_idx = int(row.get("frame_index", 0))
                # keep the lowest frame_index row seen for this episode
                if episode not in found or frame_idx < found[episode][0]:
                    found[episode] = (frame_idx, raw)
    missing = wanted - set(found)
    if missing:
        raise RuntimeError(f"{dataset_root}: no image rows for episodes {sorted(missing)}")
    return {ep: found[ep] for ep in sorted(found)}


def preprocess(raw_jpeg: bytes, img_size: int):
    """vggt.utils.load_fn.load_and_preprocess_images-equivalent for square inputs.

    PIL bicubic resize to img_size x img_size, then uint8 HWC -> float CHW in
    [0, 1]. ImageNet mean/std normalization happens INSIDE the aggregator.
    """
    import numpy as np
    import torch
    from PIL import Image

    img = Image.open(io.BytesIO(raw_jpeg)).convert("RGB")
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size), Image.Resampling.BICUBIC)
    arr = np.array(img, dtype=np.uint8)  # writable copy (PIL buffers are read-only)
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


def load_vggt(model_id: str, device: str, n_layers: int):
    from vggt.models.vggt import VGGT

    model = VGGT.from_pretrained(model_id).to(device)
    model.eval()
    # main-branch default only caches layers (4, 11, 17, 23); we need all 24.
    model.aggregator.cached_layer_indices = set(range(n_layers))
    return model


def camera_tokens(model, images, device: str, n_layers: int):
    """One aggregator forward -> [n_layers, n_frames, token_dim] CAMERA tokens."""
    import torch

    assert images.shape[0] == 1 and images.shape[2] == 3
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        aggregated_tokens_list, ps_idx = model.aggregator(images)
    non_none = [t for t in aggregated_tokens_list if t is not None]
    if len(non_none) != n_layers:
        raise RuntimeError(f"expected {n_layers} cached layers, got {len(non_none)}")
    # token 0 of each frame is the CAMERA token; ps_idx=5 (1 camera + 4 registers)
    cam = torch.stack([t[0, :, 0, :].float() for t in aggregated_tokens_list])
    return cam, ps_idx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=str, default=",".join(DEFAULT_CONDITIONS),
                        help="comma-separated condition names (no domain hash suffix)")
    parser.add_argument("--out", type=Path, required=True, help="output npz path")
    parser.add_argument("--feature-cache-dir", type=Path,
                        default=Path("/home/zhy/vla/meta_lora/outputs/feature_cache_90"))
    parser.add_argument("--domains-yaml", type=Path, default=REPO_ROOT / "configs" / "domains.yaml")
    parser.add_argument("--domain-suffix", type=str, default="__de2a6ce7")
    parser.add_argument("--model-id", type=str, default="facebook/VGGT-1B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--n-layers", type=int, default=24)
    args = parser.parse_args()

    import numpy as np
    import torch
    import yaml

    from mlvla.meta.e2e.evidence import EvidenceBank

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    domains_cfg = yaml.safe_load(args.domains_yaml.read_text())
    suffix = args.domain_suffix

    def domain_of(cond: str) -> str:
        key = f"{cond}{suffix}"
        if key not in domains_cfg:
            raise KeyError(f"condition {cond!r}: domain {key!r} missing from {args.domains_yaml}")
        return key

    # ---- clean reference frame: clean domain, train split, sorted episode ids [0], frame 0
    ref_domain = domain_of(REF_CONDITION)
    ref_blob = args.feature_cache_dir / f"{ref_domain}_train.pt"
    ref_blob_ids = sorted(int(i) for i in torch.load(ref_blob, map_location="cpu")["episode_ids"].tolist())
    ref_episode_id = ref_blob_ids[0]
    ref_rows = first_frames_from_parquet(Path(domains_cfg[ref_domain]["dataset_root"]), [ref_episode_id])
    ref_frame_idx, ref_raw = ref_rows[ref_episode_id]
    ref_tensor = preprocess(ref_raw, args.img_size)
    print(f"reference frame: domain={ref_domain} episode={ref_episode_id} frame_idx={ref_frame_idx}",
          flush=True)

    model = load_vggt(args.model_id, args.device, args.n_layers)

    out: dict[str, np.ndarray] = {}
    meta = {
        "model": args.model_id,
        "vggt_install": f"editable /home/zhy/vla/vggt_src @ {VGGT_COMMIT} (pip install --no-deps)",
        "n_layers": args.n_layers,
        "layer_definition": (
            "output_list[i] = torch.cat([frame_branch_block_i_out, "
            "global_branch_block_i_out], dim=-1) at CAMERA token (token index 0; "
            "ps_idx=5: 1 camera + 4 register tokens precede patches)"
        ),
        "token_dim": 2 * 1024,
        "token_dim_layout": "[0:1024] frame-branch, [1024:2048] global-branch",
        "cached_layer_indices": "overridden to all 24 (main-branch default is {4,11,17,23})",
        "dtype": "computed under bf16 autocast, stored float32",
        "img_size": args.img_size,
        "preprocess": ("lerobot parquet image bytes -> PIL RGB bicubic resize 518x518 -> "
                       "float [0,1]; ImageNet normalization inside aggregator"),
        "reference_frame": {
            "domain": ref_domain,
            "split": "train",
            "selection": "train episode_ids sorted ascending, index 0",
            "episode_id": ref_episode_id,
            "frame_idx": ref_frame_idx,
            "dataset_root": domains_cfg[ref_domain]["dataset_root"],
        },
        "episode_pool": "feature_cache_90 <domain>_train.pt episode_ids (EvidenceBank.split_episode_ids)",
        "device": args.device,
        "gauge_note": ("reference frame is frame 0 of every forward; VGGT anchors frame 0 "
                       "with its dedicated first-frame camera token variant"),
        "conditions": {},
    }

    for cond in conditions:
        domain = domain_of(cond)
        t0 = time.time()

        pool_source = "feature_cache_train_split"
        blob = args.feature_cache_dir / f"{domain}_train.pt"
        if blob.exists():
            bank = EvidenceBank(args.feature_cache_dir, [domain])
            episode_ids = sorted(bank.split_episode_ids(domain, "train"))
            del bank
        else:  # fallback: all episodes present in the condition's parquet
            pool_source = "all_parquet_episodes (feature-cache blob missing)"
            import pyarrow.parquet as pq

            parquets = sorted((Path(domains_cfg[domain]["dataset_root"]) / "data")
                              .glob("chunk-*/file-*.parquet"))
            episode_ids = sorted({int(r["episode_index"])
                                  for p in parquets
                                  for r in pq.read_table(p, columns=["episode_index"]).to_pylist()})

        rows = first_frames_from_parquet(Path(domains_cfg[domain]["dataset_root"]), episode_ids)
        cond_frames = [preprocess(rows[ep][1], args.img_size) for ep in episode_ids]
        frame_idx = np.array([rows[ep][0] for ep in episode_ids], dtype=np.int64)

        # single forward: [clean_ref, condition episode frames]
        images = torch.stack([ref_tensor] + cond_frames).unsqueeze(0).to(args.device)
        cam, ps_idx = camera_tokens(model, images, args.device, args.n_layers)
        cam = cam[:, 1:, :].cpu().numpy()  # drop the reference frame -> [L, n_cond, D]

        if cam.shape != (args.n_layers, len(episode_ids), meta["token_dim"]):
            raise RuntimeError(f"{cond}: unexpected camera-token shape {cam.shape}")
        if not np.isfinite(cam).all():
            raise RuntimeError(f"{cond}: non-finite CAMERA-token values")

        out[f"cond:{cond}"] = cam.astype(np.float32)
        out[f"episode_ids:{cond}"] = np.array(episode_ids, dtype=np.int64)
        out[f"frame_idx:{cond}"] = frame_idx
        meta["conditions"][cond] = {
            "domain": domain,
            "n_frames": len(episode_ids),
            "episode_pool_source": pool_source,
            "wall_seconds": round(time.time() - t0, 2),
            "ps_idx": int(ps_idx),
        }
        print(f"{cond}: domain={domain} n_ep={len(episode_ids)} "
              f"cam={cam.shape} t={time.time() - t0:.1f}s pool={pool_source}", flush=True)

    import datetime

    meta["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    out["meta"] = np.array(json.dumps(meta, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    total = sum(c["wall_seconds"] for c in meta["conditions"].values())
    print(f"wrote {args.out} ({len(conditions)} conditions, {total:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
