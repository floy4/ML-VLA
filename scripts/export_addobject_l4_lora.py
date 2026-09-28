#!/usr/bin/env python3
"""Export the add_object_L4 (constructed +8..10 OOD) LoRA from rendered t=0 frames.

No demo dataset exists for L4, so evidence = env-reset first frames
(render_addobject_l4_evidence.py, openvla env). This script (qwen3vl env for
DINOv3) encodes them with the SAME encoder/pooling as the training cache,
conditions the checkpoint's hypernet with add_object's neutral pose vector,
and writes params.canonical.npz via the add_object_L1 template.

Orientation: pass --flip if the orientation check showed max_chr add_object
videos are rot180 vs local EGL (mirrors the lighting/noise/texture rule).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True,
                    help="dir with <suite>.npy from the render script")
    ap.add_argument("--output", type=Path, required=True,
                    help="params.canonical.npz path")
    ap.add_argument("--flip", action="store_true",
                    help="rot180 rendered frames before DINO")
    ap.add_argument("--template-domain", default="add_object_L1")
    args = ap.parse_args()

    import numpy as np
    import torch
    import yaml
    from mlvla.meta.dinov3_encoder import encode_frames, load_dinov3
    from mlvla.meta.hypernet import V4DirectABHyperNetwork
    from mlvla.meta.view_params import append_view_params, pose_vector_from_config
    from mlvla.meta.weights.lora_io import load_canonical, save_canonical

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert ckpt.get("evidence") == "dino_view_v4", ckpt.get("evidence")
    domains = yaml.safe_load((REPO_ROOT / "configs" / "domains.yaml").read_text())
    device = "cuda"

    frames = []
    for suite in ["libero_goal", "libero_10", "libero_object", "libero_spatial"]:
        p = args.frames / f"{suite}.npy"
        frames.append(np.load(p))
    frames = np.concatenate(frames)  # [N,256,256,3] uint8
    if args.flip:
        frames = frames[:, ::-1, ::-1]
    print(f"frames: {frames.shape} flip={args.flip}", flush=True)

    model_dino, _, _, _ = load_dinov3(
        __import__("mlvla.paths", fromlist=["PATHS"]).PATHS["dinov3_model"], device)
    feats = []
    for s in range(0, len(frames), 32):
        out = encode_frames(model_dino, frames[s:s + 32], device)
        feats.append(out["patch"].mean(axis=1))
    evidence = torch.from_numpy(np.concatenate(feats)).float()  # [N,1024]
    print(f"evidence: {evidence.shape}", flush=True)

    net = V4DirectABHyperNetwork(**ckpt["model_args"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval().to(device)

    pose_cfg = ckpt.get("pose_feature")
    evidence = append_view_params(evidence.unsqueeze(1), args.template_domain, pose_cfg)
    n = evidence.shape[0]
    selected = {row["source_index"]: row["key"] for row in ckpt["modules"]}
    sum_a = {k: None for k in selected.values()}
    sum_b = {k: None for k in selected.values()}
    with torch.inference_mode():
        for s in range(0, n, 32):
            p = net(evidence[s:s + 32].to(device))
            for k, ab in p.items():
                if k not in sum_a:
                    continue
                sa, sb = ab["A"].sum(0), ab["B"].sum(0)
                sum_a[k] = sa if sum_a[k] is None else sum_a[k] + sa
                sum_b[k] = sb if sum_b[k] is None else sum_b[k] + sb
            del p
    prediction = {k: {"A": sum_a[k] / n, "B": sum_b[k] / n} for k in sum_a}

    arrays = {}
    template_path = domains[args.template_domain]["canonical_npz"]
    with load_canonical(template_path) as template:
        for module_index in range(len(template)):
            shape = template.module(module_index)
            if module_index in selected:
                module_key = selected[module_index]
                arrays[f"module_{module_index}_A"] = (
                    prediction[module_key]["A"].cpu().numpy()
                    * ckpt["scales"][module_key]["A"]).astype(np.float32)
                arrays[f"module_{module_index}_B"] = (
                    prediction[module_key]["B"].cpu().numpy()
                    * ckpt["scales"][module_key]["B"]).astype(np.float32)
            else:
                arrays[f"module_{module_index}_A"] = shape.A.copy()
                arrays[f"module_{module_index}_B"] = np.zeros_like(shape.B)
        manifest = dict(template.manifest)
    manifest["metadata"] = {
        **manifest.get("metadata", {}),
        "generated_by": "V4DirectABHyperNetwork",
        "conditioning": ckpt["source"],
        "domain": "add_object_L4",
        "rung": f"{args.frames.name}+flip" if args.flip else args.frames.name,
        "generated_module_count": len(selected),
        "unselected_modules": "zero_B",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_canonical(args.output, manifest, arrays)
    print(f"WROTE {args.output} ({len(arrays) // 2} modules, evidence n={n})",
          flush=True)


if __name__ == "__main__":
    main()
