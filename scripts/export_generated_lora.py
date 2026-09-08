#!/usr/bin/env python3
"""Export generated canonical LoRA npz per eval domain from an N-domain checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rung", type=str, required=True)
    parser.add_argument("--eval-domains", choices=("test", "train"), default="test")
    parser.add_argument("--features", type=Path, default=Path(__import__("mlvla.paths", fromlist=["PATHS"]).PATHS["feature_cache_dir"]))
    parser.add_argument("--output-dir", type=Path, default=Path(__import__("mlvla.paths", fromlist=["PATHS"]).PATHS["generated_lora_root"]))
    parser.add_argument("--pooling", choices=("patch", "cls"), default=None)
    parser.add_argument("--splits", type=Path, default=REPO_ROOT / "configs" / "splits_view.yaml")
    parser.add_argument(
        "--output-subdir",
        type=str,
        default=None,
        help="Subdir under output-dir for exports; defaults to --rung (backward compatible)",
    )
    args = parser.parse_args()

    import numpy as np
    import torch
    import yaml
    from mlvla.meta.hypernet import DirectABHyperNetwork
    from mlvla.meta.view_params import append_view_params, view_vector_for_domain
    from mlvla.meta.weights.lora_io import load_canonical, save_canonical

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    domains = yaml.safe_load((REPO_ROOT / "configs" / "domains.yaml").read_text())
    splits = yaml.safe_load(args.splits.read_text())
    eval_domains = splits[args.rung][args.eval_domains]
    selected = {row["source_index"]: row["key"] for row in checkpoint["modules"]}
    key = args.pooling or checkpoint.get("pooling", "patch")
    key = "features" if key == "patch" else "cls"
    split = "val" if args.eval_domains == "train" else "test"

    if checkpoint.get("evidence") == "dino_film":
        from mlvla.meta.hypernet import FiLMABHyperNetwork
        model = FiLMABHyperNetwork(**checkpoint["model_args"])
    elif checkpoint.get("evidence") == "dino_film_shared":
        from mlvla.meta.hypernet import SharedFiLMABHyperNetwork
        model = SharedFiLMABHyperNetwork(**checkpoint["model_args"])
    elif checkpoint.get("evidence") == "dino_view_v4":
        from mlvla.meta.hypernet import V4DirectABHyperNetwork
        model = V4DirectABHyperNetwork(**checkpoint["model_args"])
    else:
        model = DirectABHyperNetwork(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for domain in eval_domains:
        template_path = domains[domain]["canonical_npz"]
        blob = torch.load(args.features / f"{domain}_{split}.pt", map_location="cpu")
        evidence = blob[key].to(device)
        with torch.inference_mode():
            if checkpoint.get("evidence") in ("dino_film", "dino_film_shared"):
                view = torch.tensor(view_vector_for_domain(domain), device=device).expand(evidence.shape[0], -1)
                prediction = model(evidence, view)
            else:
                if checkpoint.get("evidence") in ("dino_view", "dino_view_v4"):
                    evidence = append_view_params(evidence, domain)
                prediction = model(evidence)
        arrays = {}
        with load_canonical(template_path) as template:
            for module_index in range(len(template)):
                shape = template.module(module_index)
                if module_index in selected:
                    module_key = selected[module_index]
                    arrays[f"module_{module_index}_A"] = (
                        prediction[module_key]["A"].mean(dim=0).cpu().numpy()
                        * checkpoint["scales"][module_key]["A"]
                    ).astype(np.float32)
                    arrays[f"module_{module_index}_B"] = (
                        prediction[module_key]["B"].mean(dim=0).cpu().numpy()
                        * checkpoint["scales"][module_key]["B"]
                    ).astype(np.float32)
                else:
                    arrays[f"module_{module_index}_A"] = shape.A.copy()
                    arrays[f"module_{module_index}_B"] = np.zeros_like(shape.B)
            manifest = dict(template.manifest)
        manifest["metadata"] = {
            **manifest.get("metadata", {}),
            "generated_by": {
                "dino_film": "FiLMABHyperNetwork",
                "dino_film_shared": "SharedFiLMABHyperNetwork",
                "dino_view_v4": "V4DirectABHyperNetwork",
            }.get(checkpoint.get("evidence"), "DirectABHyperNetwork"),
            "conditioning": checkpoint["source"],
            "domain": domain,
            "rung": args.rung,
            "generated_module_count": len(selected),
            "unselected_modules": "zero_B",
        }
        output = args.output_dir / (args.output_subdir or args.rung) / domain / "params.canonical.npz"
        save_canonical(output, manifest, arrays)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
