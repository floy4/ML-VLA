#!/usr/bin/env python3
"""Weight-space evaluation of a hypernet checkpoint over a domain set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def nearest_oracle_hits(predictions, eval_names, pool_names, domain_targets, scales):
    """Count eval predictions whose closest oracle (delta cosine) is its own domain's.

    The comparison pool is ALL rung oracles (splits.yaml train + test for the rung),
    per the design spec — not just the eval set's own oracles.
    """
    from mlvla.meta.hypernet.evaluation import evaluate_direct_ab

    hits = 0
    for name_i, pred_i in zip(eval_names, predictions):
        best_name, best_cos = None, -2.0
        for name_j in pool_names:
            metrics = evaluate_direct_ab(pred_i, domain_targets[name_j], scales)
            cos = sum(r["delta_cosine"] for r in metrics) / len(metrics)
            if cos > best_cos:
                best_cos, best_name = cos, name_j
        if best_name == name_i:
            hits += 1
    return hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rung", type=str, required=True)
    parser.add_argument("--eval-domains", choices=("test", "train"), default="test")
    parser.add_argument("--features", type=Path, default=Path(__import__("mlvla.paths", fromlist=["PATHS"]).PATHS["feature_cache_dir"]))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pooling", choices=("patch", "cls"), default=None)
    parser.add_argument("--splits", type=Path, default=REPO_ROOT / "configs" / "splits_view.yaml")
    args = parser.parse_args()

    import torch
    import yaml
    from mlvla.meta.hypernet import DirectABHyperNetwork
    from mlvla.meta.hypernet.evaluation import evaluate_direct_ab
    from mlvla.meta.view_params import append_view_params, view_vector_for_domain
    from mlvla.meta.weights.targets import load_direct_targets, selected_rows

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    domains = yaml.safe_load((REPO_ROOT / "configs" / "domains.yaml").read_text())
    splits = yaml.safe_load(args.splits.read_text())
    eval_domains = splits[args.rung][args.eval_domains]
    train_domains = checkpoint["domains"]
    # nearest-oracle comparison pool: the rung's FULL oracle set (train ∪ test);
    # for rungs 3/4 this correctly excludes c3 since the rung never saw it.
    pool_domains = list(dict.fromkeys(splits[args.rung]["train"] + splits[args.rung]["test"]))
    rows = [{"source_index": m["source_index"]} for m in checkpoint["modules"]]
    oracle_paths = {d: Path(domains[d]["canonical_npz"]) for d in pool_domains}
    domain_targets, shapes, _ = load_direct_targets(oracle_paths, rows)
    scales = checkpoint["scales"]

    if checkpoint.get("evidence") == "dino_film":
        from mlvla.meta.hypernet import FiLMABHyperNetwork
        model = FiLMABHyperNetwork(**checkpoint["model_args"])
    else:
        model = DirectABHyperNetwork(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    domain_targets = {
        d: {m: {f: t.to(device) for f, t in factors.items()} for m, factors in mods.items()}
        for d, mods in domain_targets.items()
    }
    key = args.pooling or checkpoint.get("pooling", "patch")
    key = "features" if key == "patch" else "cls"

    per_domain, all_pred, all_names = {}, [], []
    with torch.inference_mode():
        for domain in eval_domains:
            blob = torch.load(args.features / f"{domain}_{'val' if args.eval_domains == 'train' else 'test'}.pt",
                              map_location="cpu")
            evidence = blob[key].to(device)
            if checkpoint.get("evidence") == "dino_view":
                evidence = append_view_params(evidence, domain)
            if checkpoint.get("evidence") == "dino_film":
                view = torch.tensor(view_vector_for_domain(domain), device=device).expand(evidence.shape[0], -1)
                prediction = model(evidence, view)
            else:
                prediction = model(evidence)
            mean_prediction = {k: {f: t.mean(dim=0, keepdim=True) for f, t in factors.items()}
                               for k, factors in prediction.items()}
            metrics = evaluate_direct_ab(mean_prediction, domain_targets[domain], scales)
            per_domain[domain] = {
                "delta_cosine": sum(r["delta_cosine"] for r in metrics) / len(metrics),
                "relative_A_error": sum(r["relative_A_error"] for r in metrics) / len(metrics),
                "relative_B_error": sum(r["relative_B_error"] for r in metrics) / len(metrics),
                "relative_delta_error": sum(r["relative_delta_error"] for r in metrics) / len(metrics),
            }
            all_pred.append(mean_prediction)
            all_names.append(domain)

    # nearest-oracle: is each prediction closest to its OWN oracle among ALL rung oracles?
    nearest_hits = nearest_oracle_hits(all_pred, all_names, pool_domains, domain_targets, scales)
    report = {
        "checkpoint": str(args.checkpoint), "rung": args.rung, "eval_domains": args.eval_domains,
        "evidence_split": "val" if args.eval_domains == "train" else "test",
        "per_domain": per_domain,
        "mean_delta_cosine": sum(v["delta_cosine"] for v in per_domain.values()) / len(per_domain),
        "nearest_oracle_accuracy": nearest_hits / len(all_names),
        "nearest_oracle_pool_size": len(pool_domains),
    }
    output = args.output or (
        REPO_ROOT / "outputs" / "eval" /
        f"{args.rung}_{Path(args.checkpoint).stem}_{args.eval_domains}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"mean_delta_cosine={report['mean_delta_cosine']:.6f} "
          f"nearest_oracle={report['nearest_oracle_accuracy']:.3f} -> {output}")


if __name__ == "__main__":
    main()
