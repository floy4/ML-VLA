#!/usr/bin/env python3
"""View-augmented hypernet training: evidence = DINO features ⊕ 7-d camera-pose params."""
from __future__ import annotations

import argparse
from pathlib import Path

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def first_occurrence_rows(labels, unique):
    first = {}
    for row, label in enumerate(labels):
        first.setdefault(int(label), row)
    return [first[u] for u in unique]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "hypernet_view.yaml")
    parser.add_argument("--rung", type=str, required=True)
    parser.add_argument("--domains", type=Path, default=REPO_ROOT / "configs" / "domains.yaml")
    parser.add_argument("--splits", type=Path, default=REPO_ROOT / "configs" / "splits_view.yaml")
    parser.add_argument("--features", type=Path, default=Path(__import__("mlvla.paths", fromlist=["PATHS"]).PATHS["feature_cache_dir"]))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--pooling", choices=("patch", "cls"), default="patch")
    parser.add_argument("--early-stop-cos", type=float, default=0.9999)
    parser.add_argument("--plateau-eps", type=float, default=1e-6)
    args = parser.parse_args()

    import torch
    import yaml
    from mlvla.meta.hypernet import DirectABHyperNetwork
    from mlvla.meta.hypernet.losses import reconstruction_loss
    from mlvla.meta.view_params import append_view_params
    from mlvla.meta.weights.normalization import compute_rms_scales, scales_to_dict
    from mlvla.meta.weights.targets import load_direct_targets, selected_rows

    config = yaml.safe_load(args.config.read_text())
    domains = yaml.safe_load(args.domains.read_text())
    splits = yaml.safe_load(args.splits.read_text())
    train_domains = splits[args.rung]["train"]
    rows = selected_rows(REPO_ROOT / f"configs/selected_modules_{args.rung}.json",
                         config["weight_target"]["max_modules"])
    oracle_paths = {d: Path(domains[d]["canonical_npz"]) for d in train_domains}
    domain_targets, shapes, modules = load_direct_targets(oracle_paths, rows)
    scales = scales_to_dict(compute_rms_scales(domain_targets))

    key = "features" if args.pooling == "patch" else "cls"

    def load_split(split):
        xs, ys = [], []
        for index, domain in enumerate(train_domains):
            blob = torch.load(args.features / f"{domain}_{split}.pt", map_location="cpu")
            xs.append(append_view_params(blob[key], domain))
            ys.extend([index] * blob[key].shape[0])
        return torch.cat(xs), torch.tensor(ys)

    train_x, train_y = load_split("train")
    val_x, val_y = load_split("val")
    hyper = config["hypernet"]
    model_args = {
        "condition_dim": train_x.shape[-1], "module_shapes": shapes,
        "condition_hidden_dim": hyper["condition_hidden_dim"],
        "hidden_dim": hyper["hidden_dim"], "head_bottleneck": hyper["head_bottleneck"],
    }
    torch.manual_seed(0)
    model = DirectABHyperNetwork(**model_args)
    device = torch.device("cuda")
    model = model.to(device)
    train_x, train_y = train_x.to(device), train_y.to(device)
    val_x = val_x.to(device)
    domain_targets = {
        d: {key_: {f: t.to(device) for f, t in factors.items()} for key_, factors in m.items()}
        for d, m in domain_targets.items()
    }

    def grouped_targets(unique):
        return {
            key_: {f: torch.stack([domain_targets[train_domains[i]][key_][f] for i in unique])
                   for f in ("A", "B")}
            for key_ in shapes
        }

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["lr"]),
                                  weight_decay=float(config["training"]["weight_decay"]))
    max_steps = args.steps or int(config["training"]["max_steps"])
    batch_size = int(config["training"]["batch_size"])
    delta_weight = float(config["loss"]["delta_w"])
    generator = torch.Generator(device="cpu").manual_seed(0)
    history, best_cosine, best_state = [], -1.0, None
    for step in range(1, max_steps + 1):
        idx = torch.randint(len(train_x), (batch_size,), generator=generator).to(device)
        x, y = train_x[idx], train_y[idx]
        unique = sorted(set(int(v) for v in y.tolist()))
        predictions = model(x)
        rows_keep = first_occurrence_rows(y.tolist(), unique)
        predictions_unique = {key_: {f: t[rows_keep] for f, t in factors.items()}
                              for key_, factors in predictions.items()}
        targets_unique = grouped_targets(unique)
        loss, parts = reconstruction_loss(predictions_unique, targets_unique, scales, delta_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == max_steps:
            with torch.inference_mode():
                model.eval()
                unique_val = sorted(set(int(v) for v in val_y.tolist()))
                keep = first_occurrence_rows(val_y.tolist(), unique_val)
                from mlvla.meta.hypernet.evaluation import evaluate_direct_ab
                cos_sum, cos_count = 0.0, 0
                for s in range(0, len(keep), 32):
                    rows = keep[s:s + 32]
                    preds = model(val_x[rows])
                    metrics = evaluate_direct_ab(preds, grouped_targets(unique_val[s:s + 32]), scales)
                    cos_sum += sum(r["delta_cosine"] for r in metrics)
                    cos_count += len(metrics)
                    del preds, metrics
                mean_cosine = cos_sum / cos_count
                model.train()
            history.append({"step": step, "loss": float(loss.detach()),
                            "val_delta_cosine": mean_cosine})
            print(f"step {step:5d} loss={float(loss.detach()):.8f} val_delta_cos={mean_cosine:.6f}", flush=True)
            if mean_cosine > best_cosine:
                best_cosine, best_state = mean_cosine, {k: v.detach().clone() for k, v in model.state_dict().items()}
            if best_cosine > args.early_stop_cos and step >= 500:
                print(f"early stop: val delta cosine > {args.early_stop_cos}")
                break
            if len(history) >= 20 and abs(history[-1]["val_delta_cosine"] - history[-10]["val_delta_cosine"]) < args.plateau_eps:
                print("early stop: val delta cosine plateau")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    output = args.output or REPO_ROOT / f"outputs/hypernet/{args.rung}/view.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "representation": "direct_ab", "source": "dino_view", "evidence": "dino_view",
        "state_dict": model.state_dict(),
        "model_args": model_args, "scales": scales, "modules": modules, "history": history,
        "oracle_paths": {d: str(p) for d, p in oracle_paths.items()}, "domains": train_domains,
        "feature_root": str(args.features), "pooling": args.pooling,
        "decoder_frozen": False, "rung": args.rung,
    }, output)
    print(f"wrote {output} (best val_delta_cosine={best_cosine:.6f})")


if __name__ == "__main__":
    main()
