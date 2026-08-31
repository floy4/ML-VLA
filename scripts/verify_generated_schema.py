#!/usr/bin/env python3
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
from mlvla.meta.weights.lora_io import assert_compatible, load_canonical


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify canonical file round-trip before OpenPI")
    parser.add_argument("--source", choices=("onehot", "dino"), default="dino")
    parser.add_argument("--root", type=Path, default=Path(__import__("mlvla.paths", fromlist=["PATHS"]).PATHS["generated_lora_root"]))
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="template canonical npz (default: first generated file's domain oracle from domains.yaml)",
    )
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "eval" / "generated_schema.json")
    args = parser.parse_args()
    rows = []
    with load_canonical(args.template) as template:
        for domain in ("clean", "view"):
            path = args.root / args.source / domain / "params.canonical.npz"
            with load_canonical(path) as generated:
                assert_compatible(template, generated)
                nonzero = sum(bool(module.B.any()) for module in generated.modules())
                rows.append(
                    {
                        "domain": domain,
                        "path": str(path),
                        "module_count": len(generated),
                        "nonzero_modules": nonzero,
                        "metadata": generated.manifest.get("metadata", {}),
                    }
                )
    report = {"file_roundtrip_passed": True, "openpi_policy_roundtrip": "pending", "adapters": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
