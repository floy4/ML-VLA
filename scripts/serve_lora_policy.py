#!/usr/bin/env python3
"""M7: serve pi05_base with switchable LoRA variants over the OpenPI websocket protocol.

Serves `base` plus one variant per canonical adapter found under
--adapters-root (default: paths.yaml generated_lora_root, layout
<root>/<name>/params.canonical.npz — exactly what export_generated_lora.py
writes), plus any explicit --adapter NAME=PATH additions.

The client selects a variant by sending an extra ``"variant"`` key in the
observation dict; it is stripped before the LIBERO transform chain.  All
variants share a single JIT-compiled sample function; switching a variant
swaps the nnx state (merged base + LoRA delta) on the fly.

Run inside the openpi venv (single GPU):

  CUDA_VISIBLE_DEVICES=0 python scripts/serve_lora_policy.py --port 8123
"""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mlvla import paths as _paths
from mlvla.meta.openpi_bridge.merge import merge_canonical, tree_to_numpy
from mlvla.meta.weights.lora_io import load_canonical

CHECKPOINT = Path(_paths.get("base_checkpoint")) / "params"
_LIBERO_ASSETS_OVERRIDE = [None]  # set from --libero-assets in main()
_TASK_NAME = ["open_the_middle_drawer_of_the_cabinet"]  # set from --task in main()
# Only the norm_stats assets are needed from the pi05_libero checkpoint.
LIBERO_ASSETS = Path(_paths.get("base_checkpoint")).parent / "pi05_libero"


def variant_paths(adapters_root: Path | None, explicit: list[str]) -> dict[str, Path | None]:
    """base + one variant per <adapters_root>/<name>/params.canonical.npz, plus NAME=PATH overrides."""
    paths: dict[str, Path | None] = {"base": None}
    if adapters_root and adapters_root.is_dir():
        for child in sorted(adapters_root.iterdir()):
            npz = child / "params.canonical.npz" if child.is_dir() else None
            if npz and npz.exists():
                paths[child.name] = npz
    for pair in explicit:
        name, _, path = pair.partition("=")
        if not path:
            raise SystemExit(f"--adapter expects NAME=PATH, got {pair!r}")
        if not Path(path).exists():
            raise SystemExit(f"adapter npz missing: {path}")
        paths[name] = Path(path)
    return paths


def build_transforms(libero_assets: Path | None = None):
    """Replicate openpi.policies.policy_config.create_trained_policy for pi05_libero."""
    from openpi.training import checkpoints as _checkpoints
    from openpi.training import config as _config
    from openpi import transforms as _transforms

    train_config = _config.get_config("pi05_libero")
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats = _checkpoints.load_norm_stats((libero_assets or LIBERO_ASSETS) / "assets", data_config.asset_id)
    transforms = [
        _transforms.InjectDefaultPrompt(None),
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]
    output_transforms = [
        *data_config.model_transforms.outputs,
        _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
    ]
    return _transforms.compose(transforms), _transforms.compose(output_transforms)


class MultiVariantPolicy:
    """BasePolicy-compatible wrapper: `infer(obs) -> {"actions": ...}`."""

    def __init__(self, state, host_trees: dict, jitted_sample, variants: list[str]):
        import jax
        import jax.numpy as jnp

        self._jax, self._jnp = jax, jnp
        self._host_trees = host_trees  # per-variant numpy trees (host RAM)
        self._state = state  # single nnx State, re-filled on variant switch
        self._sample = jitted_sample
        self._variants = variants
        self._current = None
        self._input_transform, self._output_transform = build_transforms(_LIBERO_ASSETS_OVERRIDE[0])
        self._rng = jax.random.key(0)
        self._switch(variants[0])

    def _switch(self, variant: str) -> None:
        if variant not in self._host_trees:
            raise KeyError(f"unknown variant {variant!r}; expected one of {self._variants}")
        self._state.replace_by_pure_dict(
            self._jax.tree.map(self._jnp.asarray, self._host_trees[variant])
        )
        self._current = variant

    @property
    def metadata(self) -> dict:
        return {
            "task": _TASK_NAME[0],
            "base": "pi05_base",
            "variants": self._variants,
            "note": "send {'variant': name, ...obs} to select the served adapter",
        }

    def infer(self, obs: dict) -> dict:
        obs = dict(obs)
        variant = obs.pop("variant", None) or self._current
        if variant != self._current:
            self._switch(variant)

        inputs = self._input_transform(obs)
        inputs = self._jax.tree.map(lambda x: self._jnp.asarray(x)[None, ...], inputs)
        self._rng, rng = self._jax.random.split(self._rng)
        from openpi.models import model as _model

        observation = _model.Observation.from_dict(inputs)
        outputs = {
            "state": inputs["state"],
            "actions": self._sample(self._state, observation, rng),
        }
        outputs = self._jax.tree.map(lambda x: self._xfer(x), outputs)
        return self._output_transform(outputs)

    def _xfer(self, x):
        import numpy as np

        return np.asarray(x[0, ...])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--adapters-root", type=Path,
                        default=Path(_paths.get("generated_lora_root")),
                        help="serve every <root>/<name>/params.canonical.npz as variant <name>")
    parser.add_argument("--adapter", action="append", default=[],
                        metavar="NAME=PATH",
                        help="explicit canonical npz variant, repeatable")
    parser.add_argument("--task", default="open_the_middle_drawer_of_the_cabinet",
                        help="task name advertised in server metadata")
    parser.add_argument("--libero-assets", type=Path, default=LIBERO_ASSETS,
                        help="pi05_libero checkpoint dir for norm_stats assets")
    parser.add_argument("--num-steps", type=int, default=10, help="flow matching steps")
    parser.add_argument("--action-horizon", type=int, default=10)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    from flax import nnx
    from openpi.models import model as _model
    from openpi.models import model as model_lib
    from openpi.models.pi0_config import Pi0Config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    global _LIBERO_ASSETS_OVERRIDE
    _LIBERO_ASSETS_OVERRIDE[0] = args.libero_assets
    _TASK_NAME[0] = args.task
    paths = variant_paths(args.adapters_root, args.adapter)
    print(f"restoring {CHECKPOINT} as {args.dtype}")
    params = model_lib.restore_params(CHECKPOINT, dtype=getattr(jnp, args.dtype))
    base_np = tree_to_numpy(params)

    # merge every variant on the host (LoRA deltas are computed in float32)
    host_trees: dict[str, dict] = {"base": base_np}
    for name, path in paths.items():
        if path is None:
            continue
        with load_canonical(path) as lora:
            host_trees[name], stats = merge_canonical(base_np, lora)
        stats = {k: v for k, v in stats.items() if k != "modules"}
        print(f"merged {name}: {stats}")

    config = Pi0Config(pi05=True, dtype=args.dtype, action_horizon=args.action_horizon)
    model = config.load(params)
    graphdef, state = nnx.split(model)

    @jax.jit
    def sample(state_arg, obs, rng):
        module = nnx.merge(graphdef, state_arg)
        return module.sample_actions(rng, obs, num_steps=args.num_steps)

    variants = list(paths)
    policy = MultiVariantPolicy(state, host_trees, sample, variants)
    print(f"serving variants {variants} on {args.host}:{args.port}")
    server = WebsocketPolicyServer(policy, args.host, args.port, metadata=policy.metadata)
    server.serve_forever()


if __name__ == "__main__":
    main()
