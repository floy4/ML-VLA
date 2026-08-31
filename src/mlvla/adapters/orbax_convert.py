"""Convert Orbax Phase-1/2 LoRA checkpoint to canonical .npz format (458-module schema).

Usage:
    python scripts/convert_orbax_to_canonical.py \
        --orbax <expert_root>/<expert_id>/phase1/final/params \
        --output <same dir>/params.canonical.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mlvla import paths as _paths
from mlvla.adapters.canonical_lora import CanonicalLoRAAdapter, LoRAModule


def _restore_orbax(path: str) -> dict:
    """Restore Orbax checkpoint and flatten to stem→array dict."""
    from etils import epath
    import orbax.checkpoint as ocp

    with ocp.PyTreeCheckpointer() as ckptr:
        restored = ckptr.restore(epath.Path(path))

    def _flatten(d, prefix=""):
        result = {}
        for k, v in d.items():
            key = f"{prefix}/{k}" if prefix else k
            if isinstance(v, dict):
                result.update(_flatten(v, key))
            else:
                result[key] = np.asarray(v)
        return result

    return _flatten(restored)


def _factor_keys(flat: dict, stem: str) -> tuple[str, str]:
    """Find lora_a/lora_b keys for a stem, supporting both /lora_a and _lora_a suffixes."""
    nested = (f"{stem}/lora_a", f"{stem}/lora_b")
    suffixed = (f"{stem}_lora_a", f"{stem}_lora_b")
    if nested[0] in flat and nested[1] in flat:
        return nested
    if suffixed[0] in flat and suffixed[1] in flat:
        return suffixed
    raise KeyError(f"No A/B factors for {stem}")


def _pair(flat: dict, stem: str, *, index=None, split=None) -> tuple[np.ndarray, np.ndarray]:
    """Extract and reshape LoRA A/B from flat dict, matching export_table8_adapter exactly."""
    a_key, b_key = _factor_keys(flat, stem)
    a, b = flat[a_key], flat[b_key]
    if index is not None:
        a, b = a[index], b[index]
    if split is not None:
        a, b = a[split], b[split]
    # JAX uses [in..., rank] and [rank, out...]. Canonical uses [rank, in] and [out, rank].
    a = a.reshape(-1, a.shape[-1]).T
    b = b.reshape(b.shape[0], -1).T
    return a.astype(np.float32), b.astype(np.float32)


def convert(orbax_path: str, base_checkpoint: str | None = None) -> CanonicalLoRAAdapter:
    if base_checkpoint is None:
        base_checkpoint = _paths.get("base_checkpoint")
    flat = _restore_orbax(orbax_path)
    modules: dict[str, LoRAModule] = {}

    def add(key: str, stem: str, **kwargs) -> None:
        a, b = _pair(flat, stem, **kwargs)
        modules[key] = LoRAModule(key, a, b, 16.0, {"jax_stem": stem, **kwargs})

    # Action head + time MLP
    for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
        add(f"policy.{name}", name)

    # LM heads
    add("paligemma_with_expert.gemma_expert.lm_head", "gemma_expert_lm_head")
    add("paligemma_with_expert.paligemma.lm_head", "paligemma_lm_head")
    add("paligemma_with_expert.paligemma.model.multi_modal_projector.linear", "PaliGemma/img/head")

    # Vision encoder (27 layers)
    vision_stem = "PaliGemma/img/Transformer/encoderblock"
    vision_components = {
        "MlpBlock_0/Dense_0": "mlp.fc1",
        "MlpBlock_0/Dense_1": "mlp.fc2",
        "MultiHeadDotProductAttention_0/key": "self_attn.k_proj",
        "MultiHeadDotProductAttention_0/out": "self_attn.out_proj",
        "MultiHeadDotProductAttention_0/query": "self_attn.q_proj",
        "MultiHeadDotProductAttention_0/value": "self_attn.v_proj",
    }
    for layer in range(27):
        prefix = (
            "paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder."
            f"layers.{layer}"
        )
        for jax_name, component in vision_components.items():
            add(f"{prefix}.{component}", f"{vision_stem}/{jax_name}", index=layer)

    # Language model (18 layers) - PaliGemma side
    language_components = {
        "q": ("attn/q_einsum", None, "self_attn.q_proj"),
        "k": ("attn/kv_einsum", 0, "self_attn.k_proj"),
        "v": ("attn/kv_einsum", 1, "self_attn.v_proj"),
        "o": ("attn/attn_vec_einsum", None, "self_attn.o_proj"),
        "gate": ("mlp/gating_einsum", 0, "mlp.gate_proj"),
        "up": ("mlp/gating_einsum", 1, "mlp.up_proj"),
        "down": ("mlp/linear", None, "mlp.down_proj"),
    }
    for layer in range(18):
        lp = f"paligemma_with_expert.paligemma.model.language_model.layers.{layer}"
        for _, (stem, split, component) in language_components.items():
            add(f"{lp}.{component}", f"PaliGemma/llm/layers/{stem}", index=layer, split=split)

    # Expert language model (18 layers)
    expert_components = {
        **language_components,
        "input_norm": ("pre_attention_norm_1/Dense_0", None, "input_layernorm.dense"),
        "post_norm": ("pre_ffw_norm_1/Dense_0", None, "post_attention_layernorm"),
    }
    for layer in range(18):
        ep = f"paligemma_with_expert.gemma_expert.model.layers.{layer}"
        for _, (stem, split, component) in expert_components.items():
            if stem.startswith("attn/"):
                first, last = stem.rsplit("/", 1)
                stem_full = f"{first}/{last}_1"
            elif stem.startswith("mlp/"):
                stem_full = stem.replace("mlp/", "mlp_1/", 1)
            else:
                stem_full = stem
            add(f"{ep}.{component}", f"PaliGemma/llm/layers/{stem_full}", index=layer, split=split)

    # Expert final norm
    add("paligemma_with_expert.gemma_expert.model.norm.dense",
        "PaliGemma/llm/final_norm_1/Dense_0")

    adapter = CanonicalLoRAAdapter(
        modules,
        base_checkpoint=base_checkpoint,
        metadata={"source_format": "wizard_pi05_jax_table8"},
    )
    adapter.validate(rank=16, alpha=16)
    return adapter


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orbax", type=str, required=True,
                        help="Path to Orbax checkpoint directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .npz path")
    parser.add_argument("--base_checkpoint", type=str,
                        default=_paths.get("base_checkpoint"),
                        help="Base checkpoint path")
    args = parser.parse_args(argv)

    print(f"Loading Orbax checkpoint from {args.orbax}...")
    adapter = convert(args.orbax, base_checkpoint=args.base_checkpoint)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    adapter.save_npz(output)
    print(f"Saved canonical adapter: {len(adapter.modules)} modules → {output}")


if __name__ == "__main__":
    main()
