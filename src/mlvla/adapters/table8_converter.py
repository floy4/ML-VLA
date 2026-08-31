"""Lossless conversion between WizardPi0's scanned JAX factors and canonical A/B."""

from __future__ import annotations

from collections.abc import Mapping

from flax import nnx
import jax.numpy as jnp
import numpy as np

from mlvla.adapters.canonical_lora import CanonicalLoRAAdapter, LoRAModule


def _as_flat_arrays(state_or_model) -> dict[str, np.ndarray]:
    state = nnx.state(state_or_model) if isinstance(state_or_model, nnx.Module) else state_or_model
    if hasattr(state, "flat_state"):
        return {"/".join(map(str, path)): np.asarray(var.value) for path, var in state.flat_state().items()
                if "lora" in "/".join(map(str, path)).lower()}
    return {str(key): np.asarray(value) for key, value in state.items() if "lora" in str(key).lower()}


def _factor_keys(flat: Mapping[str, object], stem: str) -> tuple[str, str]:
    nested = (f"{stem}/lora_a", f"{stem}/lora_b")
    suffixed = (f"{stem}_lora_a", f"{stem}_lora_b")
    if nested[0] in flat and nested[1] in flat:
        return nested
    if suffixed[0] in flat and suffixed[1] in flat:
        return suffixed
    raise KeyError(f"No A/B factors for {stem}")


def _pair(flat: Mapping[str, np.ndarray], stem: str, *, index=None, split=None,
          output_grouped: bool = False) -> tuple[np.ndarray, np.ndarray]:
    a_key, b_key = _factor_keys(flat, stem)
    a, b = flat[a_key], flat[b_key]
    if index is not None:
        a, b = a[index], b[index]
    if split is not None:
        a, b = a[split], b[split]
    # JAX uses [in..., rank] and [rank, out...]. Canonical uses [rank,in]
    # and [out,rank], flattening only architectural head axes.
    a = a.reshape(-1, a.shape[-1]).T
    b = b.reshape(b.shape[0], -1).T
    return a.astype(np.float32), b.astype(np.float32)


def export_table8_adapter(state_or_model, *, base_checkpoint: str | None = None) -> CanonicalLoRAAdapter:
    flat = _as_flat_arrays(state_or_model)
    modules: dict[str, LoRAModule] = {}

    def add(key: str, stem: str, **kwargs) -> None:
        a, b = _pair(flat, stem, **kwargs)
        modules[key] = LoRAModule(key, a, b, 16.0, {"jax_stem": stem, **kwargs})

    for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
        add(f"policy.{name}", name)
    add("paligemma_with_expert.gemma_expert.lm_head", "gemma_expert_lm_head")
    add("paligemma_with_expert.paligemma.lm_head", "paligemma_lm_head")
    add("paligemma_with_expert.paligemma.model.multi_modal_projector.linear", "PaliGemma/img/head")

    vision_stem = "PaliGemma/img/Transformer/encoderblock"
    vision_components = {
        "MlpBlock_0/Dense_0": "mlp.fc1", "MlpBlock_0/Dense_1": "mlp.fc2",
        "MultiHeadDotProductAttention_0/key": "self_attn.k_proj",
        "MultiHeadDotProductAttention_0/out": "self_attn.out_proj",
        "MultiHeadDotProductAttention_0/query": "self_attn.q_proj",
        "MultiHeadDotProductAttention_0/value": "self_attn.v_proj",
    }
    for layer in range(27):
        prefix = ("paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder."
                  f"layers.{layer}")
        for jax_name, component in vision_components.items():
            add(f"{prefix}.{component}", f"{vision_stem}/{jax_name}", index=layer)

    language_components = {
        "q": ("attn/q_einsum", None, "self_attn.q_proj"),
        "k": ("attn/kv_einsum", 0, "self_attn.k_proj"),
        "v": ("attn/kv_einsum", 1, "self_attn.v_proj"),
        "o": ("attn/attn_vec_einsum", None, "self_attn.o_proj"),
        "gate": ("mlp/gating_einsum", 0, "mlp.gate_proj"),
        "up": ("mlp/gating_einsum", 1, "mlp.up_proj"),
        "down": ("mlp/linear", None, "mlp.down_proj"),
    }
    expert_components = {
        **language_components,
        "input_norm": ("pre_attention_norm_1/Dense_0", None, "input_layernorm.dense"),
        "post_norm": ("pre_ffw_norm_1/Dense_0", None, "post_attention_layernorm"),
    }
    for layer in range(18):
        lp = f"paligemma_with_expert.paligemma.model.language_model.layers.{layer}"
        for _, (stem, split, component) in language_components.items():
            add(f"{lp}.{component}", f"PaliGemma/llm/layers/{stem}", index=layer, split=split)
        ep = f"paligemma_with_expert.gemma_expert.model.layers.{layer}"
        for _, (stem, split, component) in expert_components.items():
            # Gemma's two-expert naming suffix is applied at the projection/module level.
            if stem.startswith("attn/"):
                first, last = stem.rsplit("/", 1)
                stem = f"{first}/{last}_1"
            elif stem.startswith("mlp/"):
                stem = stem.replace("mlp/", "mlp_1/", 1)
            add(f"{ep}.{component}", f"PaliGemma/llm/layers/{stem}", index=layer, split=split)
    add("paligemma_with_expert.gemma_expert.model.norm.dense", "PaliGemma/llm/final_norm_1/Dense_0")
    adapter = CanonicalLoRAAdapter(modules, base_checkpoint=base_checkpoint,
                                   metadata={"source_format": "wizard_pi05_jax_table8"})
    adapter.validate(rank=16, alpha=16)
    return adapter


def inject_table8_adapter(model: nnx.Module, adapter: CanonicalLoRAAdapter) -> None:
    """Inject canonical factors by reversing the exporter metadata exactly."""
    adapter.validate(rank=16, alpha=16)
    state = nnx.state(model)
    flat = state.flat_state()
    replacements: dict[tuple, np.ndarray] = {}
    grouped: dict[tuple[str, int | None], dict[int | None, LoRAModule]] = {}
    for module in adapter.modules.values():
        meta = module.metadata
        if "jax_stem" not in meta:
            raise ValueError(f"{module.key}: missing jax_stem metadata")
        grouped.setdefault((str(meta["jax_stem"]), meta.get("index")), {})[meta.get("split")] = module

    for (stem, index), splits in grouped.items():
        string_flat = {"/".join(map(str, path)): variable for path, variable in flat.items()}
        a_key, b_key = _factor_keys(string_flat, stem)
        a_path, b_path = tuple(a_key.split("/")), tuple(b_key.split("/"))
        old_a, old_b = np.asarray(flat[a_path].value), np.asarray(flat[b_path].value)
        # A scanned tensor is shared by every logical layer.  Several grouped
        # entries therefore resolve to the same a_path/b_path.  Start from the
        # updates accumulated for earlier indices instead of the original
        # tensor, otherwise each layer overwrites the preceding one and only
        # the final logical layer survives injection.
        new_a = np.asarray(replacements.get(a_path, old_a)).copy()
        new_b = np.asarray(replacements.get(b_path, old_b)).copy()
        for split, module in splits.items():
            a = module.A.T
            b = module.B.T
            target_a = new_a[index] if index is not None else new_a
            target_b = new_b[index] if index is not None else new_b
            if split is not None:
                target_a, target_b = target_a[split], target_b[split]
            a, b = a.reshape(target_a.shape), b.reshape(target_b.shape)
            if index is None and split is None:
                new_a, new_b = a, b
            elif split is None:
                new_a[index], new_b[index] = a, b
            else:
                new_a[index, split], new_b[index, split] = a, b
        replacements[a_path], replacements[b_path] = new_a, new_b

    new_items = []
    for path, variable in flat.items():
        value = replacements.get(path)
        new_items.append((path, variable if value is None else variable.replace(value=jnp.asarray(value, variable.value.dtype))))
    nnx.update(model, nnx.State.from_flat_path(new_items))
