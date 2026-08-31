"""JAX π0.5 with the paper's rank-16 Table-8 LoRA topology."""

from __future__ import annotations

import dataclasses

import einops
from flax import nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp

from mlvla.jax._compat import isolate_unused_pytorch_backend

isolate_unused_pytorch_backend()

import openpi.models.gemma as gemma
from openpi.models import model as model_lib
from openpi.models import pi0 as pi0_lib
from openpi.models import pi0_config
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils

from mlvla.jax.gemma_lora import install_patches
from mlvla.jax.lora_layers import NnxLoRAFactors, NnxLoRALinear
from mlvla.jax import siglip_lora


@dataclasses.dataclass(frozen=True)
class WizardPi05Config(pi0_config.Pi0Config):
    """Drop-in model config that keeps the original Orbax base parameter paths."""
    pi05: bool = True
    action_horizon: int = 10
    discrete_state_input: bool = False
    # Static switch used by the zero-adapter equivalence gate. Parameters are
    # still instantiated, so checkpoint topology is identical in both modes.
    adapter_enabled: bool = True

    def create(self, rng: at.KeyArrayLike) -> "WizardPi0":
        return WizardPi0(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter(self):
        return nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex(".*lora.*")))


class WizardPi0(pi0_lib.Pi0):
    """Pi0 methods with a Table-8-compatible constructor and evidence hook."""

    def __init__(self, config: WizardPi05Config, rngs: nnx.Rngs):
        # Deliberately bypass Pi0.__init__; all inherited policy methods use the
        # same public attributes created below.
        model_lib.BaseModel.__init__(self, config.action_dim, config.action_horizon, config.max_token_len)
        if not config.pi05:
            raise ValueError("WIZARD's Table-8 topology is defined for π0.5")
        install_patches(enabled=config.adapter_enabled)
        self.pi05 = True
        rank16 = {"attn": gemma.lora.LoRAConfig(rank=16, alpha=16.0),
                  "ffn": gemma.lora.LoRAConfig(rank=16, alpha=16.0)}
        paligemma_config = dataclasses.replace(gemma.get_config("gemma_2b"), lora_configs=rank16)
        expert_config = dataclasses.replace(gemma.get_config("gemma_300m"), lora_configs=rank16)
        llm = nnx_bridge.ToNNX(gemma.Module(
            configs=[paligemma_config, expert_config], embed_dtype=config.dtype, adarms=True,
        ))
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])
        img = nnx_bridge.ToNNX(siglip_lora.Module(
            num_classes=paligemma_config.width, variant="So400m/14", pool_type="none",
            scan=True, dtype_mm=config.dtype, enabled=config.adapter_enabled,
        ))
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = NnxLoRALinear(
            config.action_dim, expert_config.width, enabled=config.adapter_enabled, rngs=rngs,
        )
        self.time_mlp_in = NnxLoRALinear(
            expert_config.width, expert_config.width, enabled=config.adapter_enabled, rngs=rngs,
        )
        self.time_mlp_out = NnxLoRALinear(
            expert_config.width, expert_config.width, enabled=config.adapter_enabled, rngs=rngs,
        )
        self.action_out_proj = NnxLoRALinear(
            expert_config.width, config.action_dim, enabled=config.adapter_enabled, rngs=rngs,
        )
        # PyTorch π0.5 instantiates these tied/unused heads and Table 8 retains
        # their factors. Keeping factor-only modules avoids duplicating the huge
        # tied base matrices in JAX.
        self.paligemma_lm_head = NnxLoRAFactors(paligemma_config.width, gemma.PALIGEMMA_VOCAB_SIZE,
                                                rngs=rngs)
        self.gemma_expert_lm_head = NnxLoRAFactors(expert_config.width, gemma.PALIGEMMA_VOCAB_SIZE,
                                                  rngs=rngs)
        self.deterministic = True

    def encode_task_evidence(self, obs: model_lib.Observation) -> jax.Array:
        """Return fused prompt+visual tokens [B,S,2048], never state/actions."""
        prefix_tokens, prefix_mask, prefix_ar = self.embed_prefix(obs)
        attention_mask = pi0_lib.make_attn_mask(prefix_mask, prefix_ar)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        encoded, _ = self.PaliGemma.llm(
            [prefix_tokens, None], positions, attention_mask, [None, None],
        )
        evidence = encoded[0]
        if evidence.shape[-1] != 2048:
            raise ValueError(f"Expected 2048-wide evidence, got {evidence.shape}")
        return evidence
