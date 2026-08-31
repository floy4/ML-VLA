"""Global-rank LoRA patches for openpi's scanned Gemma implementation."""

from __future__ import annotations

from collections.abc import Sequence
import functools
import math

import einops
from flax import linen as nn
import jax
import jax.numpy as jnp

import openpi.models.gemma as gemma
import openpi.models.lora as openpi_lora
import openpi.training.sharding as sharding
from mlvla.jax.lora_layers import LoRADense


class GlobalInputEinsum(nn.Module):
    shape: tuple[int, ...]
    rank: int = 16
    alpha: float = 16.0
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    enabled: bool = True

    @nn.compact
    def __call__(self, x):
        # Base shape [...output_group, input_width, output_width].
        w = self.param("w", self.init_fn, self.shape)
        groups, input_width, output_width = self.shape[:-2], self.shape[-2], self.shape[-1]
        flat_output = math.prod((*groups, output_width))
        if len(groups) != 1 or x.ndim != 3:
            raise ValueError("GlobalInputEinsum is specialized for Gemma attention")
        base = jnp.einsum("BTD,NDH->BTNH", x, w.astype(x.dtype))
        a = self.param("lora_a", nn.initializers.normal(stddev=0.01), (input_width, self.rank))
        b = self.param("lora_b", nn.initializers.zeros, (self.rank, *groups, output_width))
        if not self.enabled:
            return base.astype(x.dtype)
        delta = jnp.matmul(jnp.matmul(x, a.astype(x.dtype)), b.reshape(self.rank, -1).astype(x.dtype))
        base = jax.lax.optimization_barrier(base)
        return (base + delta.reshape(base.shape) * (self.alpha / self.rank)).astype(x.dtype)


class GlobalBatchedInputEinsum(nn.Module):
    """Separate global-rank factors for K and V, retaining the leading axis."""
    shape: tuple[int, ...]  # [2, kv_heads, input_width, head_dim]
    rank: int = 16
    alpha: float = 16.0
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    enabled: bool = True

    @nn.compact
    def __call__(self, x):
        count, heads, input_width, head_dim = self.shape
        w = self.param("w", self.init_fn, self.shape)
        base = jnp.einsum("bsd,ckdh->cbskh", x, w.astype(x.dtype))
        a = self.param("lora_a", nn.initializers.normal(stddev=0.01), (count, input_width, self.rank))
        b = self.param("lora_b", nn.initializers.zeros, (count, self.rank, heads, head_dim))
        if not self.enabled:
            return base.astype(x.dtype)
        delta = jnp.einsum("bsd,cdr,crkh->cbskh", x, a.astype(x.dtype), b.astype(x.dtype))
        base = jax.lax.optimization_barrier(base)
        return (base + delta * (self.alpha / self.rank)).astype(x.dtype)


class GlobalOutputEinsum(nn.Module):
    shape: tuple[int, int, int]  # heads, head_dim, output_width
    rank: int = 16
    alpha: float = 16.0
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    enabled: bool = True

    @nn.compact
    def __call__(self, x):
        heads, head_dim, output_width = self.shape
        w = self.param("w", self.init_fn, self.shape)
        base = jnp.einsum("btnh,nhd->btd", x, w.astype(x.dtype))
        a = self.param("lora_a", nn.initializers.normal(stddev=0.01), (heads, head_dim, self.rank))
        b = self.param("lora_b", nn.initializers.zeros, (self.rank, output_width))
        if not self.enabled:
            return base.astype(x.dtype)
        delta = jnp.einsum("btnh,nhr,rd->btd", x, a.astype(x.dtype), b.astype(x.dtype))
        base = jax.lax.optimization_barrier(base)
        return (base + delta * (self.alpha / self.rank)).astype(x.dtype)


class GlobalLoRAAttention(nn.Module):
    configs: Sequence[gemma.Config]
    enabled: bool = True

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache, return_last_query_attn=False):  # noqa: FBT002
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)
        dtype = next(x.dtype for x in xs if x is not None)
        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            suffix = "" if i == 0 else f"_{i}"
            if config.num_kv_heads == config.num_heads:
                raise NotImplementedError("π0.5 uses grouped-query attention; fused qkv is not expected")
            q = GlobalInputEinsum(
                (config.num_heads, config.width, config.head_dim), name=f"q_einsum{suffix}",
                enabled=self.enabled,
                init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            )(x)
            kv = GlobalBatchedInputEinsum(
                (2, config.num_kv_heads, config.width, config.head_dim), name=f"kv_einsum{suffix}",
                enabled=self.enabled,
                init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
            )(x)
            qkvs.append((q, kv[0], kv[1]))

        q, k, v = (jnp.concatenate(items, axis=1) for items in zip(*qkvs, strict=True))
        q = gemma._apply_rope(q, positions=positions)  # noqa: SLF001
        q *= self.configs[0].head_dim ** -0.5
        k = gemma._apply_rope(k, positions=positions)  # noqa: SLF001
        assert q.dtype == k.dtype == v.dtype == dtype
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k, v = jnp.concatenate([cache_k, k], axis=1), jnp.concatenate([cache_v, v], axis=1)
        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, -2.3819763e38)
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)
        if return_last_query_attn:
            last_query_attn = einops.rearrange(probs[:, :, :, -1, :], "B K G S -> B (K G) S")
        else:
            last_query_attn = jnp.zeros((q.shape[0], self.configs[0].num_heads, k.shape[1]), dtype=dtype)
        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")
        out, start = [], 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                out.append(None)
                continue
            end = start + x.shape[1]
            suffix = "" if i == 0 else f"_{i}"
            out.append(GlobalOutputEinsum(
                (config.num_heads, config.head_dim, config.width), name=f"attn_vec_einsum{suffix}",
                enabled=self.enabled,
                init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
            )(encoded[:, start:end]))
            start = end
        return out, (k, v), last_query_attn


class LoRAGemmaRMSNorm(nn.Module):
    """Base-compatible RMSNorm with rank-16 AdaRMS dense factors."""
    enabled: bool = True

    @nn.compact
    def __call__(self, x, cond):
        dtype = x.dtype
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-6)))
        if cond is None:
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            return (normed * (1 + scale)).astype(dtype), None
        # Keep the base operation byte-for-byte identical to openpi's RMSNorm;
        # factors live alongside Dense_0 so the Orbax base path is unchanged.
        modulation = nn.Dense(
            x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype, name="Dense_0",
        )(cond)
        a = self.param("Dense_0_lora_a", nn.initializers.normal(stddev=0.01), (cond.shape[-1], 16))
        b = self.param("Dense_0_lora_b", nn.initializers.zeros, (16, x.shape[-1] * 3))
        if not self.enabled:
            scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
            return (normed * (1 + scale) + shift).astype(dtype), gate
        modulation = modulation + jnp.einsum(
            "bi,ir,ro->bo", cond.astype(dtype), a.astype(dtype), b.astype(dtype),
        )
        scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
        return (normed * (1 + scale) + shift).astype(dtype), gate


class BarrierLoRAFeedForward(nn.Module):
    """Openpi FeedForward with an explicit base/adapter optimization boundary."""
    features: int
    hidden_dim: int
    lora_config: openpi_lora.LoRAConfig | None = None
    enabled: bool = True

    def setup(self):
        self.w_gating = self.param(
            "gating_einsum", nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        )
        self.w_linear = self.param(
            "linear", nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        )
        if self.lora_config:
            init, rank = self.lora_config.init_fn, self.lora_config.rank
            self.gating_einsum_lora_a = self.param("gating_einsum_lora_a", init, (2, self.features, rank))
            self.gating_einsum_lora_b = self.param(
                "gating_einsum_lora_b", nn.initializers.zeros, (2, rank, self.hidden_dim),
            )
            self.linear_lora_a = self.param("linear_lora_a", init, (self.hidden_dim, rank))
            self.linear_lora_b = self.param("linear_lora_b", nn.initializers.zeros, (rank, self.features))

    def _dot(self, x, weight, a=None, b=None):
        base = jnp.dot(x, weight.astype(x.dtype))
        if a is None:
            return base
        base = jax.lax.optimization_barrier(base)
        return (base + jnp.dot(jnp.dot(x, a.astype(x.dtype)), b.astype(x.dtype))).astype(x.dtype)

    def __call__(self, x):
        if self.lora_config and self.enabled:
            gate = self._dot(x, self.w_gating[0], self.gating_einsum_lora_a[0], self.gating_einsum_lora_b[0])
            up = self._dot(x, self.w_gating[1], self.gating_einsum_lora_a[1], self.gating_einsum_lora_b[1])
            hidden = nn.gelu(gate) * up
            return self._dot(hidden, self.w_linear, self.linear_lora_a, self.linear_lora_b)
        hidden = nn.gelu(jnp.dot(x, self.w_gating[0].astype(x.dtype))) * jnp.dot(
            x, self.w_gating[1].astype(x.dtype),
        )
        return jnp.dot(hidden, self.w_linear.astype(x.dtype))


def install_patches(*, enabled: bool = True) -> None:
    """Install process-local patches before constructing/applying WizardPi0."""
    gemma.Attention = functools.partial(GlobalLoRAAttention, enabled=enabled)
    gemma.RMSNorm = functools.partial(LoRAGemmaRMSNorm, enabled=enabled)
    gemma.lora.FeedForward = functools.partial(BarrierLoRAFeedForward, enabled=enabled)
