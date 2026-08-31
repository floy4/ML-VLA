"""Rank-global LoRA layers whose base parameter paths match openpi JAX."""

from __future__ import annotations

from collections.abc import Sequence
import math

from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp


class LoRADense(nn.Module):
    features: int
    rank: int = 16
    alpha: float = 16.0
    use_bias: bool = True
    dtype: str | jnp.dtype = "float32"
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros
    lora_init: nn.initializers.Initializer = nn.initializers.normal(stddev=0.01)
    enabled: bool = True

    @nn.compact
    def __call__(self, x):
        kernel = self.param("kernel", self.kernel_init, (x.shape[-1], self.features))
        compute_dtype = jnp.dtype(self.dtype)
        inputs = x.astype(compute_dtype)
        kernel = kernel.astype(compute_dtype)
        y = jax.lax.dot_general(
            inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())),
        )
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,))
            y = y + jnp.reshape(bias.astype(compute_dtype), (1,) * (y.ndim - 1) + (-1,))
        a = self.param("lora_a", self.lora_init, (x.shape[-1], self.rank))
        b = self.param("lora_b", nn.initializers.zeros, (self.rank, self.features))
        if not self.enabled:
            return y.astype(compute_dtype)
        return (y + jnp.einsum("...i,ir,ro->...o", inputs, a.astype(compute_dtype), b.astype(compute_dtype))
                * (self.alpha / self.rank)).astype(compute_dtype)


class LoRAInputProjection(nn.Module):
    """DenseGeneral projection: input width -> one or more output axes."""
    features: tuple[int, ...]
    rank: int = 16
    alpha: float = 16.0
    use_bias: bool = True
    dtype: str | jnp.dtype = "float32"
    kernel_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros
    lora_init: nn.initializers.Initializer = nn.initializers.normal(stddev=0.01)
    enabled: bool = True

    @nn.compact
    def __call__(self, x):
        kernel = self.param("kernel", self.kernel_init, (x.shape[-1], *self.features))
        flat_kernel = kernel.reshape(x.shape[-1], -1).astype(x.dtype)
        y = jnp.matmul(x, flat_kernel).reshape(*x.shape[:-1], *self.features)
        if self.use_bias:
            bias = self.param("bias", self.bias_init, self.features)
            y = y + bias.astype(x.dtype)
        a = self.param("lora_a", self.lora_init, (x.shape[-1], self.rank))
        b = self.param("lora_b", nn.initializers.zeros, (self.rank, *self.features))
        if not self.enabled:
            return y.astype(x.dtype)
        delta = jnp.matmul(jnp.matmul(x, a.astype(x.dtype)), b.reshape(self.rank, -1).astype(x.dtype))
        return (y + delta.reshape(y.shape) * (self.alpha / self.rank)).astype(x.dtype)


class LoRAOutputProjection(nn.Module):
    """DenseGeneral projection: multiple input axes -> output width."""
    features: int
    rank: int = 16
    alpha: float = 16.0
    use_bias: bool = True
    dtype: str | jnp.dtype = "float32"
    kernel_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros
    lora_init: nn.initializers.Initializer = nn.initializers.normal(stddev=0.01)
    enabled: bool = True

    @nn.compact
    def __call__(self, x):
        input_shape = x.shape[-2:]
        kernel = self.param("kernel", self.kernel_init, (*input_shape, self.features))
        flat_x = x.reshape(*x.shape[:-2], -1)
        y = jnp.matmul(flat_x, kernel.reshape(-1, self.features).astype(x.dtype))
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,))
            y = y + bias.astype(x.dtype)
        a = self.param("lora_a", self.lora_init, (*input_shape, self.rank))
        b = self.param("lora_b", nn.initializers.zeros, (self.rank, self.features))
        if not self.enabled:
            return y.astype(x.dtype)
        delta = jnp.matmul(jnp.matmul(flat_x, a.reshape(-1, self.rank).astype(x.dtype)), b.astype(x.dtype))
        return (y + delta * (self.alpha / self.rank)).astype(x.dtype)


class NnxLoRALinear(nnx.Linear):
    def __init__(self, in_features: int, out_features: int, *, rank: int = 16, alpha: float = 16.0,
                 enabled: bool = True, rngs: nnx.Rngs, **kwargs):
        super().__init__(in_features, out_features, rngs=rngs, **kwargs)
        self.rank = rank
        self.alpha = alpha
        self.enabled = enabled
        self.lora_a = nnx.Param(jax.random.normal(rngs.params(), (in_features, rank)) * 0.01)
        self.lora_b = nnx.Param(jnp.zeros((rank, out_features), dtype=jnp.float32))

    def __call__(self, inputs):
        base = super().__call__(inputs)
        if not self.enabled:
            return base
        delta = jnp.matmul(jnp.matmul(inputs, self.lora_a.value.astype(inputs.dtype)),
                           self.lora_b.value.astype(inputs.dtype))
        return (base + delta * (self.alpha / self.rank)).astype(inputs.dtype)


class NnxLoRAFactors(nnx.Module):
    """LoRA factors for tied/unused LM heads retained by Table 8."""
    def __init__(self, in_features: int, out_features: int, *, rank: int = 16, alpha: float = 16.0,
                 rngs: nnx.Rngs):
        self.rank = rank
        self.alpha = alpha
        self.lora_a = nnx.Param(jax.random.normal(rngs.params(), (in_features, rank)) * 0.01)
        self.lora_b = nnx.Param(jnp.zeros((rank, out_features), dtype=jnp.float32))
