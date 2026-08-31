"""Base-path-compatible SigLIP with Table-8 rank-16 LoRA projections."""

from __future__ import annotations

from collections.abc import Sequence

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.siglip as base_siglip
import openpi.training.sharding as sharding
from mlvla.jax.lora_layers import LoRADense, LoRAInputProjection, LoRAOutputProjection


class LoRAMlpBlock(nn.Module):
    mlp_dim: int | None = None
    dropout: float = 0.0
    dtype_mm: str = "float32"
    enabled: bool = True

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        d = x.shape[-1]
        inits = {"kernel_init": nn.initializers.xavier_uniform(),
                 "bias_init": nn.initializers.normal(stddev=1e-6)}
        x = LoRADense(self.mlp_dim or 4 * d, dtype=self.dtype_mm, enabled=self.enabled,
                      name="Dense_0", **inits)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        return LoRADense(d, dtype=self.dtype_mm, enabled=self.enabled, name="Dense_1", **inits)(x)


class LoRASelfAttention(nn.Module):
    num_heads: int
    dtype_mm: str = "float32"
    enabled: bool = True

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        width = x.shape[-1]
        if width % self.num_heads:
            raise ValueError("Vision width must be divisible by number of heads")
        head_dim = width // self.num_heads
        init = nn.initializers.xavier_uniform()
        query = LoRAInputProjection((self.num_heads, head_dim), dtype=self.dtype_mm,
                                    kernel_init=init, enabled=self.enabled, name="query")(x)
        key = LoRAInputProjection((self.num_heads, head_dim), dtype=self.dtype_mm,
                                  kernel_init=init, enabled=self.enabled, name="key")(x)
        value = LoRAInputProjection((self.num_heads, head_dim), dtype=self.dtype_mm,
                                    kernel_init=init, enabled=self.enabled, name="value")(x)
        attended = nn.dot_product_attention(query, key, value, deterministic=deterministic, dtype=self.dtype_mm)
        return LoRAOutputProjection(width, dtype=self.dtype_mm, kernel_init=init,
                                    enabled=self.enabled, name="out")(attended)


class LoRAEncoder1DBlock(nn.Module):
    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"
    enabled: bool = True

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}
        x = sharding.activation_sharding_constraint(x)
        y = nn.LayerNorm(dtype=self.dtype_mm, name="LayerNorm_0")(x)
        y = out["sa"] = LoRASelfAttention(self.num_heads, self.dtype_mm, self.enabled,
                                           name="MultiHeadDotProductAttention_0")(y, deterministic)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+sa"] = x + y
        y = nn.LayerNorm(dtype=self.dtype_mm, name="LayerNorm_1")(x)
        y = out["mlp"] = LoRAMlpBlock(self.mlp_dim, self.dropout, self.dtype_mm, self.enabled,
                                      name="MlpBlock_0")(y, deterministic)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        return sharding.activation_sharding_constraint(x + y), out


class LoRAEncoder(nn.Module):
    depth: int
    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"
    enabled: bool = True

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}
        if self.scan:
            block = nn.remat(LoRAEncoder1DBlock, prevent_cse=False, static_argnums=(2,),
                             policy=getattr(jax.checkpoint_policies, self.remat_policy, None))
            x, scan_out = nn.scan(block, variable_axes={"params": 0},
                                  split_rngs={"params": True, "dropout": True},
                                  in_axes=nn.broadcast, length=self.depth)(
                name="encoderblock", dtype_mm=self.dtype_mm, mlp_dim=self.mlp_dim,
                num_heads=self.num_heads, dropout=self.dropout, enabled=self.enabled,
            )(x, deterministic)
            for layer in range(self.depth):
                out[f"block{layer:02d}"] = jax.tree.map(lambda value, i=layer: value[i], scan_out)
        else:
            for layer in range(self.depth):
                x, out[f"block{layer:02d}"] = LoRAEncoder1DBlock(
                    self.mlp_dim, self.num_heads, self.dropout, self.dtype_mm,
                    self.enabled,
                    name=f"encoderblock_{layer}",
                )(x, deterministic)
            out["pre_ln"] = x
        return nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x), out


class _LoRASiglip(nn.Module):
    num_classes: int | None = None
    patch_size: Sequence[int] = (16, 16)
    width: int = 768
    depth: int = 12
    mlp_dim: int | None = None
    num_heads: int = 12
    posemb: str = "learn"
    rep_size: int | bool = False
    dropout: float = 0.0
    pool_type: str = "gap"
    head_zeroinit: bool = True
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"
    enabled: bool = True

    @nn.compact
    def __call__(self, image, *, train=False):
        out = {}
        image = jnp.asarray(image, jnp.float32)
        x = out["stem"] = nn.Conv(self.width, self.patch_size, strides=self.patch_size,
                                   padding="VALID", name="embedding", dtype=jnp.float32)(image)
        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])
        x = out["with_posemb"] = x + base_siglip.get_posemb(
            self, self.posemb, (h, w), c, "pos_embedding", jnp.float32,
        )
        if self.pool_type == "tok":
            cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
            x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)
        x = nn.Dropout(rate=self.dropout)(x, not train).astype(self.dtype_mm)
        x, out["encoder"] = LoRAEncoder(
            self.depth, self.mlp_dim, self.num_heads, self.dropout, self.scan,
            self.remat_policy, self.dtype_mm, self.enabled, name="Transformer",
        )(x, deterministic=not train)
        encoded = out["encoded"] = x
        if self.pool_type == "map":
            x = out["head_input"] = base_siglip.MAPHead(
                num_heads=self.num_heads, mlp_dim=self.mlp_dim, dtype_mm=self.dtype_mm,
            )(x)
        elif self.pool_type == "gap":
            x = out["head_input"] = jnp.mean(x, axis=1)
        elif self.pool_type in ("0", "tok"):
            x = out["head_input"] = x[:, 0]
            if self.pool_type == "tok":
                encoded = encoded[:, 1:]
        elif self.pool_type != "none":
            raise ValueError(f"Unknown pool type: {self.pool_type!r}")
        x_2d = jnp.reshape(encoded, [n, h, w, -1])
        if self.rep_size:
            rep_size = self.width if self.rep_size is True else self.rep_size
            hid = LoRADense(rep_size, dtype=self.dtype_mm, enabled=self.enabled, name="pre_logits")
            x_2d, x = nn.tanh(hid(x_2d)), nn.tanh(hid(x))
        out["pre_logits_2d"], out["pre_logits"] = x_2d, x
        if self.num_classes:
            init = nn.initializers.zeros if self.head_zeroinit else nn.initializers.lecun_normal()
            head = LoRADense(self.num_classes, dtype=self.dtype_mm, enabled=self.enabled,
                             name="head", kernel_init=init)
            x_2d, x = head(x_2d), head(x)
            out["logits_2d"], out["logits"] = x_2d, x
        return x, out


def Module(num_classes=None, *, variant=None, **kwargs):  # noqa: N802
    return _LoRASiglip(num_classes, **{**base_siglip.decode_variant(variant), **kwargs})
