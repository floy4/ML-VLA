# tests/test_lora_mapping.py
"""assemble/disassemble roundtrip on a synthetic template covering index & split slots."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlvla.meta.e2e.lora_mapping import assemble, build_mapping, disassemble


def _write_template(tmp_path: Path) -> Path:
    """5 modules: flat suffixed, kv-style index+split, and index-only N-D slots (q/o-style)."""
    modules = [
        # canonical A (16,8) / B (32,16);jax 平铺 a (8,16) 后缀式、b (16,32)
        {"key": "flat.proj", "A": np.arange(16 * 8, dtype=np.float32).reshape(16, 8),
         "B": np.arange(32 * 16, dtype=np.float32).reshape(32, 16),
         "meta": {"jax_stem": "policy/action_in_proj"}},
        # index=0, split=0;jax a (1,2,8,16) 嵌套式 → slot (2,8,16) → 此模块取 [0,0]
        {"key": "stacked.k", "A": np.arange(16 * 8, dtype=np.float32).reshape(16, 8) + 100,
         "B": np.arange(32 * 16, dtype=np.float32).reshape(32, 16) + 100,
         "meta": {"jax_stem": "PaliGemma/llm/layers/attn/kv_einsum", "index": 0, "split": 0}},
        # index=0, split=1 → 同一张量另一 slot
        {"key": "stacked.v", "A": np.arange(16 * 8, dtype=np.float32).reshape(16, 8) + 200,
         "B": np.arange(32 * 16, dtype=np.float32).reshape(32, 16) + 200,
         "meta": {"jax_stem": "PaliGemma/llm/layers/attn/kv_einsum", "index": 0, "split": 1}},
        # q 风格:index only;b slot 为 3-D (r, heads, head_dim) → 验证 B 侧 N-D 展平
        {"key": "stacked.q", "A": np.arange(16 * 8, dtype=np.float32).reshape(16, 8) + 300,
         "B": np.arange(32 * 16, dtype=np.float32).reshape(32, 16) + 300,
         "meta": {"jax_stem": "PaliGemma/llm/layers/attn/q_einsum", "index": 0}},
        # o(attn_vec)风格:index only;a slot 为 3-D (heads, head_dim, r) → 验证 A 侧 N-D 展平
        {"key": "stacked.o", "A": np.arange(16 * 8, dtype=np.float32).reshape(16, 8) + 400,
         "B": np.arange(32 * 16, dtype=np.float32).reshape(32, 16) + 400,
         "meta": {"jax_stem": "PaliGemma/llm/layers/attn/attn_vec_einsum", "index": 0}},
    ]
    arrays, manifest = {}, {"modules": []}
    for i, m in enumerate(modules):
        arrays[f"module_{i}_A"], arrays[f"module_{i}_B"] = m["A"], m["B"]
        manifest["modules"].append({"key": m["key"], "alpha": 16.0, "metadata": m["meta"]})
    arrays["manifest_json"] = np.asarray(json.dumps(manifest))
    path = tmp_path / "template.npz"
    np.savez(path, **arrays)
    return path


def _fake_model_shapes():
    return {
        ("policy", "action_in_proj_lora_a"): (8, 16),
        ("policy", "action_in_proj_lora_b"): (16, 32),
        ("PaliGemma", "llm", "layers", "attn", "kv_einsum", "lora_a"): (1, 2, 8, 16),
        ("PaliGemma", "llm", "layers", "attn", "kv_einsum", "lora_b"): (1, 2, 16, 32),
        # q: a 2-D;b 物理形状 (1, r, heads, head_dim) → index 后 slot (16, 4, 8)
        ("PaliGemma", "llm", "layers", "attn", "q_einsum", "lora_a"): (1, 8, 16),
        ("PaliGemma", "llm", "layers", "attn", "q_einsum", "lora_b"): (1, 16, 4, 8),
        # attn_vec: a 物理形状 (1, heads, head_dim, r) → index 后 slot (2, 4, 16);b 2-D
        ("PaliGemma", "llm", "layers", "attn", "attn_vec_einsum", "lora_a"): (1, 2, 4, 16),
        ("PaliGemma", "llm", "layers", "attn", "attn_vec_einsum", "lora_b"): (1, 16, 32),
    }


def test_roundtrip_identity(tmp_path):
    mapping = build_mapping(_write_template(tmp_path), _fake_model_shapes())
    assert set(mapping.keys) == {"flat.proj", "stacked.k", "stacked.v", "stacked.q", "stacked.o"}
    values = {
        "flat.proj": {"A": np.random.default_rng(0).normal(size=(16, 8)).astype(np.float32),
                      "B": np.random.default_rng(1).normal(size=(32, 16)).astype(np.float32)},
        "stacked.k": {"A": np.random.default_rng(2).normal(size=(16, 8)).astype(np.float32),
                      "B": np.random.default_rng(3).normal(size=(32, 16)).astype(np.float32)},
        "stacked.v": {"A": np.random.default_rng(4).normal(size=(16, 8)).astype(np.float32),
                      "B": np.random.default_rng(5).normal(size=(32, 16)).astype(np.float32)},
        "stacked.q": {"A": np.random.default_rng(6).normal(size=(16, 8)).astype(np.float32),
                      "B": np.random.default_rng(7).normal(size=(32, 16)).astype(np.float32)},
        "stacked.o": {"A": np.random.default_rng(8).normal(size=(16, 8)).astype(np.float32),
                      "B": np.random.default_rng(9).normal(size=(32, 16)).astype(np.float32)},
    }
    tensors = assemble(mapping, values)
    # 把“梯度”设为组装结果本身 ⇒ 拆解应还原出原值(转置往返恒等)
    back = disassemble(mapping, tensors)
    for key, factors in values.items():
        np.testing.assert_allclose(back[key]["A"], factors["A"], rtol=1e-6)
        np.testing.assert_allclose(back[key]["B"], factors["B"], rtol=1e-6)


def test_split_slots_do_not_clobber(tmp_path):
    mapping = build_mapping(_write_template(tmp_path), _fake_model_shapes())
    values = {k: {"A": np.ones((16, 8), np.float32), "B": np.ones((32, 16), np.float32)}
              for k in mapping.keys}
    values["stacked.k"]["A"] = np.full((16, 8), 3.0, np.float32)
    values["stacked.v"]["A"] = np.full((16, 8), 7.0, np.float32)
    tensors = assemble(mapping, values)
    a = tensors[("PaliGemma", "llm", "layers", "attn", "kv_einsum", "lora_a")]
    # k 写 split0(全 3),v 写 split1(全 7);A.T → slot 形状 (8,16) 值均匀,互不覆盖
    assert a[0, 0].min() == 3.0 and a[0, 0].max() == 3.0
    assert a[0, 1].min() == 7.0 and a[0, 1].max() == 7.0


def test_build_mapping_validates_shapes(tmp_path):
    template = _write_template(tmp_path)
    # stacked 模块的 stem 在缺失 kv 张量时无法解析 → build 必须失败(KeyError 先于形状校验)
    missing = {k: v for k, v in _fake_model_shapes().items() if "kv" not in "/".join(k)}
    with pytest.raises(KeyError, match="kv_einsum"):
        build_mapping(template, missing)
    # 张量存在但形状错误 → 模块级形状校验报 ValueError 并点名模块
    wrong = dict(_fake_model_shapes())
    wrong[("policy", "action_in_proj_lora_a")] = (8, 17)
    with pytest.raises(ValueError, match="flat.proj"):
        build_mapping(template, wrong)
