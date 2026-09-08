# tests/test_jax_backend.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_collect_lora_paths_pure():
    """_collect_lora_paths 是纯函数:扁平 dict → 含 'lora' 的路径与形状。"""
    from mlvla.meta.e2e.jax_backend import _collect_lora_paths
    flat = {
        ("policy", "action_in_proj_lora_a"): (8, 16),
        ("policy", "action_in_proj", "kernel"): (8, 16),
        ("PaliGemma", "llm", "layers", "attn", "kv_einsum", "lora_b"): (1, 2, 16, 32),
    }
    out = _collect_lora_paths(flat)
    assert set(out) == {("policy", "action_in_proj_lora_a"),
                        ("PaliGemma", "llm", "layers", "attn", "kv_einsum", "lora_b")}
