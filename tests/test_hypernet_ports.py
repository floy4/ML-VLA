# tests/test_hypernet_ports.py
"""Ported hypernetwork smoke tests: shapes, key coverage, identity-at-init."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MODULE_SHAPES = {
    "m.attn_q": {"A": (16, 64), "B": (128, 16)},
    "m.mlp_down": {"A": (16, 32), "B": (256, 16)},
    "m2.mlp_down": {"A": (16, 32), "B": (256, 16)},   # 同形状组(共享字典覆盖)
}


def test_shared_film_shapes_and_keys():
    from mlvla.meta.hypernet import SharedFiLMABHyperNetwork
    torch.manual_seed(0)
    net = SharedFiLMABHyperNetwork(dino_dim=16, view_dim=7, module_shapes=MODULE_SHAPES,
                                   condition_hidden_dim=8, hidden_dim=8, dict_dim=4)
    dino = torch.randn(2, 4, 16)
    view = torch.zeros(2, 7)
    out = net(dino, view)
    assert set(out) == set(MODULE_SHAPES)
    for key, factors in out.items():
        b = 2
        assert factors["A"].shape == (b, *MODULE_SHAPES[key]["A"])
        assert factors["B"].shape == (b, *MODULE_SHAPES[key]["B"])
        assert torch.isfinite(factors["A"]).all() and torch.isfinite(factors["B"]).all()


def test_shared_film_identity_at_init():
    """未训练 FiLM:gamma=1/beta=0 ⇒ 输出与 view 无关(逐位一致)。"""
    from mlvla.meta.hypernet import SharedFiLMABHyperNetwork
    torch.manual_seed(0)
    net = SharedFiLMABHyperNetwork(dino_dim=16, view_dim=7, module_shapes=MODULE_SHAPES,
                                   condition_hidden_dim=8, hidden_dim=8, dict_dim=4)
    net.eval()
    dino = torch.randn(1, 4, 16)
    with torch.no_grad():
        o1 = net(dino, torch.zeros(1, 7))
        o2 = net(dino, torch.tensor([[0.5, -0.2, 0.1, 1.0, 0.0, 0.3, -0.3]]))
    for key in MODULE_SHAPES:
        assert torch.allclose(o1[key]["A"], o2[key]["A"], atol=1e-6), key
        assert torch.allclose(o1[key]["B"], o2[key]["B"], atol=1e-6), key


def test_v4_shapes_and_keys():
    from mlvla.meta.hypernet import V4DirectABHyperNetwork
    torch.manual_seed(0)
    net = V4DirectABHyperNetwork(condition_dim=16 + 7, module_shapes=MODULE_SHAPES,
                                 condition_hidden_dim=8, hidden_dim=8, id_dim=8,
                                 trunk_layers=2, trunk_hidden=16)
    out = net(torch.randn(2, 4, 23))     # 3D: 4 帧自动 mean
    assert set(out) == set(MODULE_SHAPES)
    assert out["m.attn_q"]["A"].shape == (2, 16, 64)
    assert out["m.attn_q"]["B"].shape == (2, 128, 16)


def test_film_gradients_flow():
    from mlvla.meta.hypernet import SharedFiLMABHyperNetwork
    torch.manual_seed(0)
    net = SharedFiLMABHyperNetwork(dino_dim=16, view_dim=7, module_shapes=MODULE_SHAPES,
                                   condition_hidden_dim=8, hidden_dim=8, dict_dim=4)
    out = net(torch.randn(2, 4, 16), torch.randn(2, 7))
    loss = sum(factors["A"].sum() + factors["B"].sum() for factors in out.values())
    loss.backward()
    # identity-at-init FiLM: gamma_out/beta_out weights start at 0, so pose_encoder
    # provably receives zero grad at step 0 and comes alive after one update —
    # exclude it; every other parameter must carry a nonzero gradient.
    grads = [p.grad for n, p in net.named_parameters()
             if p.grad is not None and not n.startswith("pose_encoder")]
    assert grads and all(g.abs().sum() > 0 for g in grads)
