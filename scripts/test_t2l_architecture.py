#!/usr/bin/env python3
"""
Test script for T2L-style meta-network architecture.
Verifies forward pass and output shapes.
"""
import sys
from pathlib import Path

# Add ML-VLA src to path
mlvla_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(mlvla_root / "src"))

import torch
from mlvla.meta.t2l_style import (
    TwoBranchConditionEncoder,
    T2LStyleHyperNet,
    ReconstructionLoss,
    compute_cosine_similarity,
)


def test_architecture():
    """Test basic architecture and forward pass."""
    print("=" * 80)
    print("Testing T2L-style Meta-Network Architecture")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Simulate π0.5 module shapes (17 groups)
    # Format: "A{r}x{in}_B{out}x{r}" -> {"A": (r, in), "B": (out, r)}
    module_shapes = {
        "A16x512_B512x16": {"A": (16, 512), "B": (512, 16)},
        "A16x1024_B1024x16": {"A": (16, 1024), "B": (1024, 16)},
        "A16x2048_B2048x16": {"A": (16, 2048), "B": (2048, 16)},
    }
    max_layers = 27  # π0.5 has max 27 layers per module type (actual: 18-27)

    print(f"\nModule groups: {len(module_shapes)}")
    print(f"Max layers: {max_layers}")

    # Create condition encoder
    print("\n[1] Creating TwoBranchConditionEncoder...")
    condition_encoder = TwoBranchConditionEncoder(
        dino_dim=1024,
        geo_dim=7,
        hidden_dim=256,
        task_emb_size=256,
        fusion="additive",
    ).to(device)

    print(f"  Parameters: {sum(p.numel() for p in condition_encoder.parameters()):,}")

    # Create hypernetwork
    print("\n[2] Creating T2LStyleHyperNet...")
    hypernet = T2LStyleHyperNet(
        module_shapes=module_shapes,
        max_layers=max_layers,
        task_emb_size=256,
        depth_emb_size=64,
        type_emb_size=64,
        trunk_hidden=512,
        head_in_size=128,  # T2L-L configuration (~55M for full 17 groups)
        lora_rank=16,
        shared_AB_head=False,
    ).to(device)

    total_params = sum(p.numel() for p in hypernet.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  Target modules: {len(hypernet.target_modules)}")

    # Test forward pass
    print("\n[3] Testing forward pass...")
    batch_size = 4

    # Simulate evidence
    semantic = torch.randn(batch_size, 10, 4, 1024, device=device)  # [B, K, 4, D]
    geometric = torch.randn(batch_size, 10, 7, device=device)  # [B, K, 7]

    print(f"  Semantic input: {semantic.shape}")
    print(f"  Geometric input: {geometric.shape}")

    # Forward through condition encoder
    task_emb = condition_encoder(semantic, geometric)
    print(f"  Task embedding: {task_emb.shape}")
    assert task_emb.shape == (batch_size, 256), f"Expected [B, 256], got {task_emb.shape}"

    # Forward through hypernet for each module
    layer_indices = torch.randint(0, max_layers, (batch_size,), device=device)
    print(f"  Layer indices: {layer_indices.shape}")

    for module_key in hypernet.target_modules:
        A, B = hypernet(task_emb, layer_indices, module_key)
        expected_A_shape = (batch_size, 16, hypernet.in_features[module_key])
        expected_B_shape = (batch_size, hypernet.out_features[module_key], 16)

        print(f"\n  Module: {module_key}")
        print(f"    A: {A.shape} (expected {expected_A_shape})")
        print(f"    B: {B.shape} (expected {expected_B_shape})")

        assert A.shape == expected_A_shape, f"A shape mismatch: {A.shape} vs {expected_A_shape}"
        assert B.shape == expected_B_shape, f"B shape mismatch: {B.shape} vs {expected_B_shape}"

    # Test get_delta_weights
    print("\n[4] Testing get_delta_weights...")
    module_key = hypernet.target_modules[0]
    A, B = hypernet.get_delta_weights(
        layer_indices,
        module_key,
        task_emb,
        factorized=True,
    )
    print(f"  Factorized: A {A.shape}, B {B.shape}")

    deltaW = hypernet.get_delta_weights(
        layer_indices,
        module_key,
        task_emb,
        factorized=False,
    )
    expected_deltaW_shape = (batch_size, hypernet.out_features[module_key], hypernet.in_features[module_key])
    print(f"  Full deltaW: {deltaW.shape} (expected {expected_deltaW_shape})")
    assert deltaW.shape == expected_deltaW_shape

    # Test reconstruction loss
    print("\n[5] Testing reconstruction loss...")
    target_A = torch.randn_like(A)
    target_B = torch.randn_like(B)

    loss_fn = ReconstructionLoss(pred_z_score=False)
    loss_dict = loss_fn(A, B, target_A, target_B)

    print(f"  Loss: {loss_dict['loss'].item():.4f}")
    print(f"  L1 A: {loss_dict['l1_A'].item():.4f}")
    print(f"  L1 B: {loss_dict['l1_B'].item():.4f}")

    # Test cosine similarity
    print("\n[6] Testing cosine similarity...")
    cos_sim = compute_cosine_similarity(A, target_A)
    print(f"  Cosine similarity: {cos_sim.mean().item():.4f}")

    # Test gradient flow
    print("\n[7] Testing gradient flow...")
    loss = loss_dict["loss"]
    loss.backward()

    # Check gradients
    cond_grad_norm = sum(p.grad.norm().item() for p in condition_encoder.parameters() if p.grad is not None)
    hypernet_grad_norm = sum(p.grad.norm().item() for p in hypernet.parameters() if p.grad is not None)

    print(f"  Condition encoder grad norm: {cond_grad_norm:.4f}")
    print(f"  Hypernet grad norm: {hypernet_grad_norm:.4f}")

    assert cond_grad_norm > 0, "No gradients in condition encoder!"
    assert hypernet_grad_norm > 0, "No gradients in hypernet!"

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_architecture()
