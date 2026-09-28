"""
Standalone reconstruction trainer for T2L-style meta-network.
Pre-trains the hypernetwork by regressing to oracle expert LoRAs.
"""
from __future__ import annotations

import logging
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .hypernet import T2LStyleHyperNet
from .condition_encoder import TwoBranchConditionEncoder
from .losses import ReconstructionLoss, compute_cosine_similarity

logger = logging.getLogger(__name__)


class ReconstructionTrainer:
    """
    Trainer for reconstruction pre-training.

    Loads oracle expert LoRAs and trains the hypernetwork to regenerate them
    from evidence (DINO + pose).

    Args:
        hypernet: T2LStyleHyperNet instance
        condition_encoder: TwoBranchConditionEncoder instance
        oracle_loras: Dict mapping domain -> {module_key -> {"A": [L, r, in], "B": [L, out, r]}}
        evidence_bank: EvidenceBank instance for sampling evidence
        device: Torch device
        lr: Learning rate
        weight_decay: Weight decay
        save_dir: Directory to save checkpoints
    """

    def __init__(
        self,
        hypernet: T2LStyleHyperNet,
        condition_encoder: TwoBranchConditionEncoder,
        oracle_loras: dict[str, dict[str, dict[str, torch.Tensor]]],
        evidence_bank,  # EvidenceBank type
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        save_dir: str = "./outputs/t2l_recon",
    ):
        self.hypernet = hypernet.to(device)
        self.condition_encoder = condition_encoder.to(device)
        self.oracle_loras = oracle_loras
        self.evidence_bank = evidence_bank
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Combine parameters
        self.params = list(hypernet.parameters()) + list(condition_encoder.parameters())

        # Optimizer
        self.optimizer = optim.AdamW(
            self.params,
            lr=lr,
            weight_decay=weight_decay,
        )

        # Loss function
        self.recon_loss_fn = ReconstructionLoss(pred_z_score=False)

        # Extract domain list and module keys
        self.domains = list(oracle_loras.keys())
        self.module_keys = list(oracle_loras[self.domains[0]].keys())

        logger.info(f"ReconstructionTrainer initialized with {len(self.domains)} domains")
        logger.info(f"Module keys: {self.module_keys[:3]}... (showing first 3)")

    def train_epoch(
        self,
        epoch: int,
        batch_size: int = 8,
        n_batches: int = 100,
        log_freq: int = 10,
    ) -> dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Epoch number
            batch_size: Batch size
            n_batches: Number of batches per epoch
            log_freq: Logging frequency

        Returns:
            Dict with average losses and metrics
        """
        self.hypernet.train()
        self.condition_encoder.train()

        avg_losses = defaultdict(list)
        avg_cosine_sim = defaultdict(list)

        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch}")
        for batch_idx in pbar:
            # Sample batch of domains
            batch_domains = random.choices(self.domains, k=batch_size)

            # Sample evidence for each domain
            semantic_features = []
            geometric_features = []
            layer_indices_list = []

            for domain in batch_domains:
                # Sample evidence from EvidenceBank
                evidence = self.evidence_bank.sample(domain)
                semantic_features.append(evidence["dino"])
                geometric_features.append(evidence["pose"])

                # Sample random layer indices (for simplicity, use same for all modules)
                # In practice, you'd iterate over all layers
                layer_idx = random.randint(0, self.hypernet.max_layers - 1)
                layer_indices_list.append(layer_idx)

            # Stack features
            semantic_batch = torch.stack(semantic_features).to(self.device)
            geometric_batch = torch.stack(geometric_features).to(self.device)
            layer_indices = torch.tensor(layer_indices_list, device=self.device)

            # Forward pass through condition encoder
            task_emb = self.condition_encoder(semantic_batch, geometric_batch)

            # Compute reconstruction loss for each module
            total_loss = 0.0
            total_cosine = 0.0
            n_modules = 0

            for module_key in self.module_keys:
                # Get oracle LoRA for this module (same for all samples in batch)
                target_A = self.oracle_loras[batch_domains[0]][module_key]["A"][layer_indices[0]]
                target_B = self.oracle_loras[batch_domains[0]][module_key]["B"][layer_indices[0]]

                # Expand to batch
                target_A_batch = target_A.unsqueeze(0).expand(batch_size, -1, -1)
                target_B_batch = target_B.unsqueeze(0).expand(batch_size, -1, -1)

                # Generate LoRA
                pred_A, pred_B = self.hypernet(
                    task_emb,
                    layer_indices,
                    module_key,
                )

                # Compute loss
                loss_dict = self.recon_loss_fn(
                    pred_A,
                    pred_B,
                    target_A_batch,
                    target_B_batch,
                )

                total_loss += loss_dict["loss"]
                n_modules += 1

                # Compute cosine similarity (for logging)
                with torch.no_grad():
                    cos_A = compute_cosine_similarity(pred_A, target_A_batch).mean()
                    cos_B = compute_cosine_similarity(pred_B, target_B_batch).mean()
                    total_cosine += (cos_A + cos_B).item() / 2

            # Average over modules
            total_loss = total_loss / n_modules
            avg_cosine = total_cosine / n_modules

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Logging
            avg_losses["loss"].append(total_loss.item())
            avg_cosine_sim["cosine"].append(avg_cosine)

            if batch_idx % log_freq == 0:
                pbar.set_postfix({
                    "loss": f"{total_loss.item():.4f}",
                    "cosine": f"{avg_cosine:.4f}",
                })

        # Aggregate metrics
        metrics = {
            "train/loss": np.mean(avg_losses["loss"]),
            "train/cosine": np.mean(avg_cosine_sim["cosine"]),
        }

        logger.info(
            f"Epoch {epoch}: loss={metrics['train/loss']:.4f}, "
            f"cosine={metrics['train/cosine']:.4f}"
        )

        return metrics

    def validate(
        self,
        val_domains: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Validate on held-out domains.

        Args:
            val_domains: List of validation domains (default: all domains)

        Returns:
            Dict with validation metrics
        """
        self.hypernet.eval()
        self.condition_encoder.eval()

        if val_domains is None:
            val_domains = self.domains

        all_cosine_sim = []
        all_l1_err = []

        with torch.no_grad():
            for domain in val_domains:
                # Sample evidence
                evidence = self.evidence_bank.sample(domain)
                semantic = evidence["dino"].unsqueeze(0).to(self.device)
                geometric = evidence["pose"].unsqueeze(0).to(self.device)

                # Forward pass
                task_emb = self.condition_encoder(semantic, geometric)

                # Check each module
                for module_key in self.module_keys:
                    for layer_idx in range(min(10, self.hypernet.max_layers)):  # Check first 10 layers
                        layer_indices = torch.tensor([layer_idx], device=self.device)

                        # Get oracle
                        target_A = self.oracle_loras[domain][module_key]["A"][layer_idx]
                        target_B = self.oracle_loras[domain][module_key]["B"][layer_idx]

                        # Generate
                        pred_A, pred_B = self.hypernet(
                            task_emb,
                            layer_indices,
                            module_key,
                        )

                        # Metrics
                        cos_A = compute_cosine_similarity(pred_A, target_A).item()
                        cos_B = compute_cosine_similarity(pred_B, target_B).item()
                        all_cosine_sim.append((cos_A + cos_B) / 2)

                        l1_A = torch.abs(pred_A - target_A).mean().item()
                        l1_B = torch.abs(pred_B - target_B).mean().item()
                        all_l1_err.append((l1_A + l1_B) / 2)

        metrics = {
            "val/cosine": np.mean(all_cosine_sim),
            "val/l1_err": np.mean(all_l1_err),
        }

        logger.info(
            f"Validation: cosine={metrics['val/cosine']:.4f}, "
            f"l1_err={metrics['val/l1_err']:.6f}"
        )

        return metrics

    def train(
        self,
        n_epochs: int = 100,
        batch_size: int = 8,
        n_batches_per_epoch: int = 100,
        val_freq: int = 10,
        log_freq: int = 10,
    ):
        """
        Full training loop.

        Args:
            n_epochs: Number of epochs
            batch_size: Batch size
            n_batches_per_epoch: Batches per epoch
            val_freq: Validation frequency (epochs)
            log_freq: Logging frequency (batches)
        """
        best_cosine = 0.0

        for epoch in range(1, n_epochs + 1):
            train_metrics = self.train_epoch(
                epoch,
                batch_size=batch_size,
                n_batches=n_batches_per_epoch,
                log_freq=log_freq,
            )

            if epoch % val_freq == 0:
                val_metrics = self.validate()

                # Save best checkpoint
                if val_metrics["val/cosine"] > best_cosine:
                    best_cosine = val_metrics["val/cosine"]
                    self.save_checkpoint("best.pt")
                    logger.info(f"New best checkpoint: cosine={best_cosine:.4f}")

        # Save final checkpoint
        self.save_checkpoint("final.pt")
        logger.info(f"Training complete. Best cosine: {best_cosine:.4f}")

    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        checkpoint = {
            "hypernet": self.hypernet.state_dict(),
            "condition_encoder": self.condition_encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        path = self.save_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.hypernet.load_state_dict(checkpoint["hypernet"])
        self.condition_encoder.load_state_dict(checkpoint["condition_encoder"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info(f"Checkpoint loaded: {path}")
