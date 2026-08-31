"""Per-module per-factor RMS scales over oracle targets."""
from __future__ import annotations

import torch


def compute_rms_scales(domain_targets):
    sums: dict[tuple[str, str], float] = {}
    counts: dict[tuple[str, str], int] = {}
    # These reductions are tiny; torch's default thread pool makes each one
    # ~80x slower (OMP fork/join per op). Pin to 1 thread for this loop only.
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for modules_ in domain_targets.values():
            for key, factors in modules_.items():
                for factor in ("A", "B"):
                    tensor = factors[factor].detach().float()
                    slot = (key, factor)
                    sums[slot] = sums.get(slot, 0.0) + float(tensor.square().sum())
                    counts[slot] = counts.get(slot, 0) + tensor.numel()
    finally:
        torch.set_num_threads(prior_threads)
    return {
        key: {
            # Floor guards the 232 low-sensitivity modules whose oracle A/B are
            # zero in every train domain: rms=0 would turn target/scale into 0/0.
            factor: torch.tensor(max((sums[(key, factor)] / counts[(key, factor)]) ** 0.5, 1e-8))
            for factor in ("A", "B")
        }
        for key in {k for k, _ in sums}
    }


def scales_to_dict(scale_objects):
    return {
        key: {factor: float(value) for factor, value in factors.items()}
        for key, factors in scale_objects.items()
    }
