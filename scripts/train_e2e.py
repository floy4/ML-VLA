#!/usr/bin/env python3
import os

# Under torchrun (multi-process data parallel), pin this process to exactly
# one GPU BEFORE torch/jax backends initialize. Each child re-executes this
# script, so the remap lands before train_bridge's torch/jax imports bind.
if "LOCAL_RANK" in os.environ:
    # Index into the ORIGINAL visible list: setting CUDA_VISIBLE_DEVICES to the
    # bare LOCAL_RANK would point at absolute GPU ids (0,1,2...), which can hit
    # GPUs owned by other jobs. With no outer restriction, LOCAL_RANK is the id.
    _vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    _lr = int(os.environ["LOCAL_RANK"])
    os.environ["CUDA_VISIBLE_DEVICES"] = _vis.split(",")[_lr] if _vis else str(_lr)

from mlvla.meta.e2e.train_bridge import main

if __name__ == "__main__":
    main()
