"""Frozen DINOv3 ViT encoder for VTT extraction.

Loads the model once and exposes a batched `encode_frames` that returns
both the CLS token (1024-dim) and the 14x14 patch-token grid (196x1024).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoModel

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DINOV3_INPUT_SIZE = 224


def load_dinov3(
    checkpoint_path: str | Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.nn.Module, int, int, int]:
    """Load frozen DINOv3 ViT. Returns (model, hidden_size, num_register_tokens, patch_size)."""
    config = AutoConfig.from_pretrained(str(checkpoint_path), local_files_only=True)
    model = AutoModel.from_pretrained(
        str(checkpoint_path), torch_dtype=dtype, local_files_only=True
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    hidden_size = int(getattr(config, "hidden_size", 1024))
    num_register = int(getattr(config, "num_register_tokens", 4))
    patch_size = int(getattr(config, "patch_size", 16))
    return model, hidden_size, num_register, patch_size


def _preprocess(frames: Sequence[np.ndarray]) -> torch.Tensor:
    """Resize each H×W×3 uint8 frame to 224×224, normalize, return [N,3,224,224] float32."""
    out = np.zeros((len(frames), 3, DINOV3_INPUT_SIZE, DINOV3_INPUT_SIZE), dtype=np.float32)
    for i, frame in enumerate(frames):
        if frame.dtype != np.uint8:
            raise ValueError(f"frame[{i}] dtype={frame.dtype}, expected uint8")
        img = Image.fromarray(frame).resize(
            (DINOV3_INPUT_SIZE, DINOV3_INPUT_SIZE), Image.BILINEAR
        )
        arr = np.asarray(img, dtype=np.float32) / 255.0  # H,W,3
        arr = (arr - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, dtype=np.float32)
        out[i] = arr.transpose(2, 0, 1)  # 3,H,W
    return torch.from_numpy(out)


@torch.inference_mode()
def encode_frames(
    model: torch.nn.Module,
    frames_np_uint8: Sequence[np.ndarray],
    device: str = "cuda",
    num_register_tokens: int = 4,
    hidden_size: int = 1024,
) -> dict[str, np.ndarray]:
    """Encode a batch of uint8 frames. Returns {'cls': [N,1024], 'patch': [N,196,1024]} as float32."""
    if not frames_np_uint8:
        raise ValueError("empty frame batch")
    pixel_values = _preprocess(frames_np_uint8).to(device=device, dtype=model.dtype)
    out = model(pixel_values=pixel_values)
    last = out.last_hidden_state  # [N, 1+num_register+196, hidden]
    expected_total = 1 + num_register_tokens + (DINOV3_INPUT_SIZE // 16) ** 2
    if last.shape != (len(frames_np_uint8), expected_total, hidden_size):
        raise RuntimeError(
            f"DINOv3 output shape mismatch: got {tuple(last.shape)}, "
            f"expected (N, {expected_total}, {hidden_size})"
        )
    cls = last[:, 0, :].float().cpu().numpy()
    patch_start = 1 + num_register_tokens
    patch_end = patch_start + (DINOV3_INPUT_SIZE // 16) ** 2
    patch = last[:, patch_start:patch_end, :].float().cpu().numpy()
    return {"cls": cls, "patch": patch}
