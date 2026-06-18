"""Shared training utilities: reproducibility seeding, checkpoint I/O, and
a lightweight SpecAugment implementation applied to the encoder's frame
features (paper Sec. 4.4: F=27, T=100, 2 masks each)."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer, epoch: int, best_metric: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
        },
        path,
    )


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer=None, device: str = "cpu") -> dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def spec_augment(
    features: torch.Tensor,
    freq_mask_param: int = 27,
    time_mask_param: int = 100,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
) -> torch.Tensor:
    """Applies frequency and time masking to a [B, T, D] feature tensor in
    place of a spectrogram (the paper applies SpecAugment-style masking to
    the encoder's frame-level features rather than a raw mel-spectrogram,
    since WavLM is used as a fixed feature extractor)."""
    B, T, D = features.shape
    out = features.clone()

    for b in range(B):
        for _ in range(num_freq_masks):
            f = random.randint(0, min(freq_mask_param, D - 1))
            if f == 0:
                continue
            f0 = random.randint(0, max(D - f, 0))
            out[b, :, f0 : f0 + f] = 0.0

        for _ in range(num_time_masks):
            t = random.randint(0, min(time_mask_param, T - 1))
            if t == 0:
                continue
            t0 = random.randint(0, max(T - t, 0))
            out[b, t0 : t0 + t, :] = 0.0

    return out


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
