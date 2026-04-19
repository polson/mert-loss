from __future__ import annotations

import torch


EPS = 1e-7


def ensure_tensor(waveform: torch.Tensor) -> torch.Tensor:
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    if not waveform.is_floating_point():
        waveform = waveform.float()
    return waveform


def ensure_batch(waveform: torch.Tensor) -> torch.Tensor:
    waveform = ensure_tensor(waveform)

    if waveform.ndim == 1:
        return waveform.unsqueeze(0)
    if waveform.ndim == 2:
        return waveform
    if waveform.ndim == 3:
        return waveform.mean(dim=1)

    raise ValueError(
        "Expected waveform with shape [T], [B, T], or [B, C, T], "
        f"but got {tuple(waveform.shape)}"
    )


def resample_if_needed(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sample_rate: int,
) -> torch.Tensor:
    if sample_rate == target_sample_rate:
        return waveform

    try:
        import torchaudio.functional as AF
    except ImportError as exc:
        raise ImportError(
            "torchaudio is required when input audio must be resampled."
        ) from exc

    return AF.resample(waveform, sample_rate, target_sample_rate)


def zero_mean_unit_var_norm(waveform: torch.Tensor) -> torch.Tensor:
    """Differentiable per-waveform zero-mean unit-variance normalization.

    Mirrors ``Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm`` but stays
    on GPU and preserves the autograd graph.
    """
    mean = waveform.mean(dim=-1, keepdim=True)
    var = waveform.var(dim=-1, unbiased=False, keepdim=True)
    return (waveform - mean) / torch.sqrt(var + EPS)
