from __future__ import annotations

from typing import Optional, Union

import torch

from .cache import get_cached_encoder, get_cached_loss

__all__ = ["mert_encode", "mert_loss", "mert_perceptual_loss"]


def mert_encode(
    waveform: torch.Tensor,
    sample_rate: int = 44_100,
    model_name: str = "m-a-p/MERT-v1-95M",
    layers: Union[int, list[int], None] = None,
    normalize: bool = True,
    device: Optional[torch.device | str] = "cuda",
) -> torch.Tensor:
    encoder = get_cached_encoder(
        model_name=model_name,
        layers=layers,
        normalize=normalize,
        device=device,
    )
    return encoder(waveform, sample_rate=sample_rate)


def mert_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int = 44_100,
    model_name: str = "m-a-p/MERT-v1-95M",
    layers: Union[int, list[int], None] = None,
    layer_weights: Union[float, list[float], None] = None,
    reduction: str = "mean",
    normalize: bool = True,
    detach_target: bool = True,
    align: str = "truncate",
    loss_type: str = "cosine",
    device: Optional[torch.device | str] = "cuda",
) -> torch.Tensor:
    loss_fn = get_cached_loss(
        model_name=model_name,
        sample_rate=sample_rate,
        layers=layers,
        layer_weights=layer_weights,
        reduction=reduction,
        normalize=normalize,
        detach_target=detach_target,
        align=align,
        loss_type=loss_type,
        device=device,
    )
    return loss_fn(pred, target)


# Alias
def mert_perceptual_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int = 44_100,
    model_name: str = "m-a-p/MERT-v1-95M",
    layers: Union[int, list[int], None] = None,
    layer_weights: Union[float, list[float], None] = None,
    reduction: str = "mean",
    normalize: bool = True,
    detach_target: bool = True,
    align: str = "truncate",
    loss_type: str = "cosine",
    device: Optional[torch.device | str] = "cuda",
) -> torch.Tensor:
    return mert_loss(
        pred=pred,
        target=target,
        sample_rate=sample_rate,
        model_name=model_name,
        layers=layers,
        layer_weights=layer_weights,
        reduction=reduction,
        normalize=normalize,
        detach_target=detach_target,
        align=align,
        loss_type=loss_type,
        device=device,
    )
