from __future__ import annotations

from typing import Optional, Union

import torch

from .encoder import MERTEncoder, _normalize_layers
from .loss import MERTLoss

_ENCODER_CACHE: dict[tuple[object, ...], MERTEncoder] = {}
_LOSS_CACHE: dict[tuple[object, ...], MERTLoss] = {}


def get_cached_encoder(
    model_name: str = "m-a-p/MERT-v1-95M",
    layers: Union[int, list[int], None] = None,
    normalize: bool = True,
    device: Optional[torch.device | str] = None,
) -> MERTEncoder:
    resolved = _normalize_layers(layers)
    key = (
        model_name,
        tuple(resolved),
        normalize,
        str(device) if device is not None else None,
    )

    if key not in _ENCODER_CACHE:
        _ENCODER_CACHE[key] = MERTEncoder(
            model_name=model_name,
            model_sample_rate=24_000,
            layers=resolved,
            normalize=normalize,
            device=device,
        )

    return _ENCODER_CACHE[key]


def get_cached_loss(
    model_name: str = "m-a-p/MERT-v1-95M",
    sample_rate: int = 44_100,
    layers: Union[int, list[int], None] = None,
    reduction: str = "mean",
    normalize: bool = True,
    detach_target: bool = True,
    align: str = "truncate",
    device: Optional[torch.device | str] = None,
) -> MERTLoss:
    resolved = _normalize_layers(layers)
    key = (
        model_name,
        sample_rate,
        tuple(resolved),
        reduction,
        normalize,
        detach_target,
        align,
        str(device) if device is not None else None,
    )

    if key not in _LOSS_CACHE:
        _LOSS_CACHE[key] = MERTLoss(
            sample_rate=sample_rate,
            layers=resolved,
            reduction=reduction,
            detach_target=detach_target,
            align=align,
            encoder=get_cached_encoder(
                model_name=model_name,
                layers=resolved,
                normalize=normalize,
                device=device,
            ),
        )

    return _LOSS_CACHE[key]
