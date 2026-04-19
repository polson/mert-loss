from __future__ import annotations

from typing import Literal, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from .encoder import MERTEncoder, _normalize_layers

_DEFAULT_LAYERS = [6, 7, 8, 9]

AlignMode = Literal["truncate", "strict"]
LossType = Literal["cosine", "mse", "l1"]


def _normalize_weights(
    weights: Union[float, list[float], None], num_layers: int,
) -> Optional[torch.Tensor]:
    if weights is None:
        return None
    if isinstance(weights, (int, float)):
        weights = [float(weights)] * num_layers
    weights = [float(w) for w in weights]
    if len(weights) != num_layers:
        raise ValueError(
            f"layer_weights has {len(weights)} elements but "
            f"{num_layers} layers were selected"
        )
    return torch.tensor(weights)


class MERTLoss(nn.Module):
    def __init__(
        self,
        model_name: str = "m-a-p/MERT-v1-95M",
        sample_rate: int = 44_100,
        layers: Union[int, list[int], None] = None,
        layer_weights: Union[float, list[float], None] = None,
        reduction: str = "mean",
        normalize: bool = True,
        detach_target: bool = True,
        align: AlignMode = "truncate",
        loss_type: LossType = "cosine",
        device: Optional[torch.device | str] = "cuda",
        encoder: Optional[MERTEncoder] = None,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.reduction = reduction
        self.detach_target = detach_target
        self.align = align
        self.loss_type = loss_type
        resolved = _normalize_layers(layers)
        self.encoder = encoder or MERTEncoder(
            model_name=model_name,
            model_sample_rate=24_000,
            layers=resolved,
            normalize=normalize,
            device=device,
        )
        self._layers = resolved
        self._layer_weights = _normalize_weights(layer_weights, len(resolved))

    def _align_features(
        self,
        pred_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pred_features.ndim != 3 or target_features.ndim != 3:
            raise ValueError(
                "Expected encoded features with shape [B, T, C], "
                f"got {tuple(pred_features.shape)} and {tuple(target_features.shape)}"
            )

        if pred_features.shape[0] != target_features.shape[0]:
            raise ValueError(
                "Batch size mismatch between pred and target features: "
                f"{pred_features.shape[0]} vs {target_features.shape[0]}"
            )

        if pred_features.shape[-1] != target_features.shape[-1]:
            raise ValueError(
                "Feature dimension mismatch between pred and target features: "
                f"{pred_features.shape[-1]} vs {target_features.shape[-1]}"
            )

        if pred_features.shape[1] == target_features.shape[1]:
            return pred_features, target_features

        if self.align == "strict":
            raise ValueError(
                "Time dimension mismatch between pred and target features: "
                f"{pred_features.shape[1]} vs {target_features.shape[1]}"
            )

        if self.align != "truncate":
            raise ValueError(f"Unsupported align mode: {self.align}")

        time_steps = min(pred_features.shape[1], target_features.shape[1])
        return pred_features[:, :time_steps], target_features[:, :time_steps]

    def _mix_layers(self, stacked: torch.Tensor) -> torch.Tensor:
        # stacked: [num_layers, B, T, C]
        # 1. Layer-norm each feature vector
        normed = F.layer_norm(stacked, stacked.shape[-1:])
        # 2. Average
        mixed = normed.mean(dim=0)  # [B, T, C]
        # 3. Layer-norm the mixture
        return F.layer_norm(mixed, mixed.shape[-1:])

    def _compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute element-wise distance between aligned feature tensors."""
        if self.loss_type == "cosine":
            sim = F.cosine_similarity(pred, target, dim=-1)
            return 1 - sim
        elif self.loss_type == "mse":
            return F.mse_loss(pred, target, reduction="none").mean(dim=-1)
        else:  # l1
            return F.l1_loss(pred, target, reduction="none").mean(dim=-1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_features = self.encoder(pred, sample_rate=self.sample_rate)

        if self.detach_target:
            with torch.no_grad():
                target_features = self.encoder(target, sample_rate=self.sample_rate)
        else:
            target_features = self.encoder(target, sample_rate=self.sample_rate)

        # Per-layer weighted loss (L3AC style)
        if self._layer_weights is not None:
            weights = self._layer_weights.to(pred_features.device)
            total = torch.tensor(0.0, device=pred_features.device)
            for i in range(pred_features.shape[0]):
                pf = F.layer_norm(pred_features[i], pred_features.shape[-1:])
                tf = F.layer_norm(target_features[i], target_features.shape[-1:])
                pf, tf = self._align_features(pf, tf)
                total = total + self._compute_loss(pf, tf).mean() * weights[i]
            return total / weights.sum()

        # Blended embedding comparison (original style)
        if len(self._layers) > 1:
            pred_features = self._mix_layers(pred_features)
            target_features = self._mix_layers(target_features)
        else:
            pred_features = pred_features.squeeze(0)
            target_features = target_features.squeeze(0)
            pred_features = F.layer_norm(pred_features, pred_features.shape[-1:])
            target_features = F.layer_norm(target_features, target_features.shape[-1:])

        pred_features, target_features = self._align_features(
            pred_features,
            target_features,
        )

        loss = self._compute_loss(pred_features, target_features)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
