from __future__ import annotations

from typing import Optional, Union

import torch
from torch import nn

from .audio import ensure_batch, resample_if_needed, zero_mean_unit_var_norm

_DEFAULT_LAYERS = [6, 7, 8, 9]


def load_pretrained_model(model_name: str) -> nn.Module:
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError(
            "transformers is required to load MERT. "
            "Install with `pip install transformers<4.45`."
        ) from exc

    return AutoModel.from_pretrained(model_name, trust_remote_code=True)


def load_pretrained_processor(model_name: str):
    try:
        from transformers import Wav2Vec2FeatureExtractor
    except ImportError as exc:
        raise ImportError(
            "transformers is required to load MERT processor. "
            "Install with `pip install transformers<4.45`."
        ) from exc

    return Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)


def _normalize_layers(layers: Union[int, list[int], None]) -> list[int]:
    if layers is None:
        return list(_DEFAULT_LAYERS)
    if isinstance(layers, int):
        return [layers]
    return list(layers)


class MERTEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "m-a-p/MERT-v1-95M",
        model_sample_rate: int = 24_000,
        layers: Union[int, list[int], None] = None,
        normalize: bool = True,
        device: Optional[torch.device | str] = "cuda",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.model_sample_rate = model_sample_rate
        self.layers: list[int] = _normalize_layers(layers)
        self.normalize = normalize
        self.device = device

        self._model: Optional[nn.Module] = None
        self._processor = None
        self._loaded_device: Optional[torch.device] = None

    def _target_device(self, waveform: torch.Tensor) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return waveform.device

    def _ensure_model(self, device: torch.device) -> tuple[nn.Module, object]:
        if self._model is None:
            self._model = load_pretrained_model(self.model_name)
            self._model.eval()
            for parameter in self._model.parameters():
                parameter.requires_grad_(False)
            self._model.to(device)
            self._processor = load_pretrained_processor(self.model_name)
            self._loaded_device = device
        elif self._loaded_device != device:
            self._model.to(device)
            self._loaded_device = device

        return self._model, self._processor

    def _resolve_layers(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        num_hs = len(hidden_states)
        indices: list[int] = []
        for l in self.layers:
            idx = l if l >= 0 else num_hs + l
            if idx < 0 or idx >= num_hs:
                raise ValueError(
                    f"Invalid layer index {l}. "
                    f"Model returned {num_hs} hidden states."
                )
            indices.append(idx)
        return torch.stack([hidden_states[i] for i in indices])

    def forward(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        target_device = self._target_device(waveform)
        model, processor = self._ensure_model(target_device)

        # Shape handling: [T], [B, T], or [B, C, T] -> [B, T]
        waveform = ensure_batch(waveform)
        # Resample to model's expected rate if needed
        waveform = resample_if_needed(waveform, sample_rate, self.model_sample_rate)

        # Apply the same normalization the processor would do, but keep it
        # differentiable (the HF processor goes through CPU/numpy).
        if self.normalize and getattr(processor, "do_normalize", False):
            # Wav2Vec2FeatureExtractor uses per-waveform zero-mean unit-var
            input_values = zero_mean_unit_var_norm(waveform)
        else:
            input_values = waveform

        input_values = input_values.to(target_device)

        outputs = model(input_values=input_values, output_hidden_states=True)
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("MERT model did not return hidden states.")

        return self._resolve_layers(tuple(hidden_states))
