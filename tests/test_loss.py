from __future__ import annotations

from types import SimpleNamespace

import torch

from mert_loss import mert_encode, mert_loss, mert_perceptual_loss
from mert_loss.loss import MERTLoss

NUM_FAKE_LAYERS = 13  # embedding + 12 transformer layers (matches MERT-v1-95M)


class FakeMERTModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(1, 4)
        self.last_input_shape = None

    def forward(self, input_values: torch.Tensor, output_hidden_states: bool = True):
        self.last_input_shape = tuple(input_values.shape)
        x = input_values.unsqueeze(-1)
        hidden_states = tuple(
            self.proj(x * (0.5 ** i)) for i in range(NUM_FAKE_LAYERS)
        )
        return SimpleNamespace(hidden_states=hidden_states)


class FakeProcessor:
    """Minimal Wav2Vec2FeatureExtractor stand-in for tests."""
    def __call__(self, waveforms, sampling_rate=24000, return_tensors="pt", padding=True):
        import numpy as np
        import torch
        if isinstance(waveforms, np.ndarray):
            arr = torch.from_numpy(waveforms).float()
        else:
            arr = torch.as_tensor(waveforms, dtype=torch.float32)
        if arr.ndim == 1:
            arr = arr.unsqueeze(0)
        return {"input_values": arr}


def patch_model(monkeypatch):
    created = []

    def fake_from_pretrained(*args, **kwargs):
        model = FakeMERTModel()
        created.append(model)
        return model

    monkeypatch.setattr(
        "mert_loss.encoder.load_pretrained_model",
        lambda model_name: fake_from_pretrained(),
    )
    monkeypatch.setattr(
        "mert_loss.encoder.load_pretrained_processor",
        lambda model_name: FakeProcessor(),
    )
    return created


def test_backward_flows_to_pred(monkeypatch):
    patch_model(monkeypatch)

    loss_fn = MERTLoss(sample_rate=24_000)
    pred = torch.randn(2, 1000, requires_grad=True)
    target = torch.randn(2, 1000)

    loss = loss_fn(pred, target)
    loss.backward()

    assert loss.ndim == 0
    assert pred.grad is not None
    assert pred.grad.shape == pred.shape
    assert torch.isfinite(pred.grad).all()


def test_stereo_input_is_mixed_to_mono(monkeypatch):
    created = patch_model(monkeypatch)

    loss_fn = MERTLoss(sample_rate=24_000)
    pred = torch.randn(2, 2, 1000, requires_grad=True)
    target = torch.randn(2, 2, 1000)

    loss = loss_fn(pred, target)
    loss.backward()

    assert created
    assert created[0].last_input_shape == (2, 1000)


def test_truncate_alignment_handles_length_mismatch(monkeypatch):
    patch_model(monkeypatch)

    loss_fn = MERTLoss(sample_rate=24_000, align="truncate")
    pred = torch.randn(1, 1200, requires_grad=True)
    target = torch.randn(1, 1000)

    loss = loss_fn(pred, target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_mert_encode_returns_features_and_preserves_grad(monkeypatch):
    patch_model(monkeypatch)

    waveform = torch.randn(2, 1000, requires_grad=True)
    features = mert_encode(waveform, sample_rate=24_000, layers=[9])
    features.sum().backward()

    assert features.shape == (1, 2, 1000, 4)
    assert waveform.grad is not None
    assert waveform.grad.shape == waveform.shape
    assert torch.isfinite(waveform.grad).all()


def test_helper_functions_reuse_cached_instance(monkeypatch):
    created = patch_model(monkeypatch)

    waveform = torch.randn(1, 1000, requires_grad=True)
    target = torch.randn(1, 1000)

    features1 = mert_encode(waveform, sample_rate=24_000)
    features2 = mert_encode(waveform, sample_rate=24_000)
    loss1 = mert_loss(waveform, target, sample_rate=24_000)
    loss2 = mert_loss(waveform, target, sample_rate=24_000)

    assert torch.isfinite(features1).all()
    assert torch.isfinite(features2).all()
    assert torch.isfinite(loss1)
    assert torch.isfinite(loss2)
    assert len(created) == 1


def test_identical_waveform_has_zero_loss_and_perturbed_is_larger(monkeypatch):
    patch_model(monkeypatch)

    x = torch.randn(1, 1000)
    same_loss = mert_loss(x, x)

    torch.manual_seed(0)
    y = x + 0.5 * torch.randn_like(x)
    different_loss = mert_loss(y, x)

    assert torch.isclose(same_loss, torch.zeros_like(same_loss), atol=1e-6)
    assert different_loss > -1e-6


def test_aliases_match_primary(monkeypatch):
    patch_model(monkeypatch)

    pred = torch.randn(1, 1000)
    target = torch.randn(1, 1000)

    primary = mert_loss(pred, target, sample_rate=24_000)
    perceptual = mert_perceptual_loss(pred, target, sample_rate=24_000)

    assert torch.equal(primary, perceptual)


# --- Multi-layer tests ---


def test_multi_layer_backward_flows_to_pred(monkeypatch):
    patch_model(monkeypatch)

    loss_fn = MERTLoss(sample_rate=24_000, layers=[6, 7, 8, 9])
    pred = torch.randn(2, 1000, requires_grad=True)
    target = torch.randn(2, 1000)

    loss = loss_fn(pred, target)
    loss.backward()

    assert loss.ndim == 0
    assert pred.grad is not None
    assert pred.grad.shape == pred.shape
    assert torch.isfinite(pred.grad).all()


def test_multi_layer_identical_has_zero_loss(monkeypatch):
    patch_model(monkeypatch)

    x = torch.randn(1, 1000)
    loss_fn = MERTLoss(sample_rate=24_000, layers=[6, 7, 8, 9])
    loss = loss_fn(x, x)

    assert torch.isclose(loss, torch.zeros_like(loss), atol=1e-5)


def test_multi_layer_perturbed_is_positive(monkeypatch):
    patch_model(monkeypatch)

    x = torch.randn(1, 1000)
    torch.manual_seed(0)
    y = x + 0.5 * torch.randn_like(x)

    loss_fn = MERTLoss(sample_rate=24_000, layers=[6, 7, 8, 9])
    loss = loss_fn(y, x)

    assert loss > 0


# --- layer_weights tests ---


def test_layer_weights_backward_flows_to_pred(monkeypatch):
    patch_model(monkeypatch)

    loss_fn = MERTLoss(
        sample_rate=24_000,
        layers=[6, 7, 8, 9],
        layer_weights=[1.0, 0.5, 0.5, 1.0],
        loss_type="l1",
    )
    pred = torch.randn(2, 1000, requires_grad=True)
    target = torch.randn(2, 1000)

    loss = loss_fn(pred, target)
    loss.backward()

    assert loss.ndim == 0
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_layer_weights_identical_has_zero_loss(monkeypatch):
    patch_model(monkeypatch)

    x = torch.randn(1, 1000)
    loss_fn = MERTLoss(
        sample_rate=24_000,
        layers=[6, 8, 10],
        layer_weights=[2.0, 1.0, 0.5],
        loss_type="l1",
    )
    loss = loss_fn(x, x)

    assert torch.isclose(loss, torch.zeros_like(loss), atol=1e-6)
