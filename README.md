# mert-loss

Differentiable [MERT](https://huggingface.co/m-a-p/MERT-v1-95M)-based perceptual loss for raw waveforms.

## Install

```bash
pip install git+https://github.com/polson/mert-loss.git
```

For development:

```bash
pip install -e .[dev]
```

## Quick start

```python
import torch
from mert_loss import mert_perceptual_loss

pred = torch.randn(2, 44100 * 2, requires_grad=True)
target = torch.randn(2, 44100 * 2)

loss = mert_perceptual_loss(pred, target)
loss.backward()
```

### Recommended weight for stem separation

Use MERT as a regularizer at roughly **4% of total loss**:

```python
total_loss = l1_loss + 0.04 * mert_perceptual_loss(pred, target)
```

For separation, layers 6–9 work well. For generation, try layers 9–12.

## API

```python
from mert_loss import mert_encode, mert_perceptual_loss
```

- `mert_encode(waveform, ...)` → MERT hidden-state features
- `mert_perceptual_loss(pred, target, ...)` → distance on layer-normalized embeddings

Supports waveform shapes `[T]`, `[B, T]`, `[B, C, T]` (multi-channel is averaged to mono). Input is assumed to be **44.1 kHz** and resampled internally to 24 kHz.

### Kwargs

Both functions accept:

- `sample_rate` — input sample rate (default `44100`)
- `layers` — single layer (`int`) or list of layers to use (default `[6, 7, 8, 9]`)
- `normalize` — normalize features using the model's `Wav2Vec2FeatureExtractor` config (default `True`)
- `device` — device for the encoder (default `"cuda"`)

`mert_perceptual_loss` also accepts:

- `loss_type` — `"cosine"` (default), `"mse"`, or `"l1"`
- `layer_weights` — per-layer weights for weighted-sum loss (default `None`). When provided, switches from blended-embedding comparison to per-layer loss with weighted sum. Accepts a list of floats (one per layer) or a single float (equal weight for all layers).
- `reduction` — `"mean"`, `"sum"`, or `"none"` (default `"mean"`, only applies when `layer_weights` is `None`)
- `detach_target` — run target without gradients (default `True`)
- `align` — how to handle time mismatches: `"truncate"` or `"strict"` (default `"truncate"`)

### Class-based API

```python
from mert_loss.loss import MERTLoss

# Default: cosine distance on blended multi-layer embedding
loss_fn = MERTLoss(layers=[6, 7, 8, 9])
loss = loss_fn(pred, target)

# Per-layer L1 with custom weights (L3AC-compatible)
loss_fn = MERTLoss(layers=[6, 8, 10], layer_weights=[2.0, 1.0, 0.5], loss_type="l1")
loss = loss_fn(pred, target)
```

No trainable parameters — MERT stays frozen.

## Notes

- The first call downloads [`m-a-p/MERT-v1-95M`](https://huggingface.co/m-a-p/MERT-v1-95M) from Hugging Face (~380 MB). After that it's cached at `~/.cache/huggingface/hub`.
- Requires **Python 3.10+**, **transformers >= 5.5, < 6**.
- The model has its own license — check the [model card](https://huggingface.co/m-a-p/MERT-v1-95M) before use.
