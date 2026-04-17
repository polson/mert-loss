from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def clear_mert_caches():
    from mert_loss.cache import _ENCODER_CACHE, _LOSS_CACHE

    _ENCODER_CACHE.clear()
    _LOSS_CACHE.clear()
    yield
    _ENCODER_CACHE.clear()
    _LOSS_CACHE.clear()
