"""Test configuration and shared fixtures."""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
import pytorch_lightning as pl
import torch

from lightning_profiler.device_bottleneck import (
    DeviceBottleneckCallback,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def callback_default() -> DeviceBottleneckCallback:
    """A callback with default parameters (no CUDA mock)."""
    return DeviceBottleneckCallback()


@pytest.fixture
def callback_with_output(tmp_path) -> DeviceBottleneckCallback:
    """Callback configured to write metrics to a temp file."""
    return DeviceBottleneckCallback(
        log_every_n_steps=0,
        output_path=str(tmp_path / "metrics.json"),
        warmup_steps=0,
    )


@pytest.fixture
def mock_cuda_available() -> Generator[None, None, None]:
    """Force ``torch.cuda.is_available()`` to return ``True``.

    Also provides a mock ``torch.cuda.Event`` so that CUDA-event code paths
    can be exercised without actual GPU hardware.
    """
    mock_event = MagicMock(spec=torch.cuda.Event)
    # `.record()` and `.synchronize()` are no-ops
    # We control `elapsed_time` via call attributes
    mock_event.elapsed_time.return_value = 42.0  # ms

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.Event", return_value=mock_event),
        patch("torch.cuda.synchronize"),
    ):
        yield


# ---------------------------------------------------------------------------
# Reusable helpers
# ---------------------------------------------------------------------------


def make_mock_trainer() -> MagicMock:
    """Create a lightweight mock Lightning Trainer."""
    trainer = MagicMock(spec=pl.Trainer)
    trainer.global_rank = 0
    trainer.num_devices = 1
    return trainer


def make_mock_module() -> MagicMock:
    """Create a lightweight mock LightningModule."""
    return MagicMock(spec=pl.LightningModule)
