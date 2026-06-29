"""Tests for the ``_compute_verdict`` helper and the ``DeviceBottleneckCallback``."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import pytorch_lightning as pl
import torch

from lightning_profiler.device_bottleneck import (
    DeviceBottleneckCallback,
    StepMetrics,
    _compute_verdict,
)

# ======================================================================
# _compute_verdict — pure-logic unit tests
# ======================================================================


class TestComputeVerdict:
    def test_cpu_bottleneck_below_threshold(self) -> None:
        assert _compute_verdict(0.50, cpu_threshold=0.85) == "cpu_bottleneck"

    def test_cpu_bottleneck_at_threshold(self) -> None:
        assert _compute_verdict(0.84, cpu_threshold=0.85) == "cpu_bottleneck"

    def test_gpu_bottleneck_above_threshold(self) -> None:
        assert _compute_verdict(0.99, gpu_threshold=0.98) == "gpu_bottleneck"

    def test_gpu_bottleneck_at_threshold(self) -> None:
        assert _compute_verdict(0.99, gpu_threshold=0.98) == "gpu_bottleneck"

    def test_balanced_mid_range(self) -> None:
        assert _compute_verdict(0.90) == "balanced"

    def test_custom_thresholds(self) -> None:
        assert _compute_verdict(0.70, cpu_threshold=0.80, gpu_threshold=0.95) == "cpu_bottleneck"
        assert _compute_verdict(0.85, cpu_threshold=0.80, gpu_threshold=0.95) == "balanced"
        assert _compute_verdict(0.96, cpu_threshold=0.80, gpu_threshold=0.95) == "gpu_bottleneck"


# ======================================================================
# StepMetrics dataclass
# ======================================================================


class TestStepMetrics:
    def test_fields(self) -> None:
        m = StepMetrics(
            step_idx=1,
            wall_time_ms=500.0,
            gpu_time_ms=200.0,
            bottleneck_ratio=0.4,
            verdict="cpu_bottleneck",
        )
        assert m.step_idx == 1
        assert m.wall_time_ms == 500.0
        assert m.gpu_time_ms == 200.0
        assert m.bottleneck_ratio == 0.4
        assert m.verdict == "cpu_bottleneck"


# ======================================================================
# DeviceBottleneckCallback — construction & validation
# ======================================================================


class TestCallbackInit:
    def test_defaults(self) -> None:
        cb = DeviceBottleneckCallback()
        assert cb.cpu_threshold == 0.85
        assert cb.gpu_threshold == 0.98
        assert cb.log_every_n_steps == 50
        assert cb.warmup_steps == 5
        assert cb.output_path is None

    def test_custom_parameters(self) -> None:
        cb = DeviceBottleneckCallback(
            cpu_threshold=0.80,
            gpu_threshold=0.95,
            log_every_n_steps=10,
            output_path="/tmp/metrics.json",
            warmup_steps=2,
        )
        assert cb.cpu_threshold == 0.80
        assert cb.gpu_threshold == 0.95
        assert cb.log_every_n_steps == 10
        assert cb.warmup_steps == 2
        assert str(cb.output_path) == "/tmp/metrics.json"

    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    def test_invalid_cpu_threshold(self, bad: float) -> None:
        with pytest.raises(ValueError, match="cpu_threshold"):
            DeviceBottleneckCallback(cpu_threshold=bad)

    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    def test_invalid_gpu_threshold(self, bad: float) -> None:
        with pytest.raises(ValueError, match="gpu_threshold"):
            DeviceBottleneckCallback(gpu_threshold=bad)

    def test_cpu_threshold_greater_than_gpu(self) -> None:
        with pytest.raises(ValueError, match="cpu_threshold"):
            DeviceBottleneckCallback(cpu_threshold=0.95, gpu_threshold=0.85)

    def test_negative_log_every_n(self) -> None:
        with pytest.raises(ValueError, match="log_every_n_steps"):
            DeviceBottleneckCallback(log_every_n_steps=-1)


# ======================================================================
# DeviceBottleneckCallback — behaviour (no CUDA)
# ======================================================================


class TestCallbackNoCuda:
    def test_no_cuda_does_not_crash(self, callback_default: DeviceBottleneckCallback) -> None:
        """Without CUDA, hooks should be no-ops."""
        mock_trainer = MagicMock(spec=pl.Trainer)
        mock_module = MagicMock(spec=pl.LightningModule)

        callback_default.on_train_batch_start(mock_trainer, mock_module, None, 0)
        callback_default.on_train_batch_end(mock_trainer, mock_module, None, None, 0)
        callback_default.on_train_end(mock_trainer, mock_module)

        assert callback_default.metrics == []

    def test_summary_without_cuda(self, callback_default: DeviceBottleneckCallback) -> None:
        s = callback_default.summary
        assert s["total_steps"] == 0
        assert s["avg_bottleneck_ratio"] == 0.0

    def test_properties(self, callback_default: DeviceBottleneckCallback) -> None:
        assert callback_default.metrics == []


# ======================================================================
# DeviceBottleneckCallback — behaviour (with mocked CUDA)
# ======================================================================


def _run_steps(
    callback: DeviceBottleneckCallback,
    n: int,
    *,
    wall_deltas: list[float] | None = None,
) -> None:
    """Simulate ``n`` training steps, advancing ``time.perf_counter`` by ``wall_deltas``.

    Each step calls on_train_batch_start → small clock advance → on_train_batch_end.

    Parameters
    ----------
    wall_deltas:
        Per-step deltas for ``time.perf_counter`` that are added *before* each
        step's on_train_batch_start call. This controls the inter-batch gap.
        If ``None``, a fixed 50 ms delta is used for every step.
    """
    if wall_deltas is None:
        wall_deltas = [0.050] * n

    mock_trainer = MagicMock(spec=pl.Trainer)
    mock_module = MagicMock(spec=pl.LightningModule)
    fake_clock = 0.0

    with patch("time.perf_counter", wraps=lambda: fake_clock):
        for i in range(n):
            # Advance the clock to simulate the inter-batch gap (CPU work)
            fake_clock += wall_deltas[i]

            callback.on_train_batch_start(mock_trainer, mock_module, None, i)

            # Move the clock forward a bit within the step (GPU work)
            fake_clock += 0.010

            callback.on_train_batch_end(mock_trainer, mock_module, None, None, i)


class TestCallbackWithCuda:
    def test_warmup_steps_skipped(self, mock_cuda_available) -> None:
        """Warmup steps produce no metrics.

        Steps 1-3 are warmup.  Step 1 also has no ``_prev_step_end_wall``.
        So steps 1-3 all skip.  Steps 4 and 5 are recorded → 2 metrics.
        """
        cb = DeviceBottleneckCallback(warmup_steps=3, log_every_n_steps=0)
        _run_steps(cb, 5)
        cb._flush_pending_events()
        assert len(cb.metrics) == 2  # steps 4 and 5

    def test_metrics_recorded_after_warmup(self, mock_cuda_available) -> None:
        """After warmup, metrics are collected with correct structure.

        With 4 steps (warmup=0), step 1 has no prev_step_end_wall → skipped.
        Steps 2, 3, 4 → 3 metrics.
        """
        cb = DeviceBottleneckCallback(warmup_steps=0, log_every_n_steps=0, output_path=None)
        _run_steps(cb, 4)
        cb._flush_pending_events()

        assert len(cb.metrics) == 3
        for m in cb.metrics:
            assert m.step_idx > 1
            assert m.wall_time_ms > 0
            assert m.gpu_time_ms == 42.0  # our mock returns 42 ms
            assert m.verdict in ("cpu_bottleneck", "gpu_bottleneck", "balanced")

    def test_summary_aggregation(self, mock_cuda_available) -> None:
        cb = DeviceBottleneckCallback(warmup_steps=0, log_every_n_steps=0)

        n_steps = 10
        _run_steps(cb, n_steps)
        cb._flush_pending_events()

        s = cb.summary
        # 10 steps → 9 metrics (first step has no prev_step_end_wall)
        assert s["total_steps"] == n_steps - 1
        assert s["avg_wall_time_ms"] > 0
        assert s["avg_gpu_time_ms"] == 42.0
        assert s["avg_bottleneck_ratio"] > 0

    def test_json_output_written(self, tmp_path, mock_cuda_available) -> None:
        import json

        output_file = tmp_path / "metrics.json"
        cb = DeviceBottleneckCallback(
            warmup_steps=0, log_every_n_steps=0, output_path=str(output_file)
        )

        _run_steps(cb, 3)
        cb.on_train_end(MagicMock(spec=pl.Trainer), MagicMock(spec=pl.LightningModule))

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        # 3 steps → 2 metrics (step 1 skipped)
        assert len(data) == 2
        assert data[0]["step"] == 2
        assert "verdict" in data[0]

    def test_logging_disabled(self, mock_cuda_available) -> None:
        """log_every_n_steps=0 suppresses periodic flushing."""
        cb = DeviceBottleneckCallback(warmup_steps=0, log_every_n_steps=0)
        _run_steps(cb, 10)

        # Without a flush, events are pending, not yet metrics
        assert len(cb._pending_events) > 0
        assert cb.metrics == []

        # Explicit flush
        cb._flush_pending_events()
        assert len(cb.metrics) == 9

    def test_auto_flush_at_log_interval(self, mock_cuda_available) -> None:
        """When ``log_every_n_steps`` is set, events are flushed automatically."""
        cb = DeviceBottleneckCallback(warmup_steps=0, log_every_n_steps=5)
        _run_steps(cb, 11)  # flushes at steps 5, 10

        # Step 1: skip (no prev wall)
        # Steps 2-5: flushed at step 5 → 4 metrics
        # Steps 6-10: flushed at step 10 → 5 metrics
        # Step 11: pending, not flushed yet
        assert len(cb._pending_events) == 1  # step 11 still pending
        assert len(cb.metrics) == 9  # 4 + 5

        # Final flush
        cb._flush_pending_events()
        assert len(cb.metrics) == 10


# ======================================================================
# Edge cases
# ======================================================================


class TestEdgeCases:
    def test_zero_wall_time(self, mock_cuda_available) -> None:
        """If wall_time is zero or negative (clock went backward), ratio defaults to 1.0."""
        cb = DeviceBottleneckCallback(warmup_steps=0, log_every_n_steps=0)

        mock_trainer = MagicMock(spec=pl.Trainer)
        mock_module = MagicMock(spec=pl.LightningModule)
        fake_clock = 1000.0

        with patch("time.perf_counter", wraps=lambda: fake_clock):
            # Step 1: set baseline
            cb.on_train_batch_start(mock_trainer, mock_module, None, 0)
            fake_clock += 0.010
            cb.on_train_batch_end(mock_trainer, mock_module, None, None, 0)

            # Step 2: clock goes *backwards* → wall_ms = 0 or negative
            fake_clock -= 0.100
            cb.on_train_batch_start(mock_trainer, mock_module, None, 1)
            fake_clock += 0.010
            cb.on_train_batch_end(mock_trainer, mock_module, None, None, 1)

        cb._flush_pending_events()

        assert len(cb.metrics) == 1
        assert cb.metrics[0].bottleneck_ratio == 1.0

    def test_no_training_batches(self, callback_default: DeviceBottleneckCallback) -> None:
        """Calling on_train_end with zero batches should not crash."""
        mock_trainer = MagicMock(spec=pl.Trainer)
        mock_module = MagicMock(spec=pl.LightningModule)
        callback_default.on_train_end(mock_trainer, mock_module)
        assert callback_default.summary["total_steps"] == 0

    def test_cuda_without_start_guard(self, mock_cuda_available) -> None:
        """Calling on_train_batch_end without on_train_batch_start should be a no-op."""
        cb = DeviceBottleneckCallback(warmup_steps=0, log_every_n_steps=0)
        mock_trainer = MagicMock(spec=pl.Trainer)
        mock_module = MagicMock(spec=pl.LightningModule)

        cb.on_train_batch_end(mock_trainer, mock_module, None, None, 0)
        assert cb.metrics == []
        assert cb._pending_events == []

    def test_internal_state_advances(self, mock_cuda_available) -> None:
        """After each step, internal state should advance correctly."""
        cb = DeviceBottleneckCallback(warmup_steps=0, log_every_n_steps=0)

        mock_trainer = MagicMock(spec=pl.Trainer)
        mock_module = MagicMock(spec=pl.LightningModule)

        cb.on_train_batch_start(mock_trainer, mock_module, None, 0)
        cb.on_train_batch_end(mock_trainer, mock_module, None, None, 0)

        assert cb._prev_step_end_wall is not None
        assert cb._current_gpu_start is not None
        assert cb._current_gpu_end is not None

    def test_repeated_flush_is_safe(self, mock_cuda_available) -> None:
        """Calling _flush_pending_events with an empty queue should not crash."""
        cb = DeviceBottleneckCallback(warmup_steps=0)
        cb._flush_pending_events()

    def test_clock_skew_warning(self, mock_cuda_available, caplog) -> None:
        """When gpu_ms > wall_ms, a warning is logged and ratio is clamped."""
        # The mock_cuda_available fixture patches torch.cuda.Event so that
        # torch.cuda.Event() returns a MagicMock with elapsed_time=42.0.
        # Override it to return 200 ms (greater than the wall gap).
        torch.cuda.Event.return_value.elapsed_time.return_value = 200.0

        cb = DeviceBottleneckCallback(warmup_steps=0, log_every_n_steps=0)
        mock_trainer = MagicMock(spec=pl.Trainer)
        mock_module = MagicMock(spec=pl.LightningModule)

        fake_clock = 0.0
        with patch("time.perf_counter", wraps=lambda: fake_clock):
            # Step 1: baseline (sets prev_step_end_wall)
            cb.on_train_batch_start(mock_trainer, mock_module, None, 0)
            fake_clock += 0.010
            cb.on_train_batch_end(mock_trainer, mock_module, None, None, 0)

            # Step 2: small wall gap (50 ms), large GPU time (200 ms)
            fake_clock += 0.050
            cb.on_train_batch_start(mock_trainer, mock_module, None, 1)
            fake_clock += 0.010
            cb.on_train_batch_end(mock_trainer, mock_module, None, None, 1)

        cb._flush_pending_events()

        assert len(cb.metrics) == 1
        assert cb.metrics[0].bottleneck_ratio == 1.0
        assert "clock skew" in caplog.text


# ======================================================================
# Epoch-boundary flush
# ======================================================================


class TestEpochBoundaryFlush:
    def test_epoch_end_flushes_pending_events(self, mock_cuda_available) -> None:
        """on_train_epoch_end flushes pending events even when log_every_n_steps=0.

        This prevents unbounded _pending_events growth across epochs.
        """
        cb = DeviceBottleneckCallback(warmup_steps=0, log_every_n_steps=0)
        mock_trainer = MagicMock(spec=pl.Trainer)
        mock_module = MagicMock(spec=pl.LightningModule)
        fake_clock = 0.0

        with patch("time.perf_counter", wraps=lambda: fake_clock):
            # Run one epoch (5 steps)
            for i in range(5):
                fake_clock += 0.050
                cb.on_train_batch_start(mock_trainer, mock_module, None, i)
                fake_clock += 0.010
                cb.on_train_batch_end(mock_trainer, mock_module, None, None, i)

        # Before epoch end: events queued, no metrics yet
        assert len(cb._pending_events) > 0
        assert len(cb.metrics) == 0

        # Epoch end: flush happens
        cb.on_train_epoch_end(mock_trainer, mock_module)

        # After epoch end: queue cleared, metrics populated
        assert len(cb._pending_events) == 0
        assert len(cb.metrics) > 0

    def test_epoch_end_does_not_crash_without_cuda(
        self, callback_default: DeviceBottleneckCallback
    ) -> None:
        """Calling on_train_epoch_end without CUDA is a no-op."""
        mock_trainer = MagicMock(spec=pl.Trainer)
        mock_module = MagicMock(spec=pl.LightningModule)
        callback_default.on_train_epoch_end(mock_trainer, mock_module)
        assert callback_default.metrics == []

    def test_epoch_end_does_not_crash_empty(self, mock_cuda_available) -> None:
        """Calling on_train_epoch_end with zero batches is safe."""
        cb = DeviceBottleneckCallback(warmup_steps=0, log_every_n_steps=0)
        mock_trainer = MagicMock(spec=pl.Trainer)
        mock_module = MagicMock(spec=pl.LightningModule)
        cb.on_train_epoch_end(mock_trainer, mock_module)
        assert cb.metrics == []


# ======================================================================
# Epoch warmup (warmup_epochs)
# ======================================================================


class TestEpochWarmup:
    """Tests for the ``warmup_epochs`` parameter."""

    def test_warmup_epochs_skips_entire_epoch(self, mock_cuda_available) -> None:
        """With warmup_epochs=1, all steps in epoch 0 produce no metrics."""
        cb = DeviceBottleneckCallback(warmup_epochs=1, warmup_steps=0, log_every_n_steps=0)
        mock_module = MagicMock(spec=pl.LightningModule)
        fake_clock = 0.0

        with patch("time.perf_counter", wraps=lambda: fake_clock):
            # Epoch 0 (warmup) — 3 steps, all skipped
            trainer = MagicMock(spec=pl.Trainer)
            trainer.current_epoch = 0
            cb.on_train_epoch_start(trainer, mock_module)
            for i in range(3):
                fake_clock += 0.050
                cb.on_train_batch_start(trainer, mock_module, None, i)
                fake_clock += 0.010
                cb.on_train_batch_end(trainer, mock_module, None, None, i)
            cb.on_train_epoch_end(trainer, mock_module)

        assert cb.metrics == []
        assert cb._pending_events == []

    def test_warmup_epochs_then_profiles(self, mock_cuda_available) -> None:
        """With warmup_epochs=1, epoch 0 is skipped, epoch 1 is profiled."""
        cb = DeviceBottleneckCallback(warmup_epochs=1, warmup_steps=0, log_every_n_steps=0)
        mock_module = MagicMock(spec=pl.LightningModule)
        fake_clock = 0.0

        with patch("time.perf_counter", wraps=lambda: fake_clock):
            # Epoch 0 (warmup) — skipped
            trainer_ep0 = MagicMock(spec=pl.Trainer)
            trainer_ep0.current_epoch = 0
            cb.on_train_epoch_start(trainer_ep0, mock_module)
            for i in range(2):
                fake_clock += 0.050
                cb.on_train_batch_start(trainer_ep0, mock_module, None, i)
                fake_clock += 0.010
                cb.on_train_batch_end(trainer_ep0, mock_module, None, None, i)
            cb.on_train_epoch_end(trainer_ep0, mock_module)

            # Epoch 1 (real) — profiled
            trainer_ep1 = MagicMock(spec=pl.Trainer)
            trainer_ep1.current_epoch = 1
            cb.on_train_epoch_start(trainer_ep1, mock_module)
            for i in range(3):
                fake_clock += 0.050
                cb.on_train_batch_start(trainer_ep1, mock_module, None, i)
                fake_clock += 0.010
                cb.on_train_batch_end(trainer_ep1, mock_module, None, None, i)
            cb.on_train_epoch_end(trainer_ep1, mock_module)

        cb._flush_pending_events()
        # Epoch 1 has 3 steps: step 1 skipped (no prev baseline), steps 2-3 → 2 metrics
        assert len(cb.metrics) == 2
        for m in cb.metrics:
            assert m.step_idx > 1

    def test_warmup_epochs_zero_is_noop(self, mock_cuda_available) -> None:
        """warmup_epochs=0 behaves identically to not setting it."""
        cb = DeviceBottleneckCallback(warmup_epochs=0, warmup_steps=0, log_every_n_steps=0)
        mock_module = MagicMock(spec=pl.LightningModule)
        fake_clock = 0.0

        with patch("time.perf_counter", wraps=lambda: fake_clock):
            trainer = MagicMock(spec=pl.Trainer)
            trainer.current_epoch = 0
            cb.on_train_epoch_start(trainer, mock_module)
            for i in range(4):
                fake_clock += 0.050
                cb.on_train_batch_start(trainer, mock_module, None, i)
                fake_clock += 0.010
                cb.on_train_batch_end(trainer, mock_module, None, None, i)
            cb.on_train_epoch_end(trainer, mock_module)

        cb._flush_pending_events()
        # 4 steps: step 1 skipped (no prev baseline), steps 2-4 → 3 metrics
        assert len(cb.metrics) == 3

    def test_negative_warmup_epochs_raises(self) -> None:
        with pytest.raises(ValueError, match="warmup_epochs"):
            DeviceBottleneckCallback(warmup_epochs=-1)

    def test_warmup_epochs_with_warmup_steps(self, mock_cuda_available) -> None:
        """Both warmup_epochs and warmup_steps compose correctly.

        Epoch 0 (warmup_epoch=1): entirely skipped.
        Epoch 1 (real): warmup_steps=2 skips first 2 measured steps.
        """
        cb = DeviceBottleneckCallback(warmup_epochs=1, warmup_steps=2, log_every_n_steps=0)
        mock_module = MagicMock(spec=pl.LightningModule)
        fake_clock = 0.0

        with patch("time.perf_counter", wraps=lambda: fake_clock):
            # Epoch 0 (warmup by epoch) — entirely skipped, _step_count stays 0
            trainer_ep0 = MagicMock(spec=pl.Trainer)
            trainer_ep0.current_epoch = 0
            cb.on_train_epoch_start(trainer_ep0, mock_module)
            for i in range(2):
                fake_clock += 0.050
                cb.on_train_batch_start(trainer_ep0, mock_module, None, i)
                fake_clock += 0.010
                cb.on_train_batch_end(trainer_ep0, mock_module, None, None, i)
            cb.on_train_epoch_end(trainer_ep0, mock_module)

            # Epoch 1 (real) with warmup_steps=2
            trainer_ep1 = MagicMock(spec=pl.Trainer)
            trainer_ep1.current_epoch = 1
            cb.on_train_epoch_start(trainer_ep1, mock_module)
            for i in range(5):
                fake_clock += 0.050
                cb.on_train_batch_start(trainer_ep1, mock_module, None, i)
                fake_clock += 0.010
                cb.on_train_batch_end(trainer_ep1, mock_module, None, None, i)
            cb.on_train_epoch_end(trainer_ep1, mock_module)

        cb._flush_pending_events()
        # Epoch 1: 5 steps.
        # Step 1 (i=0): step_count=1, no prev baseline → skipped.
        # Step 2 (i=1): step_count=2, warmup_steps=2 → 2 > 2? No → skipped.
        # Steps 3-5 (i=2,3,4): step_count=3,4,5 > 2 → recorded → 3 metrics.
        assert len(cb.metrics) == 3

    def test_warmup_epochs_no_cuda_noop(self) -> None:
        """With no CUDA, warmup_epochs guard does not crash."""
        cb = DeviceBottleneckCallback(warmup_epochs=2, warmup_steps=0)
        mock_module = MagicMock(spec=pl.LightningModule)

        trainer = MagicMock(spec=pl.Trainer)
        trainer.current_epoch = 0
        cb.on_train_epoch_start(trainer, mock_module)
        cb.on_train_batch_start(trainer, mock_module, None, 0)
        cb.on_train_batch_end(trainer, mock_module, None, None, 0)
        cb.on_train_epoch_end(trainer, mock_module)

        assert cb.metrics == []

    def test_warmup_epochs_warning_at_fit_start(self, mock_cuda_available, caplog) -> None:
        """on_fit_start warns when warmup_epochs >= max_epochs."""
        cb = DeviceBottleneckCallback(warmup_epochs=3, warmup_steps=0)
        trainer = MagicMock(spec=pl.Trainer)
        trainer.max_epochs = 3
        module = MagicMock(spec=pl.LightningModule)

        cb.on_fit_start(trainer, module)
        assert "warmup_epochs=3 >= max_epochs=3" in caplog.text

    def test_warmup_epochs_no_warning_when_smaller(self, mock_cuda_available, caplog) -> None:
        """No warning when warmup_epochs < max_epochs."""
        cb = DeviceBottleneckCallback(warmup_epochs=1, warmup_steps=0)
        trainer = MagicMock(spec=pl.Trainer)
        trainer.max_epochs = 5
        module = MagicMock(spec=pl.LightningModule)

        cb.on_fit_start(trainer, module)
        assert "warmup_epochs" not in caplog.text


# ======================================================================
# H2D transfer hooks exist on LightningModule
# ======================================================================


class TestH2DTransferHooks:
    """Verify that ``on_before_batch_transfer`` and ``on_after_batch_transfer``
    exist on ``LightningModule`` so users can override them to instrument H2D copies.

    These hooks are NOT available on the base ``Callback`` in Lightning 2.6.x,
    but they are first-class methods on ``LightningModule``.
    """

    def test_on_before_batch_transfer_exists(self) -> None:
        assert hasattr(pl.LightningModule, "on_before_batch_transfer")
        method = pl.LightningModule.on_before_batch_transfer
        import inspect

        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        assert "batch" in params
        assert "dataloader_idx" in params
        batch = object()
        module = MagicMock(spec=pl.LightningModule)
        result = method(module, batch, 0)
        assert result is batch

    def test_on_after_batch_transfer_exists(self) -> None:
        assert hasattr(pl.LightningModule, "on_after_batch_transfer")
        method = pl.LightningModule.on_after_batch_transfer
        import inspect

        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        assert "batch" in params
        assert "dataloader_idx" in params
        batch = object()
        module = MagicMock(spec=pl.LightningModule)
        result = method(module, batch, 0)
        assert result is batch
