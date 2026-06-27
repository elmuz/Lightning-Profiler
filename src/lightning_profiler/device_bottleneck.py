"""
Module 1: DeviceBottleneck — "the what"

Answers the most critical profiling question: **which device is the bottleneck?**

For each training step, we measure:
  - ``wall_time_ms``: total *inter-batch* time (end of previous batch → end of current batch).
    This captures the full cycle including CPU-side data loading and preprocessing
    that Lightning's DataLoader prefetches *before* ``on_train_batch_start`` fires.
  - ``gpu_time_ms``:  actual time spent executing GPU kernels (via CUDA events).
  - ``bottleneck_ratio`` = gpu_time_ms / wall_time_ms

Interpretation of bottleneck_ratio:
  - **< threshold** (default 0.85) → CPU is the bottleneck. The GPU finishes its work
    early and spends a significant fraction of the step idle, waiting for data from CPU.
  - **~1.0** → GPU is the bottleneck. The GPU is fully occupied throughout the step.
  - **in between** → both devices are well utilised (balanced).

Why CUDA events?
  GPU kernels are launched asynchronously from the CPU. Measuring GPU execution
  time from the CPU side (with Python timers) is misleading — the CPU only sees
  when it *submitted* the kernel, not when it *executed*.
  ``torch.cuda.Event`` records timestamps on the GPU device clock, giving us
  the true GPU execution time regardless of CPU-GPU async behaviour.

Sync amortisation
  ``torch.cuda.synchronize()`` is *not* called every step. Doing so would destroy
  the CPU-GPU pipeline bubble by forcing the CPU to wait for the GPU before
  prefetching the next batch. Instead, CUDA-event pairs are queued and only
  synchronised once every ``log_every_n_steps`` steps, matching the logging
  interval. This gives accurate timing with negligible profiling overhead.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import statistics
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class StepMetrics:
    """Per-step profiling data."""

    step_idx: int
    wall_time_ms: float
    gpu_time_ms: float
    bottleneck_ratio: float
    verdict: str  # "cpu_bottleneck" | "gpu_bottleneck" | "balanced"


# ---------------------------------------------------------------------------
# Verdict thresholds
# ---------------------------------------------------------------------------


def _compute_verdict(
    ratio: float,
    cpu_threshold: float = 0.85,
    gpu_threshold: float = 0.98,
) -> str:
    if ratio < cpu_threshold:
        return "cpu_bottleneck"
    if ratio > gpu_threshold:
        return "gpu_bottleneck"
    return "balanced"


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


class DeviceBottleneckCallback(Callback):
    """PyTorch Lightning callback that identifies the device-level bottleneck.

    Measures GPU utilisation per training step and classifies whether the
    CPU or GPU is the primary bottleneck.

    .. important::

        ``wall_time_ms`` is measured from the **end of the previous batch**
        to the **end of the current batch**. This captures the full inter-batch
        cycle including CPU data loading, which runs *before* Lightning calls
        ``on_train_batch_start``.

        ``torch.cuda.synchronize()`` is **not** called every step — it is
        amortised over ``log_every_n_steps`` steps to avoid destroying the
        CPU-GPU asynchronous pipeline.

    Parameters
    ----------
    cpu_threshold:
        ``bottleneck_ratio`` below this value flags a CPU bottleneck
        (GPU is idle waiting for CPU). Default ``0.85``.
    gpu_threshold:
        ``bottleneck_ratio`` above this value flags a GPU bottleneck
        (GPU is fully occupied). Default ``0.98``.
    log_every_n_steps:
        Log a summary and flush pending CUDA events every N steps.
        ``0`` disables periodic logging. Default ``50``.
    output_path:
        Optional file path to write a JSON array of per-step metrics
        at the end of training. Default ``None``.
    warmup_steps:
        Number of initial steps to skip (CUDA events are unreliable during
        GPU warmup). Default ``5``.
    """

    def __init__(
        self,
        cpu_threshold: float = 0.85,
        gpu_threshold: float = 0.98,
        log_every_n_steps: int = 50,
        output_path: str | Path | None = None,
        warmup_steps: int = 5,
    ) -> None:
        super().__init__()

        if not 0.0 <= cpu_threshold <= 1.0:
            raise ValueError(f"cpu_threshold must be in [0, 1], got {cpu_threshold}")
        if not 0.0 <= gpu_threshold <= 1.0:
            raise ValueError(f"gpu_threshold must be in [0, 1], got {gpu_threshold}")
        if cpu_threshold > gpu_threshold:
            raise ValueError(
                f"cpu_threshold ({cpu_threshold}) must be <= gpu_threshold ({gpu_threshold})"
            )
        if log_every_n_steps < 0:
            raise ValueError(f"log_every_n_steps must be >= 0, got {log_every_n_steps}")

        self.cpu_threshold = cpu_threshold
        self.gpu_threshold = gpu_threshold
        self.log_every_n_steps = log_every_n_steps
        self.output_path = Path(output_path) if output_path else None
        self.warmup_steps = warmup_steps

        self._has_cuda: bool = torch.cuda.is_available()

        # ---- Accumulated metrics ----
        self._metrics: list[StepMetrics] = []

        # ---- Per-step state ----
        self._step_count: int = 0
        # Wall-clock timestamp at the end of the *previous* batch.
        # The inter-batch gap is: now - self._prev_step_end_wall
        self._prev_step_end_wall: float | None = None

        # GPU events for the *current* step (set in on_train_batch_start,
        # recorded in on_train_batch_end, then moved to the pending queue).
        self._current_gpu_start: torch.cuda.Event | None = None
        self._current_gpu_end: torch.cuda.Event | None = None

        # ---- Deferred synchronisation queue ----
        # Each entry: (step_idx, wall_ms, start_event, end_event)
        self._pending_events: list[tuple[int, float, torch.cuda.Event, torch.cuda.Event]] = []

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_train_epoch_start(
        self,
        trainer: Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Reset the inter-batch baseline at the start of each epoch.

        Without this, the ``_prev_step_end_wall`` from the last step of the
        previous epoch would capture the entire validation + checkpointing
        phase, producing a wildly inflated ``wall_time_ms`` for the first
        step of the new epoch.
        """
        self._prev_step_end_wall = None

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Flush pending events at epoch boundaries.

        This serves two purposes:
        1. **Memory**: Prevents unbounded growth of ``_pending_events`` when
           ``log_every_n_steps=0`` (metrics-only mode). Without this, the
           queue grows linearly with training steps until ``on_train_end``.
        2. **Isolation**: Pushes metrics out before validation runs, so the
           wall-clock reset in ``on_train_epoch_start`` gives a clean
           measurement window for the next epoch.
        """
        self._flush_pending_events()

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: pl.LightningModule,
        batch: object,
        batch_idx: int,
    ) -> None:
        if not self._has_cuda:
            return

        self._current_gpu_start = torch.cuda.Event(enable_timing=True)
        self._current_gpu_end = torch.cuda.Event(enable_timing=True)
        self._current_gpu_start.record()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: object,
        batch_idx: int,
    ) -> None:
        if not self._has_cuda or self._current_gpu_start is None or self._current_gpu_end is None:
            return

        now_wall = time.perf_counter()
        self._step_count += 1

        # Record the GPU completion event (non-blocking).
        self._current_gpu_end.record()

        # The inter-batch wall time measures the gap between the end of the
        # previous step and the end of this step — this *includes* the CPU
        # data loading that happened in between.
        if self._prev_step_end_wall is not None and self._step_count > self.warmup_steps:
            wall_ms = (now_wall - self._prev_step_end_wall) * 1000.0

            self._pending_events.append(
                (
                    self._step_count,
                    wall_ms,
                    self._current_gpu_start,
                    self._current_gpu_end,
                )
            )

        # Update the baseline marker for the *next* step.
        #
        # We must first check whether a flush (with its blocking
        # ``torch.cuda.synchronize()``) is about to happen.  If so, we
        # re-sample the wall clock *after* the sync so that the sync
        # overhead is attributed to the flush itself, not to the next
        # step's measurement window.
        will_flush = self.log_every_n_steps > 0 and self._step_count % self.log_every_n_steps == 0

        if will_flush:
            self._flush_pending_events()
            # Re-sample after sync -- the next step's wall time starts cleanly
            # here, free of any profiler-induced sync delay.
            self._prev_step_end_wall = time.perf_counter()
        else:
            # No flush: use the original clean end-of-step timestamp.
            self._prev_step_end_wall = now_wall

    def on_train_end(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        # Flush any remaining straggler events.
        self._flush_pending_events()
        self._write_output()
        self._log_summary()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def metrics(self) -> list[StepMetrics]:
        """Collected step metrics (excluding warmup steps)."""
        return list(self._metrics)

    @property
    def summary(self) -> dict:
        """Aggregated summary across all recorded steps."""
        if not self._metrics:
            return {
                "total_steps": 0,
                "cpu_bottleneck_pct": 0.0,
                "gpu_bottleneck_pct": 0.0,
                "balanced_pct": 100.0,
                "avg_gpu_time_ms": 0.0,
                "avg_wall_time_ms": 0.0,
                "avg_bottleneck_ratio": 0.0,
            }

        ratios = [m.bottleneck_ratio for m in self._metrics]
        verdicts = [m.verdict for m in self._metrics]
        n = len(verdicts)

        return {
            "total_steps": n,
            "cpu_bottleneck_pct": round(verdicts.count("cpu_bottleneck") / n * 100, 1),
            "gpu_bottleneck_pct": round(verdicts.count("gpu_bottleneck") / n * 100, 1),
            "balanced_pct": round(verdicts.count("balanced") / n * 100, 1),
            "avg_gpu_time_ms": round(statistics.mean(m.gpu_time_ms for m in self._metrics), 2),
            "avg_wall_time_ms": round(statistics.mean(m.wall_time_ms for m in self._metrics), 2),
            "avg_bottleneck_ratio": round(statistics.mean(ratios), 4),
            "min_bottleneck_ratio": round(min(ratios), 4),
            "max_bottleneck_ratio": round(max(ratios), 4),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush_pending_events(self) -> None:
        """Synchronise the GPU once and process all queued event pairs.

        This is the core of the sync-amortisation strategy: one call to
        ``torch.cuda.synchronize()`` services many steps at once.
        """
        if not self._pending_events:
            return

        # Single synchronisation for the entire batch of pending events.
        torch.cuda.synchronize()

        for step_idx, wall_ms, start_evt, end_evt in self._pending_events:
            gpu_ms = start_evt.elapsed_time(end_evt)
            ratio = gpu_ms / wall_ms if wall_ms > 0 else 1.0

            # Clamp ratio to [0, 1] to account for minor clock-skew between
            # the CPU and GPU device clocks that can push the ratio slightly
            # above 1.0.
            if ratio > 1.0:
                log.warning(
                    "[DeviceBottleneck] Step %d: gpu_time (%.2f ms) > wall_time (%.2f ms). "
                    "Clamping ratio to 1.0 (possible clock skew).",
                    step_idx,
                    gpu_ms,
                    wall_ms,
                )
                ratio = 1.0

            metrics = StepMetrics(
                step_idx=step_idx,
                wall_time_ms=round(wall_ms, 2),
                gpu_time_ms=round(gpu_ms, 2),
                bottleneck_ratio=round(ratio, 4),
                verdict=_compute_verdict(ratio, self.cpu_threshold, self.gpu_threshold),
            )
            self._metrics.append(metrics)

        self._pending_events.clear()
        self._log_recent_metrics()

    def _log_recent_metrics(self) -> None:
        """Log a summary of the most recently flushed block of metrics."""
        if not self._metrics:
            return

        # The last flush is everything from end of previous log to now.
        # Use the metrics added since the last clear.
        recent = self._metrics  # We'll handle partial below
        # For simplicity, take the last log_every_n_steps (or fewer if at end).
        n = min(len(recent), self.log_every_n_steps) if self.log_every_n_steps > 0 else len(recent)
        recent = self._metrics[-n:]

        avg_ratio = statistics.mean(m.bottleneck_ratio for m in recent)
        cpu_pct = sum(1 for m in recent if m.verdict == "cpu_bottleneck") / len(recent) * 100
        gpu_pct = sum(1 for m in recent if m.verdict == "gpu_bottleneck") / len(recent) * 100

        log.info(
            "[DeviceBottleneck] Step %d | avg GPU util: %.1f%% "
            "(wall=%.1fms, gpu=%.1fms) | "
            "CPU-bn: %.0f%% GPU-bn: %.0f%% | %d steps",
            self._step_count,
            avg_ratio * 100,
            recent[-1].wall_time_ms,
            recent[-1].gpu_time_ms,
            cpu_pct,
            gpu_pct,
            len(recent),
        )

    def _log_summary(self) -> None:
        s = self.summary
        if s["total_steps"] == 0:
            log.info("[DeviceBottleneck] No metrics collected (no GPU or no training steps).")
            return

        log.info(
            "[DeviceBottleneck] === Summary (%d steps) ===\n"
            "  CPU bottleneck:  %.1f%% of steps\n"
            "  GPU bottleneck:  %.1f%% of steps\n"
            "  Balanced:        %.1f%% of steps\n"
            "  Avg wall:        %.1f ms\n"
            "  Avg GPU:         %.1f ms\n"
            "  Avg bottleneck ratio: %.4f\n"
            "  Ratio range:     [%.4f, %.4f]",
            s["total_steps"],
            s["cpu_bottleneck_pct"],
            s["gpu_bottleneck_pct"],
            s["balanced_pct"],
            s["avg_wall_time_ms"],
            s["avg_gpu_time_ms"],
            s["avg_bottleneck_ratio"],
            s["min_bottleneck_ratio"],
            s["max_bottleneck_ratio"],
        )

    def _write_output(self) -> None:
        if self.output_path is None or not self._metrics:
            return

        # In DDP or multi-server setups, only rank 0 should write the file.
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() != 0
        ):
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "step": m.step_idx,
                "wall_time_ms": m.wall_time_ms,
                "gpu_time_ms": m.gpu_time_ms,
                "bottleneck_ratio": m.bottleneck_ratio,
                "verdict": m.verdict,
            }
            for m in self._metrics
        ]
        self.output_path.write_text(json.dumps(data, indent=2))
        log.info("[DeviceBottleneck] Metrics written to %s", self.output_path)
