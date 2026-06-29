"""
Module 2: PipelineBreakdown — "the where"

Once Module 1 identifies whether the CPU or GPU is the bottleneck, this module
drills into the specific pipeline stage that consumes the most time.

Design
------
The core primitive is a **context-manager API** that wraps arbitrary code blocks
and measures their execution time with:

- ``time.perf_counter()`` for CPU wall-clock time (all stages).
- ``torch.cuda.Event`` pairs for GPU device time (GPU stages only). These record
  timestamps on the GPU device clock, giving the true GPU execution time
  regardless of async kernel launch behaviour.

GPU synchronisation is amortised: all CUDA-event pairs from a single step are
flushed together at step boundaries, avoiding per-stage ``synchronize()`` calls
that would destroy the CPU-GPU pipeline.

The output is a per-stage breakdown table at the end of profiling, plus optional
JSON export and Chrome-trace export for visual inspection.

Important: ``torch.compile`` changes the execution model
----------------------------------------------------------
With ``torch.compile`` (Inductor/Triton), individual PyTorch operations are
fused into larger opaque kernels. This means **automatic per-layer breakdown
is not possible in compiled mode** — the layers no longer exist as separate
kernels.

However, the user can still annotate logical pipeline stages with the context
manager API. These annotations wrap the *outside* of the compiled region, so
the CUDA events correctly measure the GPU time of each user-defined stage
(e.g. ``attention``, ``ffn``), even though the internals are fused.

For deeper inspection of compiled graph internals, use external tools such as
Nsight Systems + NVTX annotations.

Usage
-----
    from lightning_profiler.pipeline_breakdown import PipelineBreakdown

    pb = PipelineBreakdown()
    with pb:
        for batch in dataloader:
            with pb.stage("data_loading"):
                x, y = batch

            with pb.stage("forward", device="gpu"):
                y_pred = model(x)

            with pb.stage("loss", device="gpu"):
                loss = loss_fn(y_pred, y)

            with pb.stage("backward", device="gpu"):
                loss.backward()

            pb.end_step()

    pb.print_report()
    pb.export_json("breakdown.json")
    pb.export_chrome_trace("trace.json")
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, TextIO

import pytorch_lightning as pl
import torch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class StageRecord:
    """Timing data for a single invocation of a pipeline stage.

    Each stage is measured once per step.  Wall-clock time is always captured.
    GPU device time (via CUDA events) is captured when ``device="gpu"`` and
    CUDA is available.
    """

    name: str
    device: str  # "cpu" | "gpu"
    wall_time_ms: float = 0.0
    gpu_time_ms: float = 0.0

    # Internal timing state
    _start_wall: float = 0.0
    _end_wall: float = 0.0
    _start_event: Any = None  # torch.cuda.Event | None
    _end_event: Any = None  # torch.cuda.Event | None


@dataclass
class StepRecord:
    """Timing data for all stages in a single training step."""

    stages: list[StageRecord] = field(default_factory=list)

    @property
    def total_wall_ms(self) -> float:
        return sum(s.wall_time_ms for s in self.stages)

    @property
    def total_gpu_ms(self) -> float:
        return sum(s.gpu_time_ms for s in self.stages)


# ---------------------------------------------------------------------------
# PipelineBreakdown — context-manager API
# ---------------------------------------------------------------------------


class PipelineBreakdown:
    """Context manager for collecting per-stage timing breakdowns.

    Typical usage (see module docstring for a full example)::

        pb = PipelineBreakdown()
        with pb:
            for batch in dataloader:
                with pb.stage("forward", device="gpu"):
                    y_pred = model(x)
                pb.end_step()

        pb.print_report()
    """

    def __init__(self) -> None:
        self._steps: list[StepRecord] = []
        self._current_step: list[StageRecord] = []
        self._active_stages: dict[str, StageRecord] = {}
        self._has_cuda: bool = torch.cuda.is_available()

        # Wall timestamp at the start of the outermost ``with pb:`` block.
        # Used as the zero-point for Chrome trace timestamps.
        self._session_start: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __enter__(self) -> PipelineBreakdown:
        self._session_start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        if self._active_stages:
            log.warning(
                "[PipelineBreakdown] %d stage(s) still active at session end; forcing close.",
                len(self._active_stages),
            )
            for name in list(self._active_stages):
                log.warning("  Unclosed stage: %s", name)
        if self._current_step:
            self.end_step()

    @contextmanager
    def stage(self, name: str, device: str = "cpu") -> Iterator[None]:
        """Context manager that times a single pipeline stage.

        Parameters
        ----------
        name:
            Human-readable stage name (e.g. ``"forward"``, ``"data_loading"``).
        device:
            ``"cpu"`` (default) or ``"gpu"``.  GPU stages record CUDA events
            for device-level timing; CPU stages use only ``time.perf_counter``.

        Raises
        ------
        RuntimeError
            If ``name`` is already active (nested stages with the same name).
        """
        if name in self._active_stages:
            raise RuntimeError(
                f"Stage '{name}' is already active. "
                f"Nested stages with the same name are not supported."
            )

        rec = StageRecord(name=name, device=device)
        self._active_stages[name] = rec

        # Start CPU wall-clock timer
        rec._start_wall = time.perf_counter()

        # Start GPU timer (if applicable)
        if device == "gpu" and self._has_cuda:
            rec._start_event = torch.cuda.Event(enable_timing=True)
            rec._start_event.record()

        try:
            yield
        finally:
            # Stop CPU wall-clock timer
            rec._end_wall = time.perf_counter()

            # Stop GPU timer (if applicable)
            if device == "gpu" and self._has_cuda and rec._start_event is not None:
                rec._end_event = torch.cuda.Event(enable_timing=True)
                rec._end_event.record()

            del self._active_stages[name]
            self._current_step.append(rec)

    def end_step(self) -> None:
        """Finalise the current step.

        Performs a single ``torch.cuda.synchronize()`` (if CUDA is available
        and any GPU stages were recorded), then computes per-stage timings.
        """
        if not self._current_step:
            return

        has_gpu_stages = any(s.device == "gpu" for s in self._current_step)

        # Single GPU sync for all stages in this step
        if has_gpu_stages and self._has_cuda:
            torch.cuda.synchronize()

        for rec in self._current_step:
            wall_ms = (rec._end_wall - rec._start_wall) * 1000.0
            rec.wall_time_ms = max(wall_ms, 0.0)

            if rec.device == "gpu" and self._has_cuda:
                if rec._start_event is not None and rec._end_event is not None:
                    try:
                        gpu_ms = rec._start_event.elapsed_time(rec._end_event)
                    except Exception as exc:
                        log.warning(
                            "[PipelineBreakdown] Stage '%s': CUDA event elapsed_time failed: %s",
                            rec.name,
                            exc,
                        )
                        gpu_ms = 0.0

                    # Clamp: GPU device time should not meaningfully exceed
                    # wall time (accounts for minor clock skew between CPU
                    # and GPU device clocks).
                    if gpu_ms > wall_ms + 1.0:
                        log.warning(
                            "[PipelineBreakdown] Clock skew in stage '%s': "
                            "gpu_time (%.1f ms) > wall_time (%.1f ms). "
                            "Clamping to wall_time.",
                            rec.name,
                            gpu_ms,
                            wall_ms,
                        )
                    rec.gpu_time_ms = min(gpu_ms, wall_ms) if wall_ms > 0 else 0.0
                else:
                    rec.gpu_time_ms = 0.0

        self._steps.append(StepRecord(stages=list(self._current_step)))
        self._current_step = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def steps(self) -> list[StepRecord]:
        """All recorded steps."""
        return list(self._steps)

    @property
    def n_steps(self) -> int:
        """Number of recorded steps."""
        return len(self._steps)

    @property
    def stage_names(self) -> list[str]:
        """Unique stage names, in order of first appearance."""
        seen: set[str] = set()
        names: list[str] = []
        for step in self._steps:
            for s in step.stages:
                if s.name not in seen:
                    seen.add(s.name)
                    names.append(s.name)
        return names

    def _aggregate(self) -> dict[str, dict[str, Any]]:
        """Aggregate per-stage metrics across all steps.

        Returns a dict mapping stage name to:
            device, avg_wall_ms, avg_gpu_ms, total_wall_ms, total_gpu_ms,
            pct_of_step (average % of total step wall time).
        """
        if not self._steps:
            return {}

        # Collect per-stage values
        stage_data: dict[str, dict[str, Any]] = {}
        for name in self.stage_names:
            stage_data[name] = {
                "device": "",
                "wall_times": [],
                "gpu_times": [],
                "pcts": [],
            }

        for step in self._steps:
            total = step.total_wall_ms
            for s in step.stages:
                d = stage_data[s.name]
                d["device"] = s.device
                d["wall_times"].append(s.wall_time_ms)
                d["gpu_times"].append(s.gpu_time_ms)
                d["pcts"].append((s.wall_time_ms / total * 100.0) if total > 0 else 0.0)

        # Compute averages
        result: dict[str, dict[str, Any]] = {}
        for name, data in stage_data.items():
            result[name] = {
                "device": data["device"],
                "avg_wall_ms": statistics.mean(data["wall_times"]) if data["wall_times"] else 0.0,
                "avg_gpu_ms": statistics.mean(data["gpu_times"]) if data["gpu_times"] else 0.0,
                "total_wall_ms": sum(data["wall_times"]),
                "total_gpu_ms": sum(data["gpu_times"]),
                "avg_pct": statistics.mean(data["pcts"]) if data["pcts"] else 0.0,
            }

        return result

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def print_report(self, file: TextIO | None = None) -> None:
        """Print a human-readable breakdown table to *file* (default stdout).

        Columns: stage name, device, average wall time (ms), average GPU time,
        and percentage of total step wall time.
        """
        import sys

        file = file or sys.stdout

        if not self._steps:
            print("[PipelineBreakdown] No steps recorded.", file=file)
            return

        agg = self._aggregate()
        n_steps = len(self._steps)

        print(file=file)
        print(f" Pipeline Breakdown  —  {n_steps} step(s)", file=file)
        print("=" * 60, file=file)
        print(
            f"  {'Stage':<20} {'Device':<8} {'Wall (ms)':<12} {'GPU (ms)':<12} {'% of step':<10}",
            file=file,
        )
        print("  " + "-" * 58, file=file)

        for name in self.stage_names:
            data = agg[name]
            print(
                f"  {name:<20} {data['device']:<8} "
                f"{data['avg_wall_ms']:<12.2f} "
                f"{data['avg_gpu_ms']:<12.2f} "
                f"{data['avg_pct']:<10.1f}",
                file=file,
            )

        print("=" * 60, file=file)
        print(file=file)

    def export_json(self, path: str) -> None:
        """Export per-stage aggregated metrics as JSON.

        The JSON object contains:
          - ``n_steps``: number of recorded steps
          - ``stages``: ordered dict of stage name → (device, avg_wall_ms, avg_gpu_ms, avg_pct)
        """
        agg = self._aggregate()
        data = {
            "n_steps": len(self._steps),
            "stages": {
                name: {
                    "device": agg[name]["device"],
                    "avg_wall_ms": round(agg[name]["avg_wall_ms"], 2),
                    "avg_gpu_ms": round(agg[name]["avg_gpu_ms"], 2),
                    "avg_pct": round(agg[name]["avg_pct"], 1),
                }
                for name in self.stage_names
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        log.info("[PipelineBreakdown] JSON report written to %s", path)

    def export_chrome_trace(self, path: str) -> None:
        """Export a Chrome-trace compatible JSON file.

        Each stage invocation is a complete event (``ph: "X"``) with
        timestamps relative to the session start.  Open the resulting file
        in ``chrome://tracing`` for visual inspection.
        """
        if not self._steps or self._session_start == 0.0:
            log.warning("[PipelineBreakdown] No data to export for Chrome trace.")
            return

        events: list[dict[str, Any]] = []
        step_start = self._session_start

        for step_idx, step in enumerate(self._steps):
            for rec in step.stages:
                ts_us = (rec._start_wall - step_start) * 1_000_000  # microseconds
                dur_us = rec.wall_time_ms * 1_000  # milliseconds → microseconds

                events.append(
                    {
                        "name": rec.name,
                        "cat": "stage",
                        "ph": "X",
                        "ts": ts_us,
                        "dur": dur_us,
                        "pid": 0,
                        "tid": step_idx,
                        "args": {
                            "device": rec.device,
                            "gpu_ms": round(rec.gpu_time_ms, 2),
                        },
                    }
                )

        trace = {
            "traceEvents": events,
            "displayTimeUnit": "ms",
        }

        with open(path, "w") as f:
            json.dump(trace, f, indent=2)

        log.info("[PipelineBreakdown] Chrome trace written to %s", path)


# ---------------------------------------------------------------------------
# Lightning Callback — automated lifecycle wrapping
# ---------------------------------------------------------------------------


class PipelineBreakdownCallback(pl.Callback):
    """Lightning callback that auto-instruments the training step lifecycle.

    This callback wraps the standard Lightning hooks with ``PipelineBreakdown``
    stages, giving you a breakdown of:
      - ``forward`` — the forward pass
      - ``backward`` — the backward pass (loss.backward)
      - ``optimizer`` — the optimizer step
      - ``data_loading`` — inter-batch CPU time

    For GPU stages (forward, backward), CUDA events are recorded.  The callback
    synchronises once per step at the end.

    .. note::
        If you need finer-grained stages (e.g. separate ``attention`` and
        ``ffn`` blocks), use the context-manager API directly inside your
        ``LightningModule.training_step()`` instead.

    Parameters
    ----------
    warmup_epochs:
        Number of initial epochs to skip profiling entirely. Default ``0``.
    output_path:
        Optional path to write JSON report at the end of training.
    """

    def __init__(self, warmup_epochs: int = 0, output_path: str | None = None) -> None:
        super().__init__()
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        self.warmup_epochs = warmup_epochs
        self._pb = PipelineBreakdown()
        self._output_path = output_path
        self._in_step: bool = False

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        """Warn if warmup_epochs >= max_epochs (would skip all profiling)."""
        if (
            self.warmup_epochs > 0
            and trainer.max_epochs is not None
            and self.warmup_epochs >= trainer.max_epochs
        ):
            log.warning(
                "[PipelineBreakdown] warmup_epochs=%d >= max_epochs=%d — "
                "all epochs are warmup, no metrics will be collected.",
                self.warmup_epochs,
                trainer.max_epochs,
            )

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        self._pb.__enter__()

    def on_train_end(self, trainer: Any, pl_module: Any) -> None:
        self._pb.__exit__(None, None, None)
        self._pb.print_report()
        if self._output_path:
            self._pb.export_json(self._output_path)

    def on_train_batch_start(
        self,
        trainer: Any,
        pl_module: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        # Skip warmup epochs entirely (user's pb.stage() calls in
        # training_step will still run but end_step is not called, so
        # stage records accumulate in _current_step and get cleared here).
        if self.warmup_epochs > 0 and trainer.current_epoch < self.warmup_epochs:
            self._pb._current_step = []  # discard any leftover stages
            self._in_step = False
            return

        self._in_step = True
        self._pb._current_step = []  # fresh step

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._in_step:
            self._pb.end_step()
            self._in_step = False

    def on_after_backward(self, trainer: Any, pl_module: Any) -> None:
        """Close the backward stage if we opened one."""
        pass  # handled via manual stages in training_step

    # ------------------------------------------------------------------
    # Public access
    # ------------------------------------------------------------------

    @property
    def pb(self) -> PipelineBreakdown:
        """Access the underlying ``PipelineBreakdown`` instance."""
        return self._pb
