"""Tests for ``PipelineBreakdown`` and ``PipelineBreakdownCallback``."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest
import pytorch_lightning as pl
import torch

from lightning_profiler.pipeline_breakdown import (
    PipelineBreakdown,
    PipelineBreakdownCallback,
    StageRecord,
    StepRecord,
)

# ======================================================================
# StageRecord / StepRecord data model
# ======================================================================


class TestStageRecord:
    def test_defaults(self) -> None:
        rec = StageRecord(name="forward", device="gpu")
        assert rec.name == "forward"
        assert rec.device == "gpu"
        assert rec.wall_time_ms == 0.0
        assert rec.gpu_time_ms == 0.0


class TestStepRecord:
    def test_empty_step(self) -> None:
        step = StepRecord()
        assert step.stages == []
        assert step.total_wall_ms == 0.0
        assert step.total_gpu_ms == 0.0

    def test_sums(self) -> None:
        s1 = StageRecord(name="a", device="cpu", wall_time_ms=100.0, gpu_time_ms=0.0)
        s2 = StageRecord(name="b", device="gpu", wall_time_ms=200.0, gpu_time_ms=150.0)
        step = StepRecord(stages=[s1, s2])
        assert step.total_wall_ms == 300.0
        assert step.total_gpu_ms == 150.0


# ======================================================================
# PipelineBreakdown — basic usage (no CUDA)
# ======================================================================


class TestPipelineBreakdownNoCuda:
    def test_empty_profile(self) -> None:
        """No steps recorded should not crash."""
        pb = PipelineBreakdown()
        with pb:
            pass
        assert pb.n_steps == 0
        assert pb.stage_names == []

    def test_single_cpu_stage(self) -> None:
        pb = PipelineBreakdown()
        with pb:
            with pb.stage("data_loading"):
                time.sleep(0.001)
            pb.end_step()

        assert pb.n_steps == 1
        assert pb.stage_names == ["data_loading"]
        rec = pb.steps[0].stages[0]
        assert rec.name == "data_loading"
        assert rec.device == "cpu"
        assert rec.wall_time_ms > 0.0
        assert rec.gpu_time_ms == 0.0

    def test_multiple_cpu_stages(self) -> None:
        pb = PipelineBreakdown()
        with pb:
            for _ in range(3):
                with pb.stage("load"):
                    pass
                with pb.stage("process"):
                    pass
                pb.end_step()

        assert pb.n_steps == 3
        assert pb.stage_names == ["load", "process"]

    def test_nested_stage_same_name_raises(self) -> None:
        pb = PipelineBreakdown()
        with pytest.raises(RuntimeError, match="already active"):
            with pb.stage("dup"):
                with pb.stage("dup"):
                    pass

    def test_end_step_empty_is_safe(self) -> None:
        pb = PipelineBreakdown()
        pb.end_step()
        assert pb.n_steps == 0

    def test_aggregate_empty(self) -> None:
        pb = PipelineBreakdown()
        assert pb._aggregate() == {}

    def test_print_report_empty(self, capsys) -> None:
        pb = PipelineBreakdown()
        pb.print_report()
        captured = capsys.readouterr()
        assert "No steps recorded" in captured.out

    def test_export_json_empty(self, tmp_path) -> None:
        pb = PipelineBreakdown()
        path = tmp_path / "empty.json"
        pb.export_json(str(path))
        data = json.loads(path.read_text())
        assert data["n_steps"] == 0

    def test_export_chrome_trace_empty_does_not_crash(self, tmp_path) -> None:
        pb = PipelineBreakdown()
        path = tmp_path / "empty_trace.json"
        pb.export_chrome_trace(str(path))


# ======================================================================
# PipelineBreakdown — with mocked CUDA
# ======================================================================


@pytest.fixture
def mock_cuda() -> None:
    """Mock CUDA for PipelineBreakdown tests."""
    mock_event = MagicMock(spec=torch.cuda.Event)
    mock_event.elapsed_time.return_value = 40.0

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.Event", return_value=mock_event),
        patch("torch.cuda.synchronize"),
    ):
        yield


class TestPipelineBreakdownWithCuda:
    def test_gpu_stage(self, mock_cuda) -> None:
        pb = PipelineBreakdown()
        with pb:
            with pb.stage("forward", device="gpu"):
                time.sleep(0.06)
            pb.end_step()

        assert pb.n_steps == 1
        rec = pb.steps[0].stages[0]
        assert rec.device == "gpu"
        assert rec.gpu_time_ms == 40.0

    def test_mixed_stages(self, mock_cuda) -> None:
        pb = PipelineBreakdown()
        with pb:
            with pb.stage("data_loading"):
                time.sleep(0.03)
            with pb.stage("forward", device="gpu"):
                time.sleep(0.06)
            with pb.stage("backward", device="gpu"):
                time.sleep(0.05)
            pb.end_step()

        assert pb.n_steps == 1
        stages = pb.steps[0].stages
        assert stages[0].device == "cpu"
        assert stages[0].gpu_time_ms == 0.0
        assert stages[1].device == "gpu"
        assert stages[1].gpu_time_ms == 40.0
        assert stages[2].device == "gpu"
        assert stages[2].gpu_time_ms == 40.0

    def test_multiple_steps_aggregation(self, mock_cuda) -> None:
        pb = PipelineBreakdown()
        with pb:
            for _ in range(5):
                with pb.stage("cpu_stage"):
                    time.sleep(0.01)
                with pb.stage("gpu_stage", device="gpu"):
                    time.sleep(0.06)
                pb.end_step()

        assert pb.n_steps == 5
        agg = pb._aggregate()
        assert "cpu_stage" in agg
        assert "gpu_stage" in agg
        assert agg["gpu_stage"]["avg_gpu_ms"] == 40.0

    def test_stage_names_preserves_order(self, mock_cuda) -> None:
        pb = PipelineBreakdown()
        with pb:
            with pb.stage("a"):
                pass
            with pb.stage("b"):
                pass
            pb.end_step()
            with pb.stage("a"):
                pass
            with pb.stage("c"):
                pass
            pb.end_step()

        assert pb.stage_names == ["a", "b", "c"]

    def test_export_json_with_cuda(self, mock_cuda, tmp_path) -> None:
        pb = PipelineBreakdown()
        with pb:
            with pb.stage("forward", device="gpu"):
                time.sleep(0.06)
            pb.end_step()

        path = tmp_path / "report.json"
        pb.export_json(str(path))

        data = json.loads(path.read_text())
        assert data["n_steps"] == 1
        assert data["stages"]["forward"]["device"] == "gpu"
        assert data["stages"]["forward"]["avg_gpu_ms"] > 0

    def test_export_chrome_trace(self, mock_cuda, tmp_path) -> None:
        pb = PipelineBreakdown()
        with pb:
            with pb.stage("cpu_stage"):
                time.sleep(0.01)
            with pb.stage("gpu_stage", device="gpu"):
                time.sleep(0.06)
            pb.end_step()

        path = tmp_path / "trace.json"
        pb.export_chrome_trace(str(path))

        data = json.loads(path.read_text())
        assert "traceEvents" in data
        events = data["traceEvents"]
        assert len(events) == 2
        assert events[0]["name"] == "cpu_stage"
        assert events[1]["name"] == "gpu_stage"
        assert events[0]["ph"] == "X"

    def test_print_report_with_cuda(self, mock_cuda, capsys) -> None:
        pb = PipelineBreakdown()
        with pb:
            with pb.stage("forward", device="gpu"):
                time.sleep(0.06)
            pb.end_step()

        pb.print_report()
        captured = capsys.readouterr()
        assert "Pipeline Breakdown" in captured.out
        assert "forward" in captured.out
        assert "gpu" in captured.out

    def test_gpu_stage_without_cuda_falls_back(self) -> None:
        """If no CUDA but device='gpu', no crash, gpu_time stays 0."""
        pb = PipelineBreakdown()
        with pb:
            with pb.stage("forward", device="gpu"):
                pass
            pb.end_step()

        assert pb.n_steps == 1
        rec = pb.steps[0].stages[0]
        assert rec.gpu_time_ms == 0.0


# ======================================================================
# PipelineBreakdown — edge cases
# ======================================================================


class TestPipelineBreakdownEdgeCases:
    def test_clock_skew_warning(self, mock_cuda, caplog) -> None:
        """When gpu_ms > wall_ms + 1, warning is logged and value clamped."""
        torch.cuda.Event.return_value.elapsed_time.return_value = 500.0

        pb = PipelineBreakdown()
        with pb:
            with pb.stage("forward", device="gpu"):
                time.sleep(0.001)
            pb.end_step()

        assert "Clock skew" in caplog.text
        rec = pb.steps[0].stages[0]
        assert rec.gpu_time_ms <= rec.wall_time_ms

    def test_forgotten_end_step(self, mock_cuda) -> None:
        """Stages left open when exiting the outer context should auto-close."""
        pb = PipelineBreakdown()
        with pb, pb.stage("forward", device="gpu"):
            time.sleep(0.06)

        assert pb.n_steps == 1

    def test_no_stages_in_step(self, mock_cuda) -> None:
        """A step with no stages should produce no step records."""
        pb = PipelineBreakdown()
        with pb:
            pb.end_step()
        assert pb.n_steps == 0


# ======================================================================
# PipelineBreakdownCallback
# ======================================================================


class TestPipelineBreakdownCallback:
    def test_default_init(self) -> None:
        cb = PipelineBreakdownCallback()
        assert cb._output_path is None
        assert isinstance(cb._pb, PipelineBreakdown)

    def test_lifecycle_does_not_crash(self) -> None:
        """Hooks can be called without crashing."""
        cb = PipelineBreakdownCallback()
        trainer = MagicMock(spec=pl.Trainer)
        module = MagicMock(spec=pl.LightningModule)

        cb.on_train_start(trainer, module)
        cb.on_train_batch_start(trainer, module, None, 0)
        cb.on_train_batch_end(trainer, module, None, None, 0)
        cb.on_train_end(trainer, module)

    def test_report_at_end(self, capsys) -> None:
        """After training, print_report is called. If no stages were
        recorded, the report says so."""
        cb = PipelineBreakdownCallback()
        trainer = MagicMock(spec=pl.Trainer)
        module = MagicMock(spec=pl.LightningModule)

        cb.on_train_start(trainer, module)
        cb.on_train_batch_start(trainer, module, None, 0)
        cb.on_train_batch_end(trainer, module, None, None, 0)
        cb.on_train_end(trainer, module)

        captured = capsys.readouterr()
        assert "No steps recorded" in captured.out

    def test_json_export_at_end(self, tmp_path) -> None:
        """When output_path is set, JSON is written at train end."""
        path = tmp_path / "callback_report.json"
        cb = PipelineBreakdownCallback(output_path=str(path))
        trainer = MagicMock(spec=pl.Trainer)
        module = MagicMock(spec=pl.LightningModule)

        cb.on_train_start(trainer, module)
        cb.on_train_batch_start(trainer, module, None, 0)
        cb.on_train_batch_end(trainer, module, None, None, 0)
        cb.on_train_end(trainer, module)

        data = json.loads(path.read_text())
        assert "n_steps" in data
        assert data["n_steps"] == 0

    def test_pb_property(self) -> None:
        cb = PipelineBreakdownCallback()
        assert cb.pb is cb._pb


class TestPipelineBreakdownCallbackWarmupEpochs:
    """Tests for warmup_epochs in PipelineBreakdownCallback."""

    def test_default_warmup_epochs(self) -> None:
        cb = PipelineBreakdownCallback()
        assert cb.warmup_epochs == 0

    def test_negative_warmup_epochs_raises(self) -> None:
        with pytest.raises(ValueError, match="warmup_epochs"):
            PipelineBreakdownCallback(warmup_epochs=-1)

    def test_warmup_epochs_skips_callback(self, mock_cuda) -> None:
        """With warmup_epochs=1, epoch 0 steps are skipped."""
        cb = PipelineBreakdownCallback(warmup_epochs=1)
        trainer = MagicMock(spec=pl.Trainer)
        module = MagicMock(spec=pl.LightningModule)

        cb.on_train_start(trainer, module)

        # Epoch 0 (warmup)
        trainer.current_epoch = 0
        cb.on_train_epoch_start(trainer, module)
        cb.on_train_batch_start(trainer, module, None, 0)
        assert cb._in_step is False
        cb.on_train_batch_end(trainer, module, None, None, 0)
        cb.on_train_epoch_end(trainer, module)

        assert cb._pb.n_steps == 0

    def test_warmup_epochs_then_profiles(self, mock_cuda) -> None:
        """Epoch 0 skipped (_in_step=False), epoch 1 has _in_step=True.

        Note: ``n_steps`` stays 0 regardless because the callback never
        pushes stages (users call ``pb.stage()`` manually in
        ``training_step``). We verify warmup via ``_in_step`` state.
        """
        cb = PipelineBreakdownCallback(warmup_epochs=1)
        trainer = MagicMock(spec=pl.Trainer)
        module = MagicMock(spec=pl.LightningModule)

        cb.on_train_start(trainer, module)

        # Epoch 0 (warmup)
        trainer.current_epoch = 0
        cb.on_train_epoch_start(trainer, module)
        cb.on_train_batch_start(trainer, module, None, 0)
        assert cb._in_step is False
        cb.on_train_batch_end(trainer, module, None, None, 0)
        cb.on_train_epoch_end(trainer, module)

        # Epoch 1 (real)
        trainer.current_epoch = 1
        cb.on_train_epoch_start(trainer, module)
        cb.on_train_batch_start(trainer, module, None, 0)
        assert cb._in_step is True
        cb.on_train_batch_end(trainer, module, None, None, 0)
        cb.on_train_epoch_end(trainer, module)

    def test_warmup_epochs_warning_at_fit_start(self, mock_cuda, caplog) -> None:
        """on_fit_start warns when warmup_epochs >= max_epochs."""
        cb = PipelineBreakdownCallback(warmup_epochs=3)
        trainer = MagicMock(spec=pl.Trainer)
        trainer.max_epochs = 3
        module = MagicMock(spec=pl.LightningModule)

        cb.on_fit_start(trainer, module)
        assert "warmup_epochs=3 >= max_epochs=3" in caplog.text

    def test_warmup_epochs_no_warning_when_smaller(self, mock_cuda, caplog) -> None:
        """No warning when warmup_epochs < max_epochs."""
        cb = PipelineBreakdownCallback(warmup_epochs=1)
        trainer = MagicMock(spec=pl.Trainer)
        trainer.max_epochs = 5
        module = MagicMock(spec=pl.LightningModule)

        cb.on_fit_start(trainer, module)
        assert "warmup_epochs" not in caplog.text
