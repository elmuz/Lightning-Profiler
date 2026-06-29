"""
Profiling utilities for PyTorch Lightning training loops.

This package provides callbacks and utilities to identify device-level
bottlenecks (CPU vs GPU) and pipeline-stage breakdowns in CV training
pipelines using gaming-class GPUs (RTX series).
"""

from .device_bottleneck import DeviceBottleneckCallback
from .pipeline_breakdown import PipelineBreakdown, PipelineBreakdownCallback

__all__ = ["DeviceBottleneckCallback", "PipelineBreakdown", "PipelineBreakdownCallback"]
