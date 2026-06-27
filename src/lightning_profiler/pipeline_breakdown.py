"""
Module 2: PipelineBreakdown — "the where"  (PLAN — NOT IMPLEMENTED YET)

Once Module 1 identifies which device (CPU or GPU) is the bottleneck, this module
drills into the specific pipeline stage that consumes the most time.

Goal
----
Wrap the logical stages of a training step with per-device timers and output
a per-stage breakdown:
  - CPU stages: data loading, per-image transforms, batching, CPU→GPU transfer
  - GPU stages: data transfer (GPU side), forward pass, loss computation,
                backward pass, optimizer step

Design sketches
---------------

1.  **Context-manager API** (flexible, for manual instrumentation):

        with PipelineBreakdown(trainer) as pb:
            with pb.stage("data_loading"):
                batch = next(loader)
            with pb.stage("preprocessing"):
                batch = transforms(batch)
            with pb.stage("to_gpu"):
                batch = batch.to(device, non_blocking=True)
            with pb.stage("forward", device="gpu"):
                y = model(x)
            with pb.stage("loss", device="gpu"):
                loss = loss_fn(y, t)
            with pb.stage("backward", device="gpu"):
                loss.backward()

        pb.print_summary()

2.  **Lightning callback** (automated, hooks into standard lifecycle):

        PipelineBreakdownCallback(
            stages=[
                "data_loading",     # CPU
                "preprocessing",    # CPU
                "to_gpu",           # CPU → GPU
                "forward",          # GPU
                "loss",             # GPU
                "backward",         # GPU
                "optimizer_step",   # GPU
            ]
        )

    It hooks ``on_train_batch_start/end`` plus the internal Lightning hooks
    (``on_after_backward``, ``on_before_optimizer_step``, etc.) to infer stage
    boundaries.  However, the ``data_loading`` stage is hard to capture because
    Lightning's DataLoader runs outside the callback cycle.

3.  **CUDA-event based GPU stage timing** (same technique as Module 1,
    but at finer granularity).  Each stage gets its own pair of CUDA events.

Output format
-------------
  - Console table: stage name | device | avg time | % of step
  - JSON export for further analysis
  - Chrome trace export (``chrome://tracing`` compatible JSON) for visual
    inspection

Open questions / risks
----------------------
  1. *DataLoader overlap* — CPU preprocessing for batch N+1 happens while GPU
     processes batch N.  Naive stage-level timing conflates batches.  A queue-
     aware approach or tracing batch IDs may be needed.
  2. *Compiled model internals* — Even with CUDA-event stage timing, a
     ``torch.compile``'d forward pass remains a single fused region.  To drill
     into the compiled graph we need Nsight Systems + NVTX (external).
  3. *Non-blocking transfers* — ``batch.to(device, non_blocking=True)`` returns
     immediately on the CPU; the transfer completes asynchronously.  The CUDA-
     event measurement correctly captures the transfer as part of GPU time, but
     attributing it to a "to_gpu" stage requires careful placement of events.

Implementation priority
-----------------------
  1. Context-manager API with CPU timer + CUDA-event GPU timer.
  2. Console reporter (table).
  3. Chrome-trace exporter.
  4. Lightning callback that auto-wraps available lifecycle hooks.
  5. Queue-aware DataLoader stage measurement.

This module is a stub until the above is implemented.
"""

from __future__ import annotations

# Placeholder: implementation will go here in a future iteration.
