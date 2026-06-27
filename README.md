# Lightning Profiler

**Device-level and pipeline-level profiling for PyTorch Lightning training loops.**

---

## The Problem

Training computer vision models (segmentation, detection, OCR) on gaming-class
GPUs (RTX 5090, etc.) with PyTorch Lightning. You need to understand:

1. **"Who is the bottleneck?"** — Is the GPU idle, starving for CPU data? Or is
   the GPU fully saturated?
2. **"Where is time spent?"** — Which pipeline stage (data loading, transforms,
   forward, backward, etc.) consumes the most time?

Existing tools fall short:
- **SimpleProfiler** — CPU wall time only; misleading for GPU work.
- **AdvancedProfiler** — Python cProfile; CPU-only, no GPU insight.
- **PyTorchProfiler** — Operator-level; useless when `torch.compile()` fuses
  everything into a single kernel blob.
- **NVTX + Nsight Systems** — Excellent for deep dives, but external and
  requires manual inspection of a GUI.

---

## DeviceBottleneck — "The What" (Module 1)

A Lightning callback that answers **"CPU or GPU?"** with high accuracy and
minimal overhead.

### How it works

For each training step we capture:

- **`wall_time_ms`** — step duration from the CPU clock (`time.perf_counter()`).
- **`gpu_time_ms`** — actual GPU execution time (via `torch.cuda.Event` pairs
  recorded on the GPU device clock, immune to async launch semantics).
- **`bottleneck_ratio` = gpu_time_ms / wall_time_ms**

| Ratio | Verdict | Meaning |
|-------|---------|---------|
| `< 0.85` | CPU bottleneck | GPU finishes early and sits idle waiting for CPU data. |
| `> 0.98` | GPU bottleneck | GPU is fully saturated; step is compute-bound. |
| in between | Balanced | Both devices are well utilised. |

### Quick start

```python
import lightning as L
from lightning_profiler import DeviceBottleneckCallback

trainer = L.Trainer(
    callbacks=[
        DeviceBottleneckCallback(
            cpu_threshold=0.85,
            gpu_threshold=0.98,
            log_every_n_steps=50,
            output_path="logs/bottleneck.json",
            warmup_steps=5,
        ),
    ],
    ...
)
```

### Output at epoch end

```
[DeviceBottleneck] === Summary (1200 steps) ===
  CPU bottleneck:  34.2% of steps
  GPU bottleneck:  12.1% of steps
  Balanced:         53.7% of steps
  Avg wall:         456.2 ms
  Avg GPU:          389.1 ms
  Avg bottleneck ratio: 0.8531
  Ratio range:     [0.4210, 1.0000]
```

### API

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cpu_threshold` | `0.85` | Ratio below which the step is flagged as CPU-bottlenecked. |
| `gpu_threshold` | `0.98` | Ratio above which the step is flagged as GPU-bottlenecked. |
| `log_every_n_steps` | `50` | Log intermediate summary every N steps. `0` to disable. |
| `output_path` | `None` | Write per-step metrics as JSON to this path at `on_train_end`. |
| `warmup_steps` | `5` | Skip the first N steps (CUDA events are unreliable during GPU warmup). |

### When no GPU is available

The callback is a no-op when `torch.cuda.is_available()` is `False`. It logs
nothing and collects no metrics.

---

## PipelineBreakdown — "The Where" (Module 2)

*⚠️ Not yet implemented — design phase only.*

Once Module 1 identifies the bottleneck device, you need to drill into that
device's pipeline stages. The planned design is detailed in
[src/lightning_profiler/pipeline_breakdown.py](src/lightning_profiler/pipeline_breakdown.py),
covering:

- A context-manager API for manual instrumentation.
- A Lightning callback that auto-wraps lifecycle hooks.
- Stage-level CUDA event timing for GPU stages.
- Console table + Chrome trace + JSON export.

---

## Development

### Prerequisites

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) (package manager)
- [ruff](https://docs.astral.sh/ruff/) (linter / formatter)
- [ty](https://docs.astral.sh/ty/) (type checker)

### Setup

```bash
uv sync --dev          # create .venv and install dependencies
source .venv/bin/activate
```

### Pre-commit (optional but recommended)

Install the git hooks once:

```bash
pre-commit install
```

Now every commit will run ruff (lint + format) and ty (type-checking)
automatically. Run against all files at any time:

```bash
pre-commit run --all-files
```

### Lint, format & type-check

```bash
ruff check src tests   # lint
ruff format src tests  # format
ty check src          # type-check (tests/ excluded via --exclude)
```

### Test

```bash
pytest                 # runs tests with coverage report
```

Coverage is configured with a **80 % line-coverage threshold** (fails below).

### Project structure

```
lightning-profiler/
├── pyproject.toml          # project metadata, dependencies, tool configs
├── README.md
├── .gitignore
├── .pre-commit-config.yaml   # pre-commit hooks (ruff + ty)
├── src/
│   └── lightning_profiler/
│       ├── __init__.py
│       ├── device_bottleneck.py    # Module 1 implementation
│       └── pipeline_breakdown.py   # Module 2 plan & future implementation
└── tests/
    ├── __init__.py
    ├── conftest.py                  # fixtures and helpers
    └── test_device_bottleneck.py    # Module 1 tests
```

---

## Limitations

| Limitation | Explanation | Mitigation |
|------------|-------------|------------|
| **CUDA event overhead** | ~1-3 µs per event at pipeline-stage granularity. | ✅ Keep events at step level (not per-operator). |
| **`torch.cuda.synchronize()` cost** | ~50-200 µs penalty per synchronisation. | ✅ Synchronise once at the end of each step. |
| **Compiled model internals** | CUDA events still see a `torch.compile`'d forward as a single blob. | ✅ Use Nsight Systems + NVTX for rare deep dives. |
| **Multi-GPU / DDP** | Events are per-stream; DDP adds NCCL syncs. | ✅ Start single-GPU; expand to multi-GPU in a follow-up. |
| **No CUDA in CI** | Cannot test GPU path on non-CUDA runners. | ✅ Use `mock_cuda_available` fixture for unit tests. |

---

## Roadmap

- [x] **Module 1** — DeviceBottleneck (CPU vs GPU bottleneck detection)
- [ ] **Module 2** — PipelineBreakdown (per-stage breakdown)
  - [ ] Context-manager API
  - [ ] Stage-level CUDA events
  - [ ] Console reporter
  - [ ] Chrome trace exporter
  - [ ] Lightning callback auto-wrapper
- [ ] Multi-GPU / DDP support
- [ ] NVTX annotation export