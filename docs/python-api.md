# Python API Reference

Servebench provides a clean, fully typed Python API for running benchmarks and evaluating performance regressions directly in your scripts, Jupyter notebooks, or custom evaluation harnesses.

---

## 1. Top-Level Imports

```python
import servebench
from servebench import (
    benchmark,
    check_regression,
    compare,
    CheckThresholds,
    CheckReport,
    RunResult,
    LevelResult,
    RequestResult,
    Summary,
    Stats,
)
```

---

## 2. Running a Benchmark (`servebench.benchmark`)

Executes a synchronous benchmark run and returns a `RunResult` object containing all level summaries, per-request timings, GPU telemetry, and automated bottleneck diagnoses.

```python
import servebench

res = servebench.benchmark(
    url="http://127.0.0.1:8000/v1",
    model="meta-llama/Llama-3-8B-Instruct",
    mode="concurrency",
    levels=[1, 2, 4, 8],
    num_requests=25,
    preset="chat",             # sets input_len=128, output_len=256
    burstiness=0.8,            # clustered arrivals
    gpu_telemetry=True,        # collect NVML GPU metrics
    scrape_metrics="auto",     # scrape server /metrics
    save_path="results.json",
)

# Inspect peak throughput
peak = max(lv.summary.output_throughput for lv in res.levels)
print(f"Peak Output Throughput: {peak:.1f} tok/s")

# Inspect primary bottleneck
if res.bottleneck_diagnosis:
    print(f"Primary Bottleneck: {res.bottleneck_diagnosis['primary_bottleneck']}")
```

### Signature & Parameters

```python
def benchmark(
    url: str,
    model: str,
    *,
    mode: Literal["concurrency", "rate"] = "concurrency",
    levels: list[float] | None = None,
    num_requests: int | None = 20,
    duration: float | None = None,
    input_len: int = 256,
    output_len: int = 128,
    preset: str | None = None,
    burstiness: float = 1.0,
    trace: str | Path | None = None,
    shared_prefix_tokens: int | None = None,
    gpu_telemetry: bool = True,
    scrape_metrics: str | None = None,
    endpoint: Literal["chat", "completions"] = "chat",
    slo_ttft_ms: float | None = 500.0,
    slo_tpot_ms: float | None = 50.0,
    warmup: int = 2,
    seed: int = 0,
    timeout: float = 300.0,
    api_key: str | None = None,
    label: str | None = None,
    save_path: str | Path | None = None,
) -> RunResult:
```

---

## 3. Evaluating Regressions (`servebench.check_regression`)

Compares a candidate run against a baseline run and verifies whether metrics stayed within tolerances. Accepts `RunResult` instances or paths to JSON files.

```python
import servebench

# Compare two runs
report = servebench.check_regression(
    candidate="candidate_run.json",
    baseline="baseline_run.json",
    thresholds=servebench.CheckThresholds(
        max_ttft_pct=15.0,
        max_tpot_pct=10.0,
        max_throughput_pct=10.0,
        max_error_rate_pct=1.0,
    )
)

if not report.passed:
    print("Performance regression detected!")
    for failure in report.failures:
        print(f" - {failure}")
```

---

## 4. Multi-Run Comparison (`servebench.compare`)

Loads and normalizes multiple benchmark runs for comparison.

```python
import servebench

runs = servebench.compare(["run1.json", "run2.json"])
for r in runs:
    peak_tok_s = max(lv.summary.output_throughput for lv in r.levels)
    print(f"{r.label}: {peak_tok_s:.1f} tok/s")
```

---

## 5. Hardware Roofline Analysis (`servebench.compute_roofline`)

Programmatically compute operational intensity, MBU %, and MFU % across concurrency sweeps:

```python
import servebench

# Load benchmark run
run = servebench.compare(["results/ollama-np4.json"])[0]

# Compute Roofline metrics for RTX 3050 Laptop GPU
analysis = servebench.compute_roofline(
    model_name=run.config.model,
    levels_data=[
        {"concurrency": int(lv.value), "output_tokens_per_s": lv.summary.output_throughput}
        for lv in run.levels
    ],
    gpu_name="RTX 3050 Laptop",
)

print(f"Peak MBU: {analysis.peak_mbu}% | Peak MFU: {analysis.peak_mfu}%")
print(f"Regime: {analysis.primary_regime}")

# Print ASCII roofline curve
print(servebench.render_ascii_roofline(analysis))
```

---

## 6. Auto-Tuning Engine (`servebench.evaluate_tuning`)

Analyze concurrency sweeps, determine saturation knees, and generate tailored launch configurations:

```python
import servebench

run = servebench.compare(["results/ollama-np4.json"])[0]

report = servebench.evaluate_tuning(
    run=run,
    engine_override="ollama",
    target_ttft_p50_ms=150.0,
    target_itl_p50_ms=40.0,
)

rec = report.recommendation
print(f"Optimal concurrency: {rec.recommended_concurrency}")
print(f"Launch command: {rec.launch_command_example}")
```

