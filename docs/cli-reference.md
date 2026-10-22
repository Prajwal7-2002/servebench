# CLI Command Reference

This document provides a complete reference for all Servebench CLI commands and options.

---

## 1. `servebench quick`

Automatically probes local ports (11434 for Ollama, 8080 for llama.cpp, 8000 for vLLM, 1234 for LM Studio) to identify running engines, tests model capacity for ~1 minute, and prints clear human-readable serving capacity.

```bash
servebench quick [OPTIONS]
```

### Options
- `--url TEXT`: Explicit server URL (e.g. `http://127.0.0.1:8000/v1`).
- `--model TEXT`: Model identifier to benchmark. If omitted, queries the server's `/models` endpoint and selects the active model.
- `--max-users INT`: Upper limit of concurrent users to test (default: 32).
- `--api-key TEXT`: Authorization bearer token if required by server.
- `--out PATH`: Destination path for results JSON (default: `servebench-quick-<engine>-<model>.json`).

---

## 2. `servebench run`

The primary load-testing command. Executes closed-loop concurrency sweeps or open-loop Poisson arrival rate benchmarks with full control over input/output distributions, repeats, GPU telemetry, and Prometheus metrics.

```bash
servebench run [OPTIONS]
```

### Essential Options
- `--url TEXT`: Server API base URL (e.g. `http://127.0.0.1:8000/v1`).
- `--model TEXT`: Model name.
- `--concurrency TEXT`: Comma-separated list of concurrent in-flight streams (e.g. `1,2,4,8,16`).
- `--rate TEXT`: Comma-separated list of Poisson request arrival rates in req/s (e.g. `5,10,20`). Pass `inf` for all-at-once burst.
- `--repeat INT` / `-r`: Number of independent repeat trials per load level (default: 1). Computes mean ± spread and Student's t 95% confidence intervals.
- `--config PATH` / `-c`: Path to YAML, TOML, or JSON benchmark configuration file.

### Workload & Traffic Shaping
- `--preset [chat|rag|code|agent]`: Standard workload preset for input/output token lengths:
  - `chat`: 128 input / 256 output tokens.
  - `rag`: 2048 input / 128 output tokens.
  - `code`: 512 input / 512 output tokens.
  - `agent`: 1024 input / 384 output tokens.
- `--input-len INT`: Average prompt length in tokens (default: 256).
- `--output-len INT`: Target generation length in tokens (default: 128).
- `--input-dist [fixed|normal|lognormal|uniform]`: Input token length distribution.
- `--output-dist [fixed|normal|lognormal|uniform]`: Output token length distribution.
- `--burstiness FLOAT`: Gamma shape parameter $k$ for request arrivals (default: 1.0 = Poisson; $<1.0$ = burst clusters followed by pauses).
- `--shared-prefix-tokens INT`: Common token prefix shared across requests to benchmark server KV cache reuse.
- `--trace PATH`: Path to a JSONL file with recorded user queries to replay.

### Observability & Telemetry
- `--gpu-telemetry / --no-gpu-telemetry`: Collect GPU core utilization, VRAM usage, and power draw via pure-Python NVML (default: enabled).
- `--scrape-metrics [auto|URL]`: Scrape Prometheus `/metrics` exposition format before and after each level.

### Export & Reports
- `--out PATH`: Path to save `run.json`.
- `--csv PATH`: Path to export level summary statistics as CSV.
- `--csv-requests PATH`: Path to export raw per-request timing traces as CSV.

---

## 3. `servebench check`

CI Performance Regression Gate. Compares a candidate benchmark run against a baseline run and exits with code 1 if regressions exceed thresholds.

```bash
servebench check CANDIDATE_JSON --baseline BASELINE_JSON [OPTIONS]
```

### Regression Thresholds
- `--max-ttft-degradation-pct FLOAT`: Maximum allowed TTFT p50 latency increase percentage (default: 15.0%).
- `--max-tpot-degradation-pct FLOAT`: Maximum allowed TPOT p50 latency increase percentage (default: 10.0%).
- `--max-throughput-degradation-pct FLOAT`: Maximum allowed throughput drop percentage (default: 10.0%).
- `--max-goodput-degradation-pct FLOAT`: Maximum allowed goodput drop percentage (default: 10.0%).
- `--max-error-rate-pct FLOAT`: Maximum allowed request failure percentage (default: 1.0%).

---

## 4. `servebench compare-models`

Renders a side-by-side terminal comparison table across multiple benchmarked models or engine configurations.

```bash
servebench compare-models run1.json run2.json [run3.json ...]
```

Compares:
- Peak Output Throughput (tok/s) and relative delta (+/-%)
- TTFT p50 (ms)
- TPOT p50 (ms)
- Goodput (tok/s meeting SLO)
- Average Power Draw (W)
- Energy Efficiency (**Tokens per Joule**)
- Primary Bottleneck Diagnosis

---

## 5. `servebench find-context-limit`

Systematically tests increasing prompt context lengths (doubling from start to max) to determine the maximum workable context window before GPU out-of-memory (OOM) or request timeout.

```bash
servebench find-context-limit --url http://127.0.0.1:8000/v1 --model llama-3-8b --start 512 --max 32768
```

---

## 6. `servebench find-capacity`

Automates doubling search and binary search to identify the exact saturation knee where latency crosses target SLOs.

```bash
servebench find-capacity --url http://127.0.0.1:8000/v1 --model llama-3-8b --slo-ttft-ms 500 --slo-tpot-ms 50
```

---

## 7. `servebench report`

Generates terminal tables, interactive HTML reports with charts, and CSV files from one or more saved `run.json` files.

```bash
servebench report run.json --out report.html
```

---

## 8. `servebench share`

Generates ready-to-copy Markdown tables and social share preview cards (1200x630 PNG) to include in GitHub issues or blog posts.

```bash
servebench share run.json
```

---

## 9. `servebench mock-server`

Runs an out-of-process OpenAI-compatible mock server with deterministic microsecond timings. Used for testing, calibration, and CI verification.

```bash
servebench mock-server --port 8000 --ttft-ms 100 --itl-ms 20
```

---

## 10. `servebench top`

Real-time interactive terminal UI dashboard displaying live GPU telemetry, Prometheus server metrics, queue depth, and dynamic throughput/TTFT sparklines.

```bash
servebench top [OPTIONS]
```

### Options
- `--url TEXT` / `-u`: Server API base URL (default: `http://127.0.0.1:11434/v1`).
- `--refresh FLOAT` / `-r`: Dashboard refresh interval in seconds (default: `0.5`).
- `--duration FLOAT` / `-d`: Exit after N seconds (useful for scripts/testing).
- `--inline`: Render inline directly in the terminal scrollback without switching to alternate screen buffer.

---

## 11. `servebench roofline`

Evaluates LLM serving performance against theoretical hardware limits using the Roofline model. Computes Memory Bandwidth Utilization (MBU %), Model FLOPs Utilization (MFU %), arithmetic intensity, and renders an ASCII roofline chart.

```bash
servebench roofline [OPTIONS] [RUN_FILE]
```

### Options
- `RUN_FILE`: Optional path to a `run.json` benchmark file.
- `--model TEXT` / `-m`: Model identifier (e.g. `qwen2.5:0.5b`, `llama3:8b-q4_k_m`).
- `--gpu TEXT` / `--gpu-name` / `-g`: GPU device name override (e.g., `rtx 3050`, `a100`, `t4`).
- `--bandwidth FLOAT` / `--gpu-mem-bw` / `-b`: Manual memory bandwidth override in GB/s.
- `--peak-tflops FLOAT` / `--gpu-tflops`: Manual peak FP16/BF16 compute override in TFLOPs.

---

## 12. `servebench tune`

Analyzes concurrency sweeps to identify throughput saturation knees and latency SLO boundaries, outputting recommended engine launch flags and environment variables.

```bash
servebench tune [OPTIONS] [RUN_FILE]
```

### Options
- `RUN_FILE`: Optional completed `run.json` to derive recommendations from. If omitted, runs an automated sweep against `--url`.
- `--engine TEXT` / `-e`: Target engine override (`ollama`, `vllm`, `llamacpp`, `mini-infer`).
- `--target-ttft-ms FLOAT`: Target TTFT p50 latency threshold in ms (default: `150.0`).
- `--target-itl-ms FLOAT`: Target ITL p50 latency threshold in ms (default: `40.0`).
- `--url TEXT` / `-u`: Base URL of OpenAI-compatible server (default: `http://127.0.0.1:11434/v1`).
- `--model TEXT` / `-m`: Model name when running automated sweep (default: `qwen2.5:0.5b`).

