# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-10-19

### Changed
- Refreshed package documentation with official PyPI installation instructions (`pip install servebench`).
- Converted all relative media and documentation links in README to absolute URLs for flawless rendering on PyPI.
- Added GitHub Pages automated documentation deployment workflow (`.github/workflows/docs.yml`).
- Expanded project metadata URLs on PyPI (Documentation, Repository, and Changelog).

## [1.0.0] - 2026-10-19

### Added
- **CI Performance Regression Gate (`servebench check`)**:
  - Gating command to evaluate candidate benchmark runs against established baseline runs.
  - Automatically verifies regression thresholds on TTFT p50 latency (+%), TPOT p50 (+%), output token throughput (-%), goodput (-%), and error rate (%).
  - Exits with status code `1` and rich failure diagnostics on violations, blocking pull requests or deployment pipelines.
  - Exits with status code `0` on clean pass with full metric delta reporting.
- **Stable Programmatic Python API**:
  - Public Python API exported from top-level package: `servebench.benchmark`, `servebench.check_regression`, `servebench.compare`.
  - Enables running benchmarks, tuning sweeps, and regression gates directly inside Python test suites, Jupyter notebooks, or evaluation pipelines.
- **Comprehensive Documentation Site**:
  - Complete MkDocs documentation site (`mkdocs.yml`) with Material theme.
  - Dedicated guides: Overview (`index.md`), CLI reference (`cli-reference.md`), Python API (`python-api.md`), Metrics & Math glossary (`metrics-glossary.md`), CI/CD integration (`ci-cd-guide.md`), and Engine configuration (`engines.md`).
- **Multi-Engine Serving Matrix**:
  - Comprehensive engine compatibility verification and tuning guide for Ollama, llama.cpp, vLLM, SGLang, and LM Studio.

## [0.2.0] - 2026-10-19

### Added
- **GPU Telemetry & Energy Efficiency (NVML)**:
  - Real-time GPU telemetry collection via pure Python `nvidia-ml-py` (`ctypes`-based zero C-compilation dependency).
  - Collects average and peak GPU core utilization (%), VRAM memory usage (MB), power draw (W), temperature (°C), and SM clock frequency (MHz).
  - Derived energy efficiency metric: **Tokens per Joule** ($\text{Output Throughput (tok/s)} / \text{Average Power (W)}$) and total energy consumed in Joules.
  - Zero-fail graceful degradation when non-NVIDIA or when NVML is unavailable.
- **Server Timing Breakdown & Overhead Isolation (P11)**:
  - Extracts server-side prefill and decode timings directly from SSE chunk payloads (llama.cpp `timings` and Ollama durations).
  - Calculates `queue_network_overhead_ms = Client TTFT - Server Prefill` to isolate network transit and server FIFO wait times before prompt evaluation starts.
  - Summarizes server prefill, decode, and overhead across p50, p90, p99 percentiles in terminal, HTML, and CSV reports.
- **Server Prometheus Scraper**:
  - Scrapes `/metrics` text exposition format from vLLM, SGLang, llama.cpp, and TGI servers before and after test levels.
  - Reconciles live gauges (`num_requests_waiting`, `num_requests_running`, `gpu_cache_usage`) and computes counter deltas (`prompt_tokens_total`, `generation_tokens_total`, `prefix_cache_hits`).
  - Supports `--scrape-metrics auto` or explicit URL.
- **Automated Bottleneck Diagnosis Engine**:
  - Automatically identifies whether the system is `Slot/Queue-Bound`, `Memory-Bandwidth-Bound`, `Compute-Bound`, `Client/Loop-Bound`, or `Balanced`.
  - Provides clear diagnostic severity, empirical evidence, and actionable tuning recommendations in reports and comparisons.
- **Real-World Workloads & Bursty Traffic**:
  - Workload presets: `chat` (128/256), `rag` (2048/128), `code` (512/512), and `agent` (1024/384).
  - Variable length distributions: `fixed`, `normal`, `lognormal`, and `uniform`.
  - Bursty arrival generator using Gamma distributions (`--burstiness < 1.0` for clustered burst arrivals followed by pauses).
  - Trace replay: `--trace requests.jsonl` to replay recorded query logs.
  - Server prefix caching testing: `--shared-prefix-tokens N` to measure KV cache reuse.
- **New CLI Commands**:
  - `servebench compare-models run1.json run2.json ...`: Side-by-side terminal comparison of latency, throughput, power draw, Tokens/Joule, and bottlenecks.
  - `servebench find-context-limit`: Probes doubling prompt lengths up to 32k tokens to identify maximum working context window before server OOM or timeout.

## [0.1.0] - 2026-10-19

### Added
- **Core CLI Commands**:
  - `servebench run`: Closed-loop concurrency sweeps and open-loop Poisson arrival rate load testing for any OpenAI-compatible endpoint.
  - `servebench quick`: Automatic discovery of local engines (Ollama, llama.cpp, LM Studio, vLLM, SGLang) on standard ports; runs a 1-minute capacity benchmark and outputs plain-language serving capacity.
  - `servebench find-capacity`: Doubling and binary search to automatically identify the highest concurrency/rate meeting target latency SLOs.
  - `servebench share`: Generates copy-paste Markdown summary tables and PNG social cards (1200x630) for GitHub issues and social sharing.
  - `servebench report`: Generates multi-run terminal comparison tables and interactive standalone HTML reports with matplotlib plots (light/dark mode).
  - `servebench mock-server`: Controllable out-of-process test double with microsecond-accurate simulated server timings.
- **Statistical Rigor**:
  - Multi-trial execution via `--repeat N` with calculation of mean ± spread and Student's t two-tailed 95% confidence intervals across repeats.
  - Honest sample-size handling: `Stats.count` tracking and prominent CLI/report warnings when $N < 100$ where p99 approximates sample maximums.
  - Steady-state throughput computation (`steady_state_output_throughput` and `steady_state_request_throughput`) excluding initial cold-start / connection ramp-up.
  - Benchmark config file support (`servebench run -c bench.yaml` / `.json` / `.toml`).
  - CSV export options (`--csv` for level metrics, `--csv-requests` for raw request traces).
- **Learning & Educational Resources**:
  - `docs/learn-inference.md`: "Learn LLM inference by measuring it" featuring 7 hands-on experiments on Ollama (prefill vs decode, memory-bandwidth-bound TPOT, slot batching saturation, queueing knee, Little's Law, open vs closed loop, and measurement pitfalls).
  - Raw benchmark datasets committed in `results/learn/` and `results/`.
- **System Telemetry & Detection**:
  - Automatic hardware detection: CPU, RAM, GPU model, VRAM size, driver version (`nvidia-smi`), and engine metadata (`/api/ps`).
  - In-stream event loop lag monitoring (`LagMonitor`) to verify client is not bottlenecking benchmark measurements.
- **Packaging & CI**:
  - Cross-platform GitHub Actions CI matrix covering Windows, Linux, and macOS (Apple Silicon) on Python 3.11, 3.12, 3.13.
  - PyPI trusted publishing workflow.
