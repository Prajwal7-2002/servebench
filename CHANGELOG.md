# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
