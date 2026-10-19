# Servebench

**Load-test any OpenAI-compatible LLM endpoint with statistical rigor, hardware telemetry, and automated bottleneck diagnosis.**

[![PyPI version](https://img.shields.io/pypi/v/servebench.svg)](https://pypi.org/project/servebench/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## Why Servebench?

Benchmarking LLM serving engines (vLLM, Ollama, llama.cpp, SGLang, LM Studio, TGI) is notoriously difficult:
- **Client Conflation**: Many benchmark tools schedule requests on busy client event loops, reporting artificial client-side scheduling latency as slow server response time.
- **Statistical Inaccuracy**: Tools frequently report "p99 latency" on 10 or 20 requests—a mathematical fallacy where p99 is simply the sample maximum.
- **Conflated Timings**: Standard TTFT mixes client network transit, server FIFO queue delays, and prompt evaluation compute into a single opaque number.
- **Black-box Servers**: Benchmarks typically observe only external HTTP response streams, ignoring internal GPU power draw, VRAM usage, and server queue depth.

**Servebench solves these problems:**
1. **Accurate Timings & Client Lag Monitoring**: Continuously monitors client event loop lag and flags runs where the client is the bottleneck.
2. **Statistical Rigor**: Enforces sample count awareness ($N \ge 100$ for honest p99s), multi-trial repeats with Student's t 95% confidence intervals, and steady-state throughput calculations.
3. **P11 Server Breakdown**: Isolates queueing overhead from prompt evaluation: `Queue Overhead = Client TTFT - Server Prefill`.
4. **GPU Telemetry (NVML)**: Measures real-time GPU power (W), temperature, clocks, and computes energy efficiency in **Tokens per Joule**.
5. **Automated Bottleneck Diagnosis**: Automatically identifies if the system is `Slot/Queue-Bound`, `Memory-Bandwidth-Bound`, `Compute-Bound`, or `Client-Bound`.
6. **CI Performance Regression Gate**: `servebench check candidate.json --baseline base.json` to block deployments that regress latency or throughput.

---

## Quickstart

### Installation

Install via pip or uv:
```bash
pip install servebench
```
Or with optional dependencies (e.g. for mock server):
```bash
pip install "servebench[mock]"
```

### 1. Instant Local Benchmark (`servebench quick`)
Auto-detects any running local server (Ollama, llama.cpp, vLLM, LM Studio) on standard ports and runs a 1-minute capacity test:
```bash
servebench quick
```

Output:
```text
servebench quick: Ollama 0.6.0 at http://127.0.0.1:11434, model qwen2.5:0.5b
  c=1   16 ok, 0 failed,  64.2 tok/s, TTFT p50 43 ms
  c=2   16 ok, 0 failed, 126.8 tok/s, TTFT p50 44 ms
  c=4   16 ok, 0 failed, 203.1 tok/s, TTFT p50 47 ms
  c=8   16 ok, 0 failed, 241.5 tok/s, TTFT p50 115 ms

Ollama 0.6.0, qwen2.5:0.5b on Intel Core i7-13700H, NVIDIA RTX 3050 6GB (driver 551.76)
  Capacity: 4 concurrent users at once (203 tokens/s, first token in 47 ms)
  At 8 users: first token slowed to 115 ms (server queues requests for available slots)
```

### 2. Full Concurrency Sweep (`servebench run`)
```bash
servebench run --url http://127.0.0.1:8000/v1 --model llama-3-8b --concurrency 1,2,4,8,16 --repeat 3
```

### 3. CI Regression Gate (`servebench check`)
Fail your CI build if TTFT p50 increases by $>15\%$ or throughput drops by $>10\%$:
```bash
servebench check candidate.json --baseline baseline.json
```

---

## Architecture Overview

```
                      ┌────────────────────────────────────────┐
                      │              servebench                │
                      │                                        │
                      │  ┌──────────────┐    ┌──────────────┐  │
                      │  │ Closed-Loop  │    │  Open-Loop   │  │
                      │  │ Concurrency  │    │   Poisson    │  │
                      │  └──────┬───────┘    └──────┬───────┘  │
                      └─────────┼───────────────────┼──────────┘
                                │                   │
                    HTTP/SSE    ▼                   ▼
                      ┌────────────────────────────────────────┐
                      │          Inference Server              │
                      │  (Ollama, vLLM, llama.cpp, SGLang)     │
                      └─────────┬───────────────────┬──────────┘
                                │                   │
                    NVML Probe  ▼                   ▼  Prometheus /metrics
                      ┌────────────────────────────────────────┐
                      │            System Telemetry            │
                      │   - GPU Power (W)    - Queue Depth     │
                      │   - Tok / Joule      - KV Cache Ratio  │
                      └────────────────────────────────────────┘
```
