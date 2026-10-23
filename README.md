# servebench

[![PyPI](https://img.shields.io/pypi/v/servebench.svg?color=blue)](https://pypi.org/project/servebench/)
[![Sigstore Provenance](https://img.shields.io/badge/Sigstore-Provenance_Verified-success)](https://pypi.org/project/servebench/#security)
[![CI](https://github.com/Prajwal7-2002/servebench/actions/workflows/ci.yml/badge.svg)](https://github.com/Prajwal7-2002/servebench/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub_Pages-blue)](https://prajwal7-2002.github.io/servebench/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

Load-test any OpenAI-compatible LLM endpoint (Ollama, llama.cpp server, vLLM, SGLang, your
own engine) and get the numbers that matter for serving: **TTFT, TPOT, inter-token latency,
throughput and goodput**, at p50/p90/p99, over a sweep of load levels, with charts and
side-by-side comparison of runs.

No GPU or engine install is needed on the client: it is one async Python process talking HTTP.

> [!IMPORTANT]
> **Our Mission**: *Safe, secure, and optimized for every machine.*  
> Deploying and serving LLMs shouldn't require guessing concurrency numbers, tolerating latency spikes, or burning excess GPU energy. `servebench` provides researchers, developers, and autonomous agent engineers with exact MBU roofline telemetry, empirical bottlenecks, and automated auto-tuning to run local and cloud inference safely, reliably, and efficiently on any hardware.

> [!TIP]
> **Engineering Deep Dive**: Read our empirical case study on [Roofline Benchmarking & Auto-Tuning LLM Serving: How We Pushed Pure PyTorch to 27 tok/s](https://prajwal-pujari.vercel.app/blog/posts/roofline-serving-benchmarks.html) demonstrating saturation knee detection and 2.04 Tok/Joule energy scaling.

## Quickstart

### Installation

Install from PyPI:
```bash
pip install servebench
```
Or with [uv](https://docs.astral.sh/uv/):
```bash
uv tool install servebench
```

### Instant Local Benchmark (`servebench quick`)

`servebench quick` looks for Ollama, llama.cpp, LM Studio, vLLM, and SGLang on their usual ports,
picks the loaded model, measures, and answers in plain language (~1 minute):

```bash
servebench quick
```

A real run (Ollama with `OLLAMA_NUM_PARALLEL=4` on a laptop):

```text
servebench quick: Ollama 0.35.1 at http://127.0.0.1:11434/v1, model qwen2.5:0.5b
...
Ollama 0.35.1, qwen2.5:0.5b on AMD Ryzen 5 6600H, 15.3 GB RAM, NVIDIA GeForce RTX 3050 6GB Laptop GPU
One user:    first token in 43 ms, then 203 tokens/s
Capacity:    4 users at once (first token < 500 ms and >= 20 tokens/s each, for 90% of requests)
At 4 users:  361 tokens/s in total, 136 tokens/s each, first token p99 161 ms
```

With default Ollama settings it also explains *why* the limit is where it is ("requests
queue: the server works on about 1 at once") and suggests the fix. Then
`servebench share <run.json>` prints a Markdown table to paste into a post or issue and
makes a result card:

![servebench result card: 4 users at once, 203 tokens/s for one user, 43 ms to first token](https://raw.githubusercontent.com/Prajwal7-2002/servebench/main/results/quick-ollama-np4-card.png)

Full control:

```bash
# sweep a running server, e.g. Ollama (on Windows use 127.0.0.1, not localhost)
servebench run --url http://127.0.0.1:11434/v1 --model qwen2.5:0.5b \
    --concurrency 1,2,4,8 --input-len 256 --output-len 128 --out runs/ollama.json

# how many concurrent users fit your latency targets?
servebench find-capacity --url http://127.0.0.1:11434/v1 --model qwen2.5:0.5b \
    --out runs/ollama-cap.json

# compare multiple runs side by side, with charts and GPU telemetry
servebench compare-models runs/model-q4.json runs/model-fp16.json

# CI performance regression gate (fails with exit code 1 on regressions)
servebench check candidate.json --baseline baseline.json \
    --max-ttft-degradation-pct 15.0 --max-throughput-degradation-pct 10.0

# live terminal UI monitor (GPU wattage, VRAM, Tokens/Joule, queue depth)
servebench top --url http://127.0.0.1:11434/v1

# hardware roofline model (MFU %, MBU %, operational intensity, ASCII curve)
servebench roofline runs/ollama.json --gpu-name "RTX 3050 Laptop"

# automated engine auto-tuner & launch flag generator
servebench tune runs/ollama.json --engine ollama

# generate interactive HTML report
servebench report runs/ollama.json runs/llamacpp.json --out report.html
```

### Python API

Use Servebench programmatically in your own evaluation harnesses or notebooks:

```python
import servebench

res = servebench.benchmark(
    url="http://127.0.0.1:8000/v1",
    model="llama-3-8b",
    levels=[1, 2, 4, 8],
    preset="chat",
    gpu_telemetry=True,
)
print(f"Peak: {max(lv.summary.output_throughput for lv in res.levels):.1f} tok/s")

# Evaluate CI performance regression
report = servebench.check_regression("candidate.json", "baseline.json")
assert report.passed, "Serving latency regressed!"
```

## Hardware Roofline Analysis (`servebench roofline`)

Analyze whether your LLM serving bottleneck is **Memory-Bandwidth Bound** (typical during token generation / decode) or **Compute Bound** (large-batch prefill), and calculate your exact hardware utilization:

- **MBU (Memory Bandwidth Utilization %)**: Percentage of theoretical memory bus throughput achieved reading model weights ($\text{tok/s} \times \text{model\_bytes} / \text{bus\_bw}$).
- **MFU (Model FLOPs Utilization %)**: Percentage of theoretical peak FP16/BF16 TFLOPs achieved by GEMM matrix operations.
- Built-in hardware specs database: NVIDIA RTX 3050–4090, A100, H100, T4, and Apple Silicon M-series.

```bash
servebench roofline runs/ollama.json --gpu-name "RTX 3050 Laptop"
```

Outputs an ASCII roofline curve and bottleneck verdict:

```text
+------------------------------------------------------------+
| Roofline Model: NVIDIA GeForce RTX 3050 Laptop (qwen2.5:0.5b)
| Peak Bandwidth: 192 GB/s | Peak Compute: 14.6 TFLOPs | Ridge: 758.1 FLOP/B
+------------------------------------------------------------+
14.6 T |                                             ----------------
       |                                            /
       |                                           /
       |                                          /
       |                                         /
 0.1 T |                 * (c=4)                /
       |           * (c=1)                     /
       +------------------------------------------------------
      0.1        1.0       10.0      100.0     1000.0   (FLOPs/Byte)

[Verdict] Operating in Memory-Bandwidth Bound regime.
          Peak MBU: 51.0%  |  Peak MFU: 0.3%
```

## Production Auto-Tuning (`servebench tune`)

Don't guess serving configuration flags. `servebench tune` evaluates your concurrency sweeps, detects latency degradation knees and batching saturation points, and generates exact copy-pasteable launch flags:

```bash
servebench tune runs/ollama.json --engine ollama --slo-ttft-ms 500 --slo-tpot-ms 50
```

Generates recommended flags and environment variables tailored to your engine:

```text
======================================================================
  Auto-Tuning Recommendations: ollama
======================================================================
  Recommended Concurrency : 4
  Achieved Throughput     : 384.6 tokens/s
  Marginal Gain at Knee   : +40.4% throughput over previous level
  SLO Attainment          : 100.0% within latency budget

Engine Launch Directives:
  export OLLAMA_NUM_PARALLEL=4
  export OLLAMA_MAX_LOADED_MODELS=1
  ollama serve
======================================================================
```
Supported engines: `ollama`, `vllm`, `llamacpp`, and `mini-infer`.

## Real-Time Serving Monitor (`servebench top`)

Like `htop` or `nvtop`, but specifically designed for LLM inference serving. Displays an interactive terminal UI streaming:
- **GPU Telemetry**: Real-time wattage (W), VRAM allocation (MB), GPU core utilization (%), and live energy efficiency in **Tokens per Joule**.
- **Prometheus Scraper**: Active running requests, waiting queue depth, and prefix KV cache hit ratio.
- **Dynamic Sparklines**: Live token throughput and TTFT latency progression.

```bash
servebench top --url http://127.0.0.1:11434/v1 --interval 1.0
```

No model at hand? `uv tool install "servebench[mock]"` (or `pip install "servebench[mock]"`) adds
`servebench mock-server`, a stand-in server with set timings, so you can try every command without a GPU
([see How it's tested](#how-its-tested)).

## Learn LLM inference by measuring it

Want to understand how inference engines, batching, and queueing behave under the hood?
Follow our hands-on guide: **[docs/learn-inference.md](https://github.com/Prajwal7-2002/servebench/blob/main/docs/learn-inference.md)**.

It walks through 7 reproducible experiments with Ollama:
1. **Prefill scaling**: Why TTFT scales linearly with prompt length (~100 ms / 1k tokens).
2. **Decode bottleneck**: Why TPOT is constant (~4.9 ms/tok, ~200 tok/s) regardless of prompt length.
3. **Batching & slots**: Why `OLLAMA_NUM_PARALLEL=4` doubles throughput to 385 tok/s, and why 8 slots plateaus.
4. **The p99 latency knee**: Why p50 hides queues while p99 spikes 5× (197 ms → 943 ms) when slots fill.
5. **Little's Law in practice**: Proving $L = \lambda W$ with real serving numbers.
6. **Closed vs. open loop**: Why concurrency sweeps mask production breaking points.
7. **Benchmarking pitfalls**: Prompt cache hits, connection resets, and warm-up bias.

## Results

### Ollama on a laptop GPU: what `OLLAMA_NUM_PARALLEL` buys

Qwen2.5-0.5B (Q4_K_M) on Ollama 0.35.1, fully on an RTX 3050 6 GB laptop GPU. Synthetic
prompts of 256 tokens, 128 output tokens, 64 requests per level, `find-capacity` with
SLO TTFT < 500 ms and TPOT < 50 ms for 90% of requests. Only `OLLAMA_NUM_PARALLEL` changes.

![TTFT, TPOT, throughput and E2E latency vs concurrent requests for 1, 4 and 8 Ollama slots](https://raw.githubusercontent.com/Prajwal7-2002/servebench/main/results/ollama-parallel_files/sweep.png)

| Slots | Capacity (concurrent requests) | Completed req/s at capacity | Output tok/s | TTFT p50 / p99 | TPOT p50 | E2E p50 |
|---|---|---|---|---|---|---|
| 1 | 1 | 1.51 | 190 | 45 / 59 ms | 4.9 ms | 0.67 s |
| 4 | 4 | 3.16 | 385 | 101 / 197 ms | 9.4 ms | 1.30 s |
| 8 | 8 | 3.30 | 400 | 224 / 386 ms | 16 ms | 2.22 s |

- **Capacity follows the slot count exactly.** With one slot, a second request waits for the
  whole first answer (TTFT 688 ms at c=2), so only one user fits the SLO.
- **The GPU saturates at about 400 tok/s with 4 streams.** Going from 4 to 8 slots doubles
  the users who fit the SLO, but finishes only 4% more requests per second; each answer
  takes 71% longer (TPOT 9.4 → 16 ms) because the same GPU time is split more ways.
- So for this model and GPU, 4 slots maximise work done; 8 slots only help if you need more
  simultaneous users within the TTFT target and can accept slower answers.

Full report with every level: [results/ollama-parallel.md](https://github.com/Prajwal7-2002/servebench/blob/main/results/ollama-parallel.md).
Raw data: `results/ollama-np{1,4,8}.json` (every request's timings). Next: llama.cpp server
on the same GPU, and vLLM on a free Kaggle T4.

### mini-infer v0.2.1: Batched Paged Attention vs Serial Decoding

Live evaluation of Flagship #2 [mini-infer](https://github.com/Prajwal7-2002/mini-infer) (continuous batching LLM engine in pure PyTorch) running `Qwen2.5-0.5B-Instruct` on an **NVIDIA GeForce RTX 3050 6GB Laptop GPU** (12 requests/level):

```bash
servebench run --url http://127.0.0.1:8000/v1 --model Qwen/Qwen2.5-0.5B-Instruct --concurrency 1,2,4 --num-requests 12
```

| Concurrency | Requests (OK / Fail) | Throughput (`v0.1.0`) | Throughput (`v0.2.1`) | **Throughput Gain** | TTFT p50 | TPOT p50 | VRAM Peak | Power Avg | Tok/Joule |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **c = 1** | **12 / 0** | 9.6 tok/s | **13.6 tok/s** | **+41.6%** | 556 ms | 70 ms | 1,544 MB | 13.4 W | **1.01** |
| **c = 2** | **12 / 0** | 12.0 tok/s | **18.6 tok/s** | **+55.0%** | 1,162 ms | 98 ms | 1,548 MB | 13.1 W | **1.41** |
| **c = 4** | **12 / 0** | 11.7 tok/s | **27.0 tok/s** | **+130.8% (2.3×)** | 2,456 ms | 126 ms | 1,634 MB | 13.2 W | **2.04** |

- **DRAM Bandwidth Optimization**: Replacing unbatched iteration loops with a unified `[batch_size, 1]` forward pass over physical `BlockPool` KV caches dropped DRAM memory bandwidth pressure by over 4×, doubling concurrency throughput.
- **Energy Scaling**: Yields **2.04 Tokens / Joule** at $c=4$ while drawing only 13.2 W of package power.

## Metrics

| Metric | Meaning |
|---|---|
| TTFT | Request sent → first streamed chunk with non-empty content (role-only and empty chunks don't count) |
| TPOT | (E2E − TTFT) / (output tokens − 1) |
| ITL | Gaps between consecutive content chunks. Some servers put several tokens in one chunk; the report shows `tok/chunk` so you can tell |
| E2E | Request sent → end of stream |
| req/s, out tok/s | Completed requests and output tokens per second of wall time for the level |
| Goodput | Requests/s that meet the SLOs (default TTFT < 500 ms and TPOT < 50 ms; change with `--slo-ttft-ms`/`--slo-tpot-ms`, also at report time) |
| Errors | Failed requests grouped by kind (`HTTP 500`, `timeout`, `ConnectError`, ...) |

Token counts come from the server's `usage` (requested with `stream_options.include_usage`).
If the server sends none, servebench counts with a Hugging Face tokenizer if you pass
`--tokenizer Qwen/Qwen2.5-0.5B-Instruct` (install the `hf` extra: `servebench[hf]`), or falls back to
counting chunks. Every request records which source was used, and the report warns about
chunk counts.

## Load modes

- **Concurrency sweep** (`--concurrency 1,2,4,8`, closed loop): exactly N requests in flight
  until the level's requests are done. Shows how latency degrades as load grows.
- **Request rate** (`--rate 0.5,1,2,inf`, open loop): Poisson arrivals at R req/s, whether or
  not earlier requests finished; `inf` sends everything at once, `--max-concurrency` caps it.
  This is how you find the point where a server falls over.

Each level gets `--num-requests` requests (default `max(16, 4N)` for concurrency, 64 for
rate). Two warm-up requests run first and are not measured. Each run uses a new random
prompt seed (printed and saved in run.json): with a fixed seed, a second run sends the same
prompts and a server with prompt caching skips prefill (on Ollama TTFT dropped from 43 ms to
20 ms), so you would measure the cache. Pass `--seed N` to repeat a run exactly.

## Find capacity

`find-capacity` answers "how much load can this server take within my latency targets?"
with one number:

Real run: Ollama with `OLLAMA_NUM_PARALLEL=4` on the RTX 3050 from the results above.

```bash
servebench find-capacity --url http://127.0.0.1:11434/v1 --model qwen2.5:0.5b \
    --max 64 --num-requests 64 --out results/ollama-np4.json
```
```
         c=1  64 ok, 0 failed, 188.5 tok/s, TTFT p50 45 ms, 100% in SLO  -> pass
         c=2  64 ok, 0 failed, 274.0 tok/s, TTFT p50 71 ms, 100% in SLO  -> pass
         c=4  64 ok, 0 failed, 384.6 tok/s, TTFT p50 101 ms, 100% in SLO  -> pass
         c=8  64 ok, 0 failed, 375.7 tok/s, TTFT p50 1424 ms, 6% in SLO  -> FAIL
         c=6  64 ok, 0 failed, 405.2 tok/s, TTFT p50 566 ms, 47% in SLO  -> FAIL
         c=5  64 ok, 0 failed, 369.8 tok/s, TTFT p50 367 ms, 72% in SLO  -> FAIL
...
Capacity: 4 concurrent requests. At 4, 100% met the SLO; at 5, only 72%.
SLO: TTFT < 500 ms and TPOT < 50 ms for 90% of requests.
```

- A level **passes** when no request failed and at least `--target` (default 90%) of the
  requests met `--slo-ttft-ms` and `--slo-tpot-ms`.
- The search doubles the load from `--start` (1, 2, 4, 8, ...) until a level fails, then
  binary-searches between the last pass and the first fail until the gap is within
  `--tolerance` (10%) of the passing value. It never goes above `--max` (256).
- `--mode rate` searches Poisson arrival rates (req/s) instead of concurrency.
- Every level it tried is saved, so `servebench report` charts it like any sweep.
- Each level is measured once and the search assumes latency grows with load. Near the
  limit a noisy server can pass a level above one that failed; rerun to confirm.

## Prompts

- `--prompts synthetic` (default): `--input-len` tokens of random common words, a different
  prompt per request so prefix caches cannot reuse one prompt's work for the next.
  `--ignore-eos` (on by default) asks the server to always generate `--output-len` tokens;
  vLLM, SGLang and llama.cpp honour it, and servers that don't know it ignore it, so check
  `out tok/s` against `--output-len` when comparing.
- `--prompts dolly`: real prompts from
  [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
  (CC BY-SA 3.0, © Databricks), downloaded once (12.8 MB) into `~/.cache/servebench` (set
  `SERVEBENCH_CACHE` to move it). Not bundled with this repo. `--input-len` caps the prompt
  length in words.
- `--prompts my_prompts.jsonl` (`{"prompt": ...}` per line) or a `.txt` file, one prompt per
  line.

## How it works

```
prompts.py ──► load.py ─────────────► client.py ──HTTP/SSE──► server
 synthetic      concurrency workers    one streaming request,
 dolly          or Poisson arrivals    perf_counter per chunk
 file           + event-loop lag check
                     │
                     ▼
               metrics.py ──► run.json ──► report.py
               percentiles,               terminal table,
               goodput                    PNG charts, MD / HTML
```

- One `asyncio` process with `aiohttp` (the HTTP client vLLM's and SGLang's own serving
  benchmarks use). Timings use `time.perf_counter()`. servebench started on `httpx`; profiling
  showed its connection pool re-checks every connection on every request, so at 128
  concurrent requests the *client* capped out at ~45 req/s and inflated TTFT from 100 ms to
  1.6 s. With aiohttp the same test reaches ~400 req/s with TTFT 135 ms.
- Every request's chunk arrival times are saved in `run.json`, so reports can be recomputed
  (for example with different SLOs) without rerunning anything.
- The client checks itself: a background task measures how long a ready callback waits for
  the event loop. If the p99 is above 10 ms, the report warns that the client, not the server,
  may be the bottleneck.
## How it's tested

You can't check a benchmark's numbers against a real model, because nobody knows the true
answer in advance. So the tests use `servebench mock-server`: a FastAPI server that speaks the
same OpenAI streaming API but, instead of running a model, sends placeholder tokens with
timings we set exactly (TTFT, time per token, tokens per chunk, slowdown under load). The
tests then check servebench measures what was set:

- TTFT and TPOT within a few ms of the set delays, on Linux and Windows (CI on both)
- role-only and empty chunks don't count as the first token; several tokens per chunk
- N concurrent requests really means N in flight (the mock counts them); Poisson arrival rates
- connections are reused (the mock counts TCP connections: 4 requests, 1 connection)
- `find-capacity` finds the known limit: with TTFT = 80 ms × requests in flight and a 360 ms
  target, capacity must be exactly 4

Example: a mock set to TTFT 80 ms, 15 ms per token and 15% slowdown per extra request in
flight should show TPOT 15 × (1 + 0.15 × 15) ≈ 49 ms at 16 concurrent requests, and
servebench measures 49 ms ([examples/report.md](https://github.com/Prajwal7-2002/servebench/blob/main/examples/report.md)). The mock is a test
tool only; every number under [Results](#results) comes from a real server.

## Limitations

- One client process. Measured on a laptop (Ryzen 5 6600H, Windows) against the mock
  server: accurate to ~128 concurrent requests / ~400 req/s / ~12k streamed tokens/s (client
  loop lag p99 under 10 ms); at 256 concurrent it starts to work hard (lag ~20 ms). That
  covers what a single GPU serves; it is not a multi-machine load generator. The lag check
  warns when the client is struggling.
- The first requests of each level include opening TCP connections, as a real client would.
- TPOT and ITL are only as good as the token counts: without `usage` or `--tokenizer`, a
  server that sends several tokens per chunk will look slower per token than it is.
- Measures what the client sees over HTTP, including network and serialization, not the
  engine's internal timings.
- Text endpoints only (chat and completions); no images, tool calls or embeddings.

## Related tools

servebench does not try to replace the bigger tools; use them if they fit:
[vLLM's `vllm bench serve`](https://docs.vllm.ai/),
[NVIDIA GenAI-Perf](https://github.com/triton-inference-server/perf_analyzer),
[guidellm](https://github.com/vllm-project/guidellm),
[LLMPerf](https://github.com/ray-project/llmperf).
servebench is a small install that works against any engine, makes clean multi-run
comparisons, and is used to measure the rest of this series of projects on free hardware.

## Development

```bash
uv sync                      # creates .venv and installs the package + dev tools
uv run ruff check src tests
uv run pytest -q
```

No need to create or activate a virtualenv yourself; `uv run` uses the project's `.venv`.

## License

Apache 2.0 — Copyright (c) 2026 Prajwal Pujari. Free for commercial, research, and educational use. See [LICENSE](LICENSE) for details. The dolly-15k dataset, if you use it, is CC BY-SA 3.0 and is downloaded from Hugging Face at runtime, not distributed here.

---

## Citation

If you use `servebench` in your research, capacity benchmarking, or systems evaluations, please cite it as:

```bibtex
@software{pujari2026servebench,
  author = {Prajwal Pujari},
  title = {servebench: Production Capacity Profiler and Load-Testing Harness for Local LLM Serving Endpoints},
  year = {2026},
  url = {https://github.com/Prajwal7-2002/servebench}
}
```

