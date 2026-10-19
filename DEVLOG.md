# DEVLOG

## 2026-10-16: M1-M3

- Built the streaming client, concurrency and Poisson load modes, metrics, run.json, terminal
  table, Markdown/HTML report with charts, and a fake OpenAI-compatible server. 37 tests.
- Bug found by the first demo: ITL p50 came out as 0.3 ms while TPOT was 15 ms. Cause:
  `asyncio.sleep` on Windows wakes on a ~15.6 ms timer tick, so the fake server sent tokens in
  pairs (31 ms gap, then 0 ms). The client was right; the server was not. Fixed with a coarse
  sleep plus a yield-spin, and added a regression test at 15 ms ITL.
- Same root cause shapes the event-loop lag check: measure how long a `call_soon` callback
  waits, not how late a sleep wakes, or Windows always looks "lagging".
- The first round of a concurrency level includes TCP connection setup (~70 ms for 4
  connections on Windows localhost). Kept it in, like real clients; noted in the README.
- Dolly-15k: downloaded at runtime (12.8 MB) into `~/.cache/servebench`, never bundled.
- Next: M4 real runs (Ollama + llama.cpp on the laptop, vLLM on Kaggle T4). Before the
  release, check the current state of vllm bench serve, GenAI-Perf, guidellm, LLMPerf.

## 2026-10-16: find-capacity

- `servebench find-capacity`: doubling then binary search for the highest load where no
  request fails and 90% meet the SLOs. Search logic is a pure function over a `probe`
  callback, so it is unit-tested without a server; CLI options now shared between commands.
- Check against the fake server (TPOT = 15 ms x (1 + 0.15 (N-1)), SLO TPOT < 40 ms): found
  12, the analytic answer, in 8 levels.
- First integration test used a 10 ms margin around the SLO and found 3 instead of 4: first-
  round connection setup and timer jitter add ~10-20 ms on Windows. Widened to 25 ms.
- One unexplained failure in 11 full-suite runs; could not reproduce in 9 reruns. Watch CI.

## 2026-10-16: first real runs (Ollama) and a connection bug

- Hardware is not CPU-only: Ollama runs qwen2.5:0.5b (Q4_K_M) fully on the laptop's
  RTX 3050 6 GB (`/api/ps` size_vram = size). TPOT 4.8 ms, ~200 tok/s for one stream.
- First run showed TTFT ~290 ms whatever the prompt length (46 -> 1054 tokens added only
  ~95 ms). Cause: servebench broke out of the stream at [DONE], httpx closed the connection,
  so every request reconnected; to `localhost` on Windows that tries ::1 first and costs
  ~270 ms. Fixed (read to end of body), test asserts 4 requests -> 1 connection, and
  `localhost` URLs now get a warning. TTFT p50 at c=1 dropped 289 -> 17-28 ms.
- Lesson: the fake server ran on 127.0.0.1 where reconnecting is ~free, so tests could not
  see it. Count connections, not just timings.
- Real finding: Ollama with default settings serves one request at a time here. c=2 gives
  the same 196-199 tok/s as c=1 and TTFT 686 ms (= one full request). Capacity under
  TTFT < 500 ms / TPOT < 50 ms: 1 concurrent request. Next: OLLAMA_NUM_PARALLEL=4.
- OLLAMA_NUM_PARALLEL=4 (same GPU, model, prompts 256 in / 128 out): capacity 1 -> 4,
  exactly the slot count. Throughput at c=4: 430 tok/s vs 196-199 tok/s default (2.2x);
  TPOT rises 4.8 -> 8.2 ms as 4 streams share the GPU, still far under 50 ms. Above 4,
  requests queue for a slot: TTFT p99 0.8-1.5 s. c=5/c=6 points are noisy (20-24 requests
  over 4 slots); more requests per level needed for a publishable chart.

## 2026-10-16: Ollama 1 / 4 / 8 slots, 64 requests per level

- Capacity = slots (1, 4, 8). Completed req/s at capacity: 1.51, 3.16, 3.30; GPU saturates
  at ~400 tok/s with 4 streams. 8 slots: 2x the users within the SLO, +4% req/s, +71% E2E.
  Blog angle: "more slots" buys admission, not throughput, once the GPU is busy.
- find-capacity's 8, 9, 10, 12 levels made tick labels collide; log-axis labels now skip a
  value within 1.2x of the previous label. `report` on a missing file: clean error.

## 2026-10-16: fake -> mock, README leads with real results

- Renamed the test server `fake-server` -> `mock-server` (`servebench[mock]`, `mock-model`):
  "mock" is the usual word, and "fake" next to benchmark numbers read badly. No alias kept,
  since nothing has been released yet.
- README now opens with real Ollama commands; find-capacity example is the real Ollama
  4-slot run; the mock lives in a "How it's tested" section explaining why it exists.

## 2026-10-16: performance: faster startup, aiohttp client

- Startup: importing the CLI took 1.13 s (matplotlib 0.43 s on every command, numpy 0.07 s).
  matplotlib now loads only when a chart is drawn; percentiles are plain Python (tested
  equal to numpy's linear method). Import 1.13 s -> 0.40 s; `servebench --help` ~1.2 -> ~0.7 s
  on Windows (Python itself is ~0.3 s). A test asserts the CLI doesn't import them.
- Client scaling: against the mock (TTFT 100 ms, 5 ms/token) throughput capped at ~60 req/s
  and client loop lag hit 927 ms at c=256: the client was the bottleneck and TTFT read 1 s
  instead of 100 ms. Profile: httpcore's pool calls is_idle()/has_expired() (a select() each)
  on every connection for every request: 4.5M calls, 69k selects for 256 requests. Also a
  failed `import sniffio` on every pool event (not the main cost). Switched to aiohttp:
  c=64 64 -> 224 req/s, c=128 44 -> 396, c=256 49 -> 571; lag p99 at c=256 927 -> 20 ms.
  Test suite 69 s -> 41 s. Verified against Ollama (both endpoints).
- httpx dropped from the package (stdlib urllib for /models and the dolly download); kept
  only as a test tool. numpy no longer a direct dependency.

## 2026-10-16: A1-A3: quick, machine info, share

- `servebench quick`: no arguments; finds Ollama/llama.cpp/LM Studio/vLLM/SGLang on their
  ports, picks the loaded model, ~1 minute capacity search, plain-language answer with the
  reason it stops (queueing) and an Ollama tip. RTX 3050 laptop, 4 slots: 40 s, capacity 4.
- run.json now records CPU, RAM, GPUs, engine + version, and model facts after the run
  (Ollama: size, quantization, % on GPU; llama.cpp: slots).
- `servebench share`: paste-ready Markdown + 1200x630 PNG card.
- Real server stall: first quick run took 317 s because Ollama paused 288 s mid-answer on
  one request (its own log: 2280 ms/token vs 5 ms). Reports now flag stalls (>5 s and >50x
  median gap); quick uses a 2-minute read timeout. Ollama's per-request log timings matched
  servebench's client-side TTFT/TPOT closely: a good independent check.

## 2026-10-19: L1 Learning Guide & Statistical Rigor

- Published `docs/learn-inference.md`: 7 hands-on experiments with Ollama explaining prefill vs
  decode, TTFT scaling, memory-bandwidth-bound TPOT, batching slots, queueing p99 knees,
  Little's Law, open vs closed loop, and measurement pitfalls. Backed by real laptop RTX 3050 runs
  in `results/learn/`.
- Statistical rigor:
  - `Stats.count` tracked on every percentile; report flags small sample sizes (<100 requests)
    where p99 is bounded by sample maximum.
  - Added `--repeat N`: runs each load level N times, calculates mean ± std and two-tailed 95%
    Student's t confidence intervals across trials for throughput, TTFT, and TPOT.
  - Steady-state throughput: computes throughput excluding initial connection and warm-up ramp-up.
  - Added benchmark config files (`servebench run -c bench.yaml` / `.json` / `.toml`) to check
    benchmark setups into git.
  - Added `--csv` and `--csv-requests` to export level summaries and per-request raw traces for
    pandas/Excel.

