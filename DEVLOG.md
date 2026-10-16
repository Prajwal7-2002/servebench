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
