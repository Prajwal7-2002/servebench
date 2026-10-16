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
