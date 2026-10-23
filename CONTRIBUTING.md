# Contributing to servebench

Thank you for your interest in contributing!

`servebench` is an open-source benchmarking tool for LLM serving systems. We welcome contributions that improve measurement accuracy, support new engines, expand real-world workloads, and make local LLM benchmarking more honest and accessible.

---

## Development Setup

We use [uv](https://docs.astral.sh/uv/) for Python dependency management and environments.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Prajwal7-2002/servebench.git
   cd servebench
   ```

2. **Sync the environment**:
   ```bash
   uv sync
   ```
   This automatically installs all runtime and development dependencies into a local virtual environment.

---

## Running Tests

We run tests against both mock servers and real data:

```bash
uv run pytest -q
```

To run a specific test file:
```bash
uv run pytest tests/test_rigor.py
```

---

## Code Quality & Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
uv run ruff check src tests
```

### Architectural Principles

1. **Fast Startup**: Keep CLI import times fast (<0.5 s). Heavy libraries (like `matplotlib`) must be imported lazily only when a chart is drawn, not when `--help` or `run` starts.
2. **Local-First & Zero Extra Infrastructure**: `servebench` should work in 30 seconds on any laptop or workstation without requiring databases or external monitoring daemons.
3. **Statistical Honesty**: Always report percentiles with sample sizes, warn about small sample counts, and account for client loop lag.
4. **Apache 2.0**: All code is licensed under the Apache 2.0 license.

---

## 📬 Pull Request Rules & Submission Process

To keep `servebench` dependable and statistically rigorous, all pull requests must follow these rules:

### 1. Branch Naming Conventions
Always create a branch off `main` using descriptive prefixes:
* `feat/short-feature-name` (e.g. `feat/sglang-adapter`)
* `fix/short-bug-description` (e.g. `fix/client-lag-subtraction`)
* `perf/short-perf-description` (e.g. `perf/lazy-import-matplotlib`)
* `docs/short-doc-description` (e.g. `docs/statistical-rigor-guide`)
* `test/short-test-description` (e.g. `test/streaming-parser-mock`)

### 2. Conventional Commit Standards
Format your commit messages with clear conventional commit types:
```text
feat(adapters): add SGLang engine health check and runner
fix(metrics): correct client loop lag calculation on burst concurrency
perf(cli): make matplotlib import lazy to reduce startup latency to 0.28s
docs(readme): add vLLM vs mini-infer comparative benchmark card
test(rigor): add test cases for Wilson confidence interval bounds
```

### 3. Verification Against a Real or Mock Serving Engine
If your PR modifies request orchestration or metric calculation:
* Include the exact command line used to test it (e.g., against Ollama, vLLM, or `mini-infer`).
* Ensure both streaming (`stream=True`) and non-streaming modes work without memory leaks.

### 4. Pull Request Checklist
When opening a PR, our GitHub template will prompt you to verify:
- [x] CLI startup time remains < 0.5s (`uv run servebench --help`).
- [x] All latency percentiles report sample counts and warn on small samples ($N < 30$).
- [x] Zero extra infrastructure: No external database or server daemons required.
- [x] All 150+ unit tests pass: `uv run pytest -q`.
- [x] Linter passes: `uv run ruff check src tests`.
- [x] Documentation updated: `uv run mkdocs build --strict`.
