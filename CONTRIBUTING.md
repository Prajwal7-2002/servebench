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

## Submitting Pull Requests

1. Fork the repository and create a branch from `main`.
2. Make your changes and add tests under `tests/` covering new functionality.
3. Verify that all tests pass (`uv run pytest -q`) and lints pass (`uv run ruff check src tests`).
4. Submit a Pull Request with a clear explanation of what changed, why, and reproducible benchmark commands.
