## Description
<!-- Briefly describe what this PR does, why it is needed, and link any related issues. -->

Fixes #(issue)

## Type of Change
<!-- Please check all options that apply -->
- [ ] 🐛 Bug fix (non-breaking change fixing benchmark or metric calculation)
- [ ] ✨ New feature (new engine adapter, dataset loader, or reporting output)
- [ ] ⚡ Performance / Rigor optimization (reducing client loop lag, improving statistical precision)
- [ ] 📚 Documentation update (guides, docstrings, README)
- [ ] 🧪 Tests (adding test cases for statistical edge cases or mock server protocols)
- [ ] 💥 Breaking change (changes CLI flags or JSON schema output format)

## Benchmark / Engine Verification
<!-- What serving engines and hardware did you test this against? -->
* **Target Engine**: (e.g. vLLM / Ollama / TGI / SGLang / mini-infer)
* **Model**: (e.g. Qwen/Qwen2.5-0.5B-Instruct / Llama-3.1-8B)
* **Command Used**:
  ```bash
  servebench run --url http://127.0.0.1:8000/v1 --model <model> --num-requests 10
  ```

## Architectural Invariants Checklist
<!-- You MUST verify and check every item below before submitting -->
- [ ] **Fast CLI Startup**: CLI cold start remains under 0.5s (heavy libraries like matplotlib/scipy imported lazily).
- [ ] **Statistical Rigor**: All latency percentiles (p50, p90, p99) report sample count and warn on small samples ($N < 30$).
- [ ] **Zero Extra Infrastructure**: Works out of the box on consumer laptops without requiring external databases or background daemons.
- [ ] **Unit Tests**: Ran `uv run pytest -q` (all tests passing).
- [ ] **Code Formatting**: Ran `uv run ruff check src tests` with zero errors.
- [ ] **Clean Commits**: Follows conventional commit format (`feat:`, `fix:`, `perf:`, `docs:`, `test:`).
