# CI/CD Performance Regression Gating Guide

Learn how to integrate Servebench into your CI/CD pipelines (GitHub Actions, GitLab CI) to catch latency degradations and throughput regressions before deploying model updates to production.

---

## 1. Why Gate Serving Performance in CI?

When updating:
- Engine versions (e.g. vLLM or llama.cpp upgrades)
- Model quantization configurations (e.g. FP16 to FP8 / AWQ)
- Serving hyperparameters (`--max-num-seqs`, chunked prefill, prefix caching)

A regression in TTFT or token throughput can silently degrade end-user response times or spike GPU cloud infrastructure costs. `servebench check` acts as an automated quality gate that returns exit code `1` if regressions exceed acceptable tolerances.

---

## 2. GitHub Actions Integration

Create `.github/workflows/servebench-regression.yml`:

```yaml
name: Serving Performance Regression Gate

on:
  pull_request:
    branches: [main]

jobs:
  performance-gate:
    runs-on: self-hosted # or GPU runner with access to LLM server
    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install Servebench
        run: |
          pip install servebench

      - name: Benchmark Candidate Server
        run: |
          servebench run \
            --url http://127.0.0.1:8000/v1 \
            --model target-model \
            --concurrency 1,4,8 \
            --num-requests 50 \
            --out candidate_run.json

      - name: Download Baseline Run
        run: |
          # Download established baseline from S3, release artifact, or git LFS
          curl -sSL https://my-bucket.s3.amazonaws.com/baselines/target-model-baseline.json -o baseline_run.json

      - name: Evaluate Regression Tolerances
        run: |
          servebench check candidate_run.json \
            --baseline baseline_run.json \
            --max-ttft-degradation-pct 15.0 \
            --max-tpot-degradation-pct 10.0 \
            --max-throughput-degradation-pct 10.0 \
            --max-error-rate-pct 1.0

      - name: Upload Candidate Run Artifact
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: candidate_run.json
```

---

## 3. Exit Codes & Gating Logic

- **Exit Code 0**: All tested load levels satisfied regression limits. The pull request passes.
- **Exit Code 1**: One or more metrics degraded beyond configured limits (e.g., TTFT increased by $>15\%$, throughput dropped by $>10\%$, or error rate exceeded $1\%$). The CI pipeline blocks merge.
