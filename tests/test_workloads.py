"""Tests for workload presets, length distributions, bursty arrivals, and trace replay."""

import json
import random

import pytest

from servebench.workloads import (
    PRESETS,
    gamma_delays,
    generate_workload_requests,
    load_trace_file,
    sample_length,
)


def test_presets_defined():
    for name in ("chat", "rag", "code", "agent"):
        assert name in PRESETS
        assert PRESETS[name]["input_len"] > 0
        assert PRESETS[name]["output_len"] > 0


def test_sample_length_distributions():
    rng = random.Random(42)
    # fixed
    assert sample_length(100, "fixed", rng) == 100
    assert sample_length(100, None, rng) == 100

    # uniform (within [0.5 * 100, 1.5 * 100])
    u_samples = [sample_length(100, "uniform", rng) for _ in range(50)]
    assert all(50 <= s <= 150 for s in u_samples)

    # normal (around 100)
    n_samples = [sample_length(100, "normal", rng) for _ in range(100)]
    mean_n = sum(n_samples) / len(n_samples)
    assert 80 <= mean_n <= 120

    # lognormal (positive, skewed)
    ln_samples = [sample_length(100, "lognormal", rng) for _ in range(100)]
    assert all(s >= 4 for s in ln_samples)


def test_gamma_delays_mean_and_bursts():
    # inf rate -> 0 delay
    delays_inf = gamma_delays(10, rate=float("inf"))
    assert delays_inf == [0.0] * 10

    # Rate = 10 req/s -> expected gap = 0.1 s
    rate = 10.0
    delays_poisson = gamma_delays(1000, rate=rate, shape=1.0, seed=42)
    mean_gap = sum(delays_poisson) / len(delays_poisson)
    assert pytest.approx(mean_gap, rel=0.1) == 1.0 / rate

    # Bursty Gamma (shape = 0.2)
    delays_burst = gamma_delays(1000, rate=rate, shape=0.2, seed=42)
    mean_burst = sum(delays_burst) / len(delays_burst)
    assert pytest.approx(mean_burst, rel=0.15) == 1.0 / rate

    # Bursty should have higher variance / more near-zero gaps (clustering)
    near_zeros = sum(1 for d in delays_burst if d < 0.02)
    near_zeros_poisson = sum(1 for d in delays_poisson if d < 0.02)
    assert near_zeros > near_zeros_poisson


def test_generate_workload_requests_with_prefix():
    prompts = generate_workload_requests(
        n=5,
        input_len=30,
        output_len=20,
        shared_prefix_tokens=10,
        seed=123,
    )
    assert len(prompts) == 5
    for p in prompts:
        assert p.output_len == 20
        assert p.source == "workload"

    # All prompts should share the exact same first 10 words
    first_words = [p.text.split()[:10] for p in prompts]
    for w in first_words[1:]:
        assert w == first_words[0]


def test_load_trace_file(tmp_path):
    trace_path = tmp_path / "test_trace.jsonl"
    lines = [
        {"prompt": "What is Python?", "output_len": 50},
        {"messages": [{"role": "user", "content": "Explain KV caching"}], "max_tokens": 100},
        {"text": "Simple raw text query", "output_len": 30},
    ]
    with trace_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    prompts = load_trace_file(trace_path)
    assert len(prompts) == 3
    assert prompts[0].text == "What is Python?"
    assert prompts[0].output_len == 50
    assert "Explain KV caching" in prompts[1].text
    assert prompts[1].output_len == 100
    assert prompts[2].text == "Simple raw text query"
    assert prompts[2].output_len == 30

    # Test limit n
    limited = load_trace_file(trace_path, n=2)
    assert len(limited) == 2

    # Non-existent file
    with pytest.raises(FileNotFoundError):
        load_trace_file(tmp_path / "nonexistent.jsonl")

    # Empty file
    empty_path = tmp_path / "empty.jsonl"
    empty_path.touch()
    with pytest.raises(ValueError):
        load_trace_file(empty_path)
