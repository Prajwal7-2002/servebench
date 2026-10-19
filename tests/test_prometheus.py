"""Tests for Prometheus /metrics parsing, standard metric extraction, and endpoint resolution."""


from servebench.prometheus import (
    extract_standard_metrics,
    parse_prometheus_text,
    resolve_metrics_url,
)

SAMPLE_PROMETHEUS_TEXT = """
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting 4.0
# HELP vllm:num_requests_running Number of requests currently running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model="meta-llama/Llama-3-8B"} 8.0
# HELP vllm:gpu_cache_usage_factor GPU KV-cache usage factor between 0.0 and 1.0.
# TYPE vllm:gpu_cache_usage_factor gauge
vllm:gpu_cache_usage_factor 0.652
# HELP vllm:prompt_tokens_total Total prompt tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total 12450.0
# HELP vllm:generation_tokens_total Total generation tokens produced.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total 3120.0
# HELP vllm:prefix_cache_hit_rate Prefix cache hit rate.
vllm:prefix_cache_hit_rate 0.85
"""


def test_parse_prometheus_text():
    parsed = parse_prometheus_text(SAMPLE_PROMETHEUS_TEXT)
    assert parsed["vllm:num_requests_waiting"] == 4.0
    assert parsed["vllm:num_requests_running"] == 8.0
    assert parsed["vllm:gpu_cache_usage_factor"] == 0.652
    assert parsed["vllm:prompt_tokens_total"] == 12450.0
    assert parsed["vllm:generation_tokens_total"] == 3120.0
    assert parsed["vllm:prefix_cache_hit_rate"] == 0.85


def test_resolve_metrics_url():
    assert resolve_metrics_url("http://127.0.0.1:8000/v1", None) is None
    assert resolve_metrics_url("http://127.0.0.1:8000/v1", "") is None
    assert (
        resolve_metrics_url("http://127.0.0.1:8000/v1", "http://other-host:9090/metrics")
        == "http://other-host:9090/metrics"
    )
    assert (
        resolve_metrics_url("http://127.0.0.1:8000/v1", "auto")
        == "http://127.0.0.1:8000/metrics"
    )
    assert (
        resolve_metrics_url("https://api.myllm.com/v1", "AUTO")
        == "https://api.myllm.com/metrics"
    )


def test_extract_standard_metrics():
    start = {
        "vllm:prompt_tokens_total": 10000.0,
        "vllm:generation_tokens_total": 2000.0,
    }
    end = {
        "vllm:num_requests_waiting": 5.0,
        "vllm:num_requests_running": 10.0,
        "vllm:gpu_cache_usage_factor": 0.72,
        "vllm:prompt_tokens_total": 12500.0,
        "vllm:generation_tokens_total": 3100.0,
        "vllm:prefix_cache_hit_rate": 0.45,
    }

    metrics = extract_standard_metrics(start, end)
    assert metrics["num_requests_waiting"] == 5.0
    assert metrics["num_requests_running"] == 10.0
    assert metrics["gpu_cache_usage"] == 0.72
    assert metrics["prompt_tokens_total"] == 2500.0  # 12500 - 10000
    assert metrics["generation_tokens_total"] == 1100.0  # 3100 - 2000
    assert metrics["prefix_cache_hits"] == 0.45
