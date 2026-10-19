"""Scraper for server-side Prometheus /metrics endpoints (vLLM, SGLang, llama.cpp, TGI).

Parses standard Prometheus text exposition format to extract queue depths, running requests,
KV cache usage, token counters, and prefix cache hit rates on the same timeline as client load.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

import aiohttp

# Standard known gauges across engines
GAUGE_PATTERNS = {
    "num_requests_waiting": (
        "vllm:num_requests_waiting",
        "sglang:num_requests_waiting",
        "llamacpp:requests_deferred",
        "tgi_queue_size",
    ),
    "num_requests_running": (
        "vllm:num_requests_running",
        "sglang:num_requests_running",
        "llamacpp:requests_processing",
        "tgi_requests_running",
    ),
    "gpu_cache_usage": (
        "vllm:gpu_cache_usage_factor",
        "sglang:token_usage",
        "llamacpp:kv_cache_usage_ratio",
    ),
}

# Standard known counters across engines (we calculate delta = end - start)
COUNTER_PATTERNS = {
    "prompt_tokens_total": (
        "vllm:prompt_tokens_total",
        "llamacpp:prompt_tokens_total",
    ),
    "generation_tokens_total": (
        "vllm:generation_tokens_total",
        "llamacpp:tokens_predicted_total",
    ),
    "prefix_cache_hits": (
        "vllm:prefix_cache_hit_rate",
        "vllm:request_prefix_cache_hit",
    ),
}

_METRIC_LINE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([+-]?(?:[0-9]*[.])?[0-9]+(?:[eE][+-]?[0-9]+)?|NaN|Inf|-Inf)(?:\s+\d+)?$"
)


def parse_prometheus_text(text: str) -> dict[str, float]:
    """Parse Prometheus text-based exposition format into a name -> float value mapping."""
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC_LINE_RE.match(line)
        if not m:
            continue
        name, _, raw_val = m.groups()
        try:
            val = float(raw_val)
            metrics[name] = val
        except ValueError:
            continue
    return metrics


def resolve_metrics_url(base_url: str, scrape_option: str | None) -> str | None:
    """Determine the metrics endpoint URL.

    - If scrape_option is a full URL (http://...), return it directly.
    - If scrape_option is 'auto' or True, infer http://<host>:<port>/metrics from base_url.
    - If scrape_option is None/False, return None.
    """
    if not scrape_option:
        return None
    if scrape_option.startswith("http://") or scrape_option.startswith("https://"):
        return scrape_option
    if scrape_option.lower() == "auto":
        parsed = urlparse(base_url)
        port_str = f":{parsed.port}" if parsed.port else ""
        netloc = parsed.netloc or f"{parsed.hostname}{port_str}"
        scheme = parsed.scheme or "http"
        return f"{scheme}://{netloc}/metrics"
    return None


async def scrape_prometheus(
    url: str, session: aiohttp.ClientSession | None = None, timeout_s: float = 3.0
) -> dict[str, float] | None:
    """Fetch and parse Prometheus metrics from `url`. Never raises on failure."""
    close_session = False
    if session is None:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_s))
        close_session = True
    try:
        async with session.get(url) as resp:
            if resp.status == 200:
                text = await resp.text()
                return parse_prometheus_text(text)
            return None
    except Exception:
        return None
    finally:
        if close_session:
            await session.close()


def extract_standard_metrics(
    start_metrics: dict[str, float] | None, end_metrics: dict[str, float] | None
) -> dict[str, float]:
    """Extract and reconcile standard serving gauges and counter deltas."""
    if not end_metrics:
        return {}

    out: dict[str, float] = {}

    # Extract gauges from end_metrics (point-in-time)
    for canonical_name, aliases in GAUGE_PATTERNS.items():
        for alias in aliases:
            if alias in end_metrics:
                out[canonical_name] = round(end_metrics[alias], 4)
                break

    # Extract counters as deltas (end - start)
    for canonical_name, aliases in COUNTER_PATTERNS.items():
        for alias in aliases:
            if alias in end_metrics:
                end_val = end_metrics[alias]
                start_val = start_metrics.get(alias, 0.0) if start_metrics else 0.0
                out[canonical_name] = round(max(0.0, end_val - start_val), 2)
                break

    return out
