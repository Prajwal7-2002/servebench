"""Real-world workload presets, length distributions, bursty arrivals, and trace replay.

Supports:
- Presets: chat, rag, code, agent
- Length distributions: fixed, normal, lognormal, uniform
- Bursty traffic arrival: Gamma distribution clustering
- Trace replay: JSONL files with real user queries and token lengths
- Shared prefix generation: for testing server-side KV prefix caching
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .prompts import _WORDS, Prompt

PRESETS = {
    "chat": {"input_len": 128, "output_len": 256, "description": "Conversational assistant"},
    "rag": {"input_len": 2048, "output_len": 128, "description": "Document search & QA"},
    "code": {"input_len": 512, "output_len": 512, "description": "Code generation & completion"},
    "agent": {"input_len": 1024, "output_len": 384, "description": "Multi-step tool agent"},
}


def sample_length(target_len: int, dist: str | None, rng: random.Random) -> int:
    """Sample an integer token length according to the specified distribution."""
    if not dist or dist == "fixed":
        return max(1, target_len)

    dist = dist.lower()
    if dist in ("normal", "gaussian"):
        sigma = max(1.0, target_len / 4.0)
        val = rng.gauss(target_len, sigma)
        return max(4, int(round(val)))

    if dist == "lognormal":
        # Log-normal with mean approximately target_len and moderate skew
        sigma = 0.5
        mu = math.log(max(1.0, float(target_len))) - (sigma**2) / 2.0
        val = rng.lognormvariate(mu, sigma)
        return max(4, int(round(val)))

    if dist == "uniform":
        low = max(4, int(target_len * 0.5))
        high = max(low + 1, int(target_len * 1.5))
        return rng.randint(low, high)

    # Unknown distribution defaults to fixed
    return max(1, target_len)


def gamma_delays(n: int, rate: float, shape: float = 1.0, seed: int = 0) -> list[float]:
    """Inter-arrival gaps for a Gamma process of average `rate` req/s.

    When shape == 1.0, this is identical to a Poisson process (exponential gaps).
    When shape < 1.0, arrivals form bursty clusters followed by pauses.
    """
    if math.isinf(rate):
        return [0.0] * n
    if shape <= 0:
        shape = 1.0
    rng = random.Random(seed)
    # Mean of Gamma(alpha, beta) is alpha * beta. To have mean = 1/rate, beta = 1/(rate * shape).
    beta = 1.0 / (rate * shape)
    return [rng.gammavariate(shape, beta) for _ in range(n)]


def generate_workload_requests(
    n: int,
    input_len: int,
    output_len: int,
    *,
    input_dist: str | None = None,
    output_dist: str | None = None,
    shared_prefix_tokens: int | None = None,
    seed: int = 0,
    tokenizer: str | None = None,
) -> list[Prompt]:
    """Generate variable-length prompt and output requests with optional shared prefix."""
    rng = random.Random(seed)

    # Shared prefix for testing server KV prefix caching
    shared_words: list[str] = []
    if shared_prefix_tokens and shared_prefix_tokens > 0:
        prefix_rng = random.Random(42)  # consistent shared prefix across trials
        shared_words = [prefix_rng.choice(_WORDS) for _ in range(shared_prefix_tokens)]

    prompts: list[Prompt] = []
    head = "Instruction for query context:".split()

    for _ in range(n):
        in_len = sample_length(input_len, input_dist, rng)
        out_len = sample_length(output_len, output_dist, rng)

        if shared_words:
            # Shared prefix first, then unique suffix words
            remaining = max(1, in_len - len(shared_words))
            words = list(shared_words) + [rng.choice(_WORDS) for _ in range(remaining)]
        else:
            n_head = min(len(head), max(1, in_len // 4))
            n_body = max(in_len - n_head, 1)
            words = head[:n_head] + [rng.choice(_WORDS) for _ in range(n_body)]

        text = " ".join(words)
        if tokenizer:
            from .tokens import truncate_to_tokens

            text = truncate_to_tokens(tokenizer, text, in_len)

        prompts.append(Prompt(text=text, source="workload", output_len=out_len))

    return prompts


def load_trace_file(path: str | Path, n: int | None = None) -> list[Prompt]:
    """Load real requests from a JSONL trace file.

    Expects each JSON line to contain:
    - `prompt`: str (or `messages`: list[dict])
    - `output_len` or `max_tokens`: int (default 128)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    prompts: list[Prompt] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "prompt" in row:
                text = str(row["prompt"])
            elif "messages" in row and isinstance(row["messages"], list):
                # Extract last user message or join
                text = " ".join(
                    m.get("content", "") for m in row["messages"] if isinstance(m, dict)
                )
            else:
                text = str(row.get("text", line))

            out_len = int(row.get("output_len") or row.get("max_tokens") or 128)
            prompts.append(Prompt(text=text, source=f"trace:{p.name}", output_len=out_len))
            if n is not None and len(prompts) >= n:
                break

    if not prompts:
        raise ValueError(f"No valid request rows found in trace file: {path}")
    return prompts
