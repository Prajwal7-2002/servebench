"""Prompt sets: synthetic fixed-length prompts, databricks-dolly-15k, or your own JSONL file."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import httpx

DOLLY_URL = (
    "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/"
    "databricks-dolly-15k.jsonl"
)

# Common short English words: each is about one token in GPT-2/Qwen-style BPE vocabularies,
# so N words is roughly N tokens. Pass --tokenizer for exact lengths.
_WORDS = (
    "time year people way day man thing woman life child world school state family student "
    "group country problem hand part place case week company system program question work "
    "government number night point home water room mother area money story fact month lot "
    "right study book eye job word business issue side kind head house service friend father "
    "power hour game line end member law car city community name president team minute idea "
    "kid body information back parent face others level office door health person art war "
    "history party result change morning reason research girl guy moment air teacher force "
    "education foot boy age policy music market sense nation plan college interest death "
    "experience effect use class control care field development role effort rate heart drug "
    "show leader light voice wife police mind price report decision son view relationship "
    "town road arm difference value building action model season society tax director "
    "position player record paper space ground form event official matter center couple site "
    "project activity star table need court oil situation cost industry figure street image "
    "phone data picture practice piece land product doctor wall patient worker news test movie"
).split()


@dataclass(frozen=True)
class Prompt:
    text: str
    source: str = "synthetic"


def cache_dir() -> Path:
    root = os.environ.get("SERVEBENCH_CACHE") or Path.home() / ".cache" / "servebench"
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def synthetic(n: int, input_len: int, seed: int = 0, tokenizer: str | None = None) -> list[Prompt]:
    """`n` distinct prompts of ~`input_len` tokens.

    Every prompt is different random words, so servers with prefix caching cannot reuse
    one prompt's KV cache for the next (only the short instruction prefix is shared).
    """
    rng = random.Random(seed)
    head = "Write a long story that uses these words:".split()
    # The instruction takes at most a quarter of the prompt. Before this cap, an 8-word
    # prompt was only the 8-word instruction, so every prompt was identical.
    n_head = min(len(head), input_len // 4)
    out = []
    for _ in range(n):
        n_body = max(input_len - n_head, 1)
        if tokenizer:
            # extra words, then cut to the exact token count
            n_body = max(input_len, 1)
        words = head[:n_head] + [rng.choice(_WORDS) for _ in range(n_body)]
        text = " ".join(words)
        if tokenizer:
            from .tokens import truncate_to_tokens

            text = truncate_to_tokens(tokenizer, text, input_len)
        out.append(Prompt(text, "synthetic"))
    return out


def download_dolly(force: bool = False) -> Path:
    """Download dolly-15k (~13 MB, CC BY-SA 3.0) once into the cache dir."""
    path = cache_dir() / "databricks-dolly-15k.jsonl"
    if path.exists() and not force:
        return path
    tmp = path.with_suffix(".part")
    with httpx.stream("GET", DOLLY_URL, follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for block in r.iter_bytes():
                f.write(block)
    tmp.replace(path)
    return path


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dolly(n: int, seed: int = 0, max_words: int | None = None, path: Path | None = None):
    """`n` real prompts sampled from dolly-15k (instruction + context)."""
    rows = _read_jsonl(path or download_dolly())
    texts = []
    for r in rows:
        text = r["instruction"].strip()
        if r.get("context", "").strip():
            text += "\n\n" + r["context"].strip()
        if max_words is None or len(text.split()) <= max_words:
            texts.append(text)
    return _sample(texts, n, seed, "dolly")


def from_file(path: Path, n: int, seed: int = 0) -> list[Prompt]:
    """Your own prompts: JSONL with a `prompt` field per line, or plain text, one per line."""
    if path.suffix == ".jsonl":
        texts = [r["prompt"] for r in _read_jsonl(path)]
    else:
        texts = [t for t in path.read_text(encoding="utf-8").splitlines() if t.strip()]
    return _sample(texts, n, seed, path.name)


def _sample(texts: list[str], n: int, seed: int, source: str) -> list[Prompt]:
    if not texts:
        raise ValueError(f"no prompts found in {source}")
    rng = random.Random(seed)
    picked = rng.sample(texts, n) if n <= len(texts) else [rng.choice(texts) for _ in range(n)]
    return [Prompt(t, source) for t in picked]


def load_prompts(
    kind: str, n: int, input_len: int, seed: int = 0, tokenizer: str | None = None
) -> list[Prompt]:
    if kind == "synthetic":
        return synthetic(n, input_len, seed, tokenizer)
    if kind == "dolly":
        # input_len acts as an upper bound in words for real prompts
        return dolly(n, seed, max_words=input_len)
    path = Path(kind)
    if not path.exists():
        raise ValueError(f"--prompts must be 'synthetic', 'dolly' or a file path, got {kind!r}")
    return from_file(path, n, seed)
