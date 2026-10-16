"""Token counting when the server does not report `usage`.

Order of preference (recorded per request as `token_source`):
1. `usage` from the server (asked for with `stream_options.include_usage`)
2. a Hugging Face tokenizer, if `--tokenizer` was given (needs the `servebench[hf]` extra)
3. the number of content chunks (assumes one token per chunk; often right, never guaranteed)
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

Counter = Callable[[str], int]


@lru_cache(maxsize=4)
def load_tokenizer(name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "--tokenizer needs the servebench[hf] extra (Hugging Face transformers)"
        ) from e
    return AutoTokenizer.from_pretrained(name)


def make_counter(name: str | None) -> Counter | None:
    if not name:
        return None
    tok = load_tokenizer(name)
    return lambda text: len(tok.encode(text, add_special_tokens=False))


def truncate_to_tokens(name: str, text: str, n: int) -> str:
    """Cut `text` to exactly `n` tokens of the given tokenizer."""
    tok = load_tokenizer(name)
    ids = tok.encode(text, add_special_tokens=False)[:n]
    return tok.decode(ids)
