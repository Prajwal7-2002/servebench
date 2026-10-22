import json

import pytest

from servebench import prompts as P


def test_synthetic_is_deterministic_distinct_and_sized():
    a = P.synthetic(5, 64, seed=1)
    assert a == P.synthetic(5, 64, seed=1)
    assert a != P.synthetic(5, 64, seed=2)
    assert len({p.text for p in a}) == 5  # no shared prompts for prefix caches to reuse
    assert all(len(p.text.split()) == 64 for p in a)


def _dolly_file(tmp_path):
    rows = [
        {"instruction": "Say hi", "context": "", "response": "hi", "category": "open_qa"},
        {
            "instruction": "Summarise this",
            "context": "A long passage " * 50,
            "response": "",
            "category": "summarization",
        },
        {"instruction": "What is 2+2?", "context": "", "response": "4", "category": "open_qa"},
    ]
    path = tmp_path / "dolly.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_dolly_joins_context_and_filters_by_length(tmp_path):
    path = _dolly_file(tmp_path)
    got = P.dolly(2, seed=0, max_words=20, path=path)
    assert {p.text for p in got} == {"Say hi", "What is 2+2?"}
    assert all(p.source == "dolly" for p in got)
    long = P.dolly(3, seed=0, path=path)
    assert any(p.text.startswith("Summarise this\n\nA long passage") for p in long)


def test_dolly_download_is_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("SERVEBENCH_CACHE", str(tmp_path))
    cached = tmp_path / "databricks-dolly-15k.jsonl"
    cached.write_text('{"instruction": "x", "context": ""}\n', encoding="utf-8")

    def no_network(*a, **k):
        raise AssertionError("should not download when cached")

    monkeypatch.setattr(P.urllib.request, "urlopen", no_network)
    assert P.download_dolly() == cached


def test_from_file_jsonl_and_txt(tmp_path):
    j = tmp_path / "p.jsonl"
    j.write_text('{"prompt": "one"}\n{"prompt": "two"}\n', encoding="utf-8")
    t = tmp_path / "p.txt"
    t.write_text("alpha\n\nbeta\n", encoding="utf-8")
    assert sorted(p.text for p in P.from_file(j, 2)) == ["one", "two"]
    assert sorted(p.text for p in P.from_file(t, 2)) == ["alpha", "beta"]
    assert len(P.from_file(t, 5)) == 5  # samples with replacement when asked for more


def test_load_prompts_rejects_unknown():
    with pytest.raises(ValueError, match="synthetic"):
        P.load_prompts("nope-not-a-file", 1, 10)


@pytest.mark.parametrize("length", [1, 4, 8, 16, 256])
def test_synthetic_prompts_stay_distinct_at_any_length(length):
    got = P.synthetic(20, length, seed=0)
    assert all(len(p.text.split()) == length for p in got)
    assert len({p.text for p in got}) >= (20 if length >= 4 else 2)
