"""Tests for index_builder module."""

import json
import tempfile
from pathlib import Path

from confabulation_scaling.index_builder import IndexBuilder


def _write_corpus(tmp: Path, name: str, entries: list[dict]) -> Path:
    p = tmp / name
    p.write_text(json.dumps(entries))
    return p


def test_build_produces_per_million_frequencies(tmp_path):
    abstracts = [
        {"title": "Deep learning review", "abstract": "Neural networks are powerful."},
        {"title": "Transformer models", "abstract": "Attention is all you need."},
    ]
    wiki = [
        {"title": "Machine learning", "text": "Machine learning is a subset of AI."},
    ]
    abs_path = _write_corpus(tmp_path, "abs.json", abstracts)
    wiki_path = _write_corpus(tmp_path, "wiki.json", wiki)

    builder = IndexBuilder()
    index = builder.build(abs_path, wiki_path)

    assert "_meta" in index
    assert index["_meta"]["total_docs"] == 3
    # "learning" appears in at least 2 docs out of 3 → per_million = (2/3)*1e6
    assert "learning" in index
    assert index["learning"] > 0


def test_save_and_load_roundtrip(tmp_path):
    index = {"_meta": {"total_docs": 100}, "foo": 5000.0, "bar baz": 3000.0}
    out = tmp_path / "index.json.gz"

    builder = IndexBuilder()
    builder.save(index, out)
    loaded = builder.load(out)

    assert loaded["_meta"]["total_docs"] == 100
    assert loaded["foo"] == 5000.0
    assert loaded["bar baz"] == 3000.0


def test_bigrams_indexed(tmp_path):
    abstracts = [
        {"title": "neural network", "abstract": "A neural network model."},
    ]
    wiki: list[dict] = []
    abs_path = _write_corpus(tmp_path, "abs.json", abstracts)
    wiki_path = _write_corpus(tmp_path, "wiki.json", wiki)

    builder = IndexBuilder()
    index = builder.build(abs_path, wiki_path)

    assert "neural network" in index
