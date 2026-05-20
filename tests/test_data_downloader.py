"""Tests for data_downloader (unit tests with mocked HTTP)."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from confabulation_scaling.data_downloader import CorpusDownloader


def test_fetch_arxiv_uses_cache(tmp_path):
    cached = tmp_path / "arxiv_abstracts.json"
    data = [{"title": f"Paper {i}", "abstract": f"Abstract {i}"} for i in range(100)]
    cached.write_text(json.dumps(data))

    dl = CorpusDownloader(cache_dir=tmp_path)
    result = dl.fetch_arxiv_abstracts(n=50)
    assert result == cached


def test_fetch_wikipedia_uses_cache(tmp_path):
    cached = tmp_path / "wikipedia_sample.json"
    data = [{"title": f"Article {i}", "text": f"Content {i}"} for i in range(200)]
    cached.write_text(json.dumps(data))

    dl = CorpusDownloader(cache_dir=tmp_path)
    result = dl.fetch_wikipedia_sample(n=100)
    assert result == cached


def test_downloader_creates_cache_dir(tmp_path):
    cache = tmp_path / "subdir" / "cache"
    dl = CorpusDownloader(cache_dir=cache)
    assert cache.exists()
