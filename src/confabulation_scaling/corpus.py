"""Corpus frequency estimator using prebuilt inverted index."""

from __future__ import annotations

import math
import re
from pathlib import Path

from confabulation_scaling.index_builder import IndexBuilder

DEFAULT_INDEX_PATH = Path(__file__).parent / "data" / "index.json.gz"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class CorpusFrequencyEstimator:
    def __init__(self, index_path: Path | None = None):
        self._index_path = index_path or DEFAULT_INDEX_PATH
        self._index: dict | None = None

    def _load_index(self) -> dict:
        if self._index is None:
            builder = IndexBuilder()
            self._index = builder.load(self._index_path)
        return self._index

    def estimate(self, topic: str) -> float:
        index = self._load_index()
        tokens = _tokenize(topic)
        grams = _ngrams(tokens, 1) + _ngrams(tokens, 2)

        max_freq = 0.0
        for g in grams:
            freq = index.get(g, 0.0)
            if freq > max_freq:
                max_freq = freq

        return math.log10(max_freq + 1)
