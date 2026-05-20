"""Inverted index constructor for n-gram document frequencies."""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from pathlib import Path


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class IndexBuilder:
    def build(self, abstracts_path: Path, wiki_path: Path) -> dict:
        doc_freq: Counter = Counter()
        total_docs = 0

        abstracts = json.loads(abstracts_path.read_text())
        for entry in abstracts:
            text = entry.get("abstract") or entry.get("text", "")
            title = entry.get("title", "")
            tokens = _tokenize(f"{title} {text}")
            grams = set(_ngrams(tokens, 1)) | set(_ngrams(tokens, 2))
            for g in grams:
                doc_freq[g] += 1
            total_docs += 1

        wiki_articles = json.loads(wiki_path.read_text())
        for entry in wiki_articles:
            text = entry.get("text", "")
            title = entry.get("title", "")
            tokens = _tokenize(f"{title} {text}")
            grams = set(_ngrams(tokens, 1)) | set(_ngrams(tokens, 2))
            for g in grams:
                doc_freq[g] += 1
            total_docs += 1

        index = {
            "_meta": {"total_docs": total_docs},
        }
        for gram, count in doc_freq.items():
            per_million = (count / total_docs) * 1e6
            index[gram] = per_million

        return index

    def save(self, index: dict, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(index, f)

    def load(self, path: Path) -> dict:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
