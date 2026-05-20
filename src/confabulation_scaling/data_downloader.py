"""Reference corpus fetcher for arXiv abstracts and Wikipedia articles."""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

CACHE_DIR = Path.home() / ".cache" / "confabulation_scaling"
SEMANTIC_SCHOLAR_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


class CorpusDownloader:
    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_arxiv_abstracts(self, n: int = 50000) -> Path:
        out = self.cache_dir / "arxiv_abstracts.json"
        if out.exists():
            existing = json.loads(out.read_text())
            if len(existing) >= n:
                return out

        abstracts: list[dict] = []
        batch = 100
        offset = 0
        queries = [
            "machine learning",
            "neural network",
            "deep learning",
            "transformer",
            "language model",
            "reinforcement learning",
            "computer vision",
            "natural language processing",
            "optimization",
            "generative model",
        ]

        for query in queries:
            if len(abstracts) >= n:
                break
            offset = 0
            while len(abstracts) < n:
                params = {
                    "query": query,
                    "offset": offset,
                    "limit": batch,
                    "fields": "title,abstract,year",
                }
                try:
                    resp = requests.get(
                        SEMANTIC_SCHOLAR_SEARCH, params=params, timeout=30
                    )
                    if resp.status_code == 429:
                        time.sleep(2)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                except (requests.RequestException, ValueError):
                    break

                papers = data.get("data", [])
                if not papers:
                    break

                for p in papers:
                    abstract = p.get("abstract") or ""
                    if abstract.strip():
                        abstracts.append(
                            {
                                "title": p.get("title", ""),
                                "abstract": abstract,
                                "year": p.get("year"),
                            }
                        )
                    if len(abstracts) >= n:
                        break

                offset += batch
                if data.get("next") is None:
                    break
                time.sleep(0.5)

        out.write_text(json.dumps(abstracts[:n]))
        return out

    def fetch_wikipedia_sample(self, n: int = 10000) -> Path:
        out = self.cache_dir / "wikipedia_sample.json"
        if out.exists():
            existing = json.loads(out.read_text())
            if len(existing) >= n:
                return out

        articles: list[dict] = []
        batch = 20

        while len(articles) < n:
            params = {
                "action": "query",
                "format": "json",
                "generator": "random",
                "grnnamespace": 0,
                "grnlimit": batch,
                "prop": "extracts",
                "exchars": 1200,
                "exintro": True,
                "explaintext": True,
            }
            try:
                resp = requests.get(WIKIPEDIA_API, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except (requests.RequestException, ValueError):
                time.sleep(1)
                continue

            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                extract = page.get("extract", "").strip()
                if extract:
                    articles.append(
                        {"title": page.get("title", ""), "text": extract}
                    )
            time.sleep(0.3)

        out.write_text(json.dumps(articles[:n]))
        return out
