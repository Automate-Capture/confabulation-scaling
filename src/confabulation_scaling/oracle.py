"""Reference verification oracle using CrossRef and Semantic Scholar."""

from __future__ import annotations

import requests
import Levenshtein


CROSSREF_SEARCH = "https://api.crossref.org/works"
S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"


def _extract_surname(name: str) -> str:
    parts = name.strip().split()
    return parts[-1].lower() if parts else ""


def _score_candidate(ref: dict, candidate_title: str, candidate_year: int | None,
                     candidate_authors: list[str]) -> float:
    title_sim = Levenshtein.ratio(
        ref["title"].lower(), candidate_title.lower()
    )

    year_score = 0.0
    ref_year = ref.get("year")
    if ref_year is not None and candidate_year is not None:
        diff = abs(int(ref_year) - int(candidate_year))
        if diff == 0:
            year_score = 1.0
        elif diff == 1:
            year_score = 0.8

    author_score = 0.0
    ref_authors = ref.get("authors", [])
    if ref_authors and candidate_authors:
        ref_surname = _extract_surname(ref_authors[0])
        for ca in candidate_authors:
            cand_surname = _extract_surname(ca)
            sim = Levenshtein.ratio(ref_surname, cand_surname)
            if sim > author_score:
                author_score = sim

    return 0.6 * title_sim + 0.3 * year_score + 0.1 * author_score


class ReferenceOracle:
    def __init__(self, timeout: int = 15):
        self._timeout = timeout

    def _query_crossref(self, title: str) -> list[dict]:
        candidates = []
        try:
            resp = requests.get(
                CROSSREF_SEARCH,
                params={"query.title": title, "rows": 5},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            items = resp.json().get("message", {}).get("items", [])
            for item in items[:5]:
                authors = []
                for a in item.get("author", []):
                    family = a.get("family", "")
                    given = a.get("given", "")
                    authors.append(f"{given} {family}".strip())
                date_parts = item.get("published-print", item.get("created", {})).get(
                    "date-parts", [[None]]
                )
                year = date_parts[0][0] if date_parts and date_parts[0] else None
                candidates.append(
                    {
                        "title": " ".join(item.get("title", [])),
                        "year": year,
                        "authors": authors,
                    }
                )
        except (requests.RequestException, ValueError, KeyError):
            pass
        return candidates

    def _query_semantic_scholar(self, title: str) -> list[dict]:
        candidates = []
        try:
            resp = requests.get(
                S2_SEARCH,
                params={"query": title, "limit": 5, "fields": "title,year,authors"},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            papers = resp.json().get("data", [])
            for p in papers[:5]:
                authors = [a.get("name", "") for a in p.get("authors", [])]
                candidates.append(
                    {
                        "title": p.get("title", ""),
                        "year": p.get("year"),
                        "authors": authors,
                    }
                )
        except (requests.RequestException, ValueError, KeyError):
            pass
        return candidates

    def verify(self, reference: dict) -> float:
        title = reference.get("title", "")
        if not title:
            return 0.0

        candidates = self._query_crossref(title) + self._query_semantic_scholar(title)
        if not candidates:
            return 0.0

        best = 0.0
        for c in candidates:
            score = _score_candidate(
                reference, c["title"], c.get("year"), c.get("authors", [])
            )
            if score > best:
                best = score
        return best
