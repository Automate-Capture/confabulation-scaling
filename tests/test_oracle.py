"""Tests for oracle module (unit tests with mocked HTTP)."""

from unittest.mock import patch, MagicMock

from confabulation_scaling.oracle import ReferenceOracle, _score_candidate


def test_score_candidate_exact_match():
    ref = {"title": "Attention Is All You Need", "year": 2017, "authors": ["Vaswani"]}
    score = _score_candidate(
        ref,
        candidate_title="Attention Is All You Need",
        candidate_year=2017,
        candidate_authors=["Ashish Vaswani"],
    )
    assert score > 0.9


def test_score_candidate_year_off_by_one():
    ref = {"title": "Some Paper", "year": 2020, "authors": ["Smith"]}
    score_exact = _score_candidate(ref, "Some Paper", 2020, ["John Smith"])
    score_off1 = _score_candidate(ref, "Some Paper", 2021, ["John Smith"])
    assert score_exact > score_off1


def test_score_candidate_year_off_by_two():
    ref = {"title": "Some Paper", "year": 2020, "authors": ["Smith"]}
    score = _score_candidate(ref, "Some Paper", 2022, ["John Smith"])
    # year_score should be 0 for diff >= 2
    assert score < 0.9


def test_verify_no_results():
    oracle = ReferenceOracle()
    with patch.object(oracle, "_query_crossref", return_value=[]):
        with patch.object(oracle, "_query_semantic_scholar", return_value=[]):
            score = oracle.verify({"title": "Nonexistent Paper XYZ"})
    assert score == 0.0


def test_verify_with_mocked_candidates():
    oracle = ReferenceOracle()
    candidates = [
        {
            "title": "Test Paper on Neural Networks",
            "year": 2023,
            "authors": ["Alice Johnson"],
        }
    ]
    with patch.object(oracle, "_query_crossref", return_value=candidates):
        with patch.object(oracle, "_query_semantic_scholar", return_value=[]):
            score = oracle.verify(
                {
                    "title": "Test Paper on Neural Networks",
                    "year": 2023,
                    "authors": ["Johnson"],
                }
            )
    assert 0.0 < score <= 1.0
    assert score > 0.85


def test_verify_empty_title():
    oracle = ReferenceOracle()
    assert oracle.verify({"title": ""}) == 0.0
