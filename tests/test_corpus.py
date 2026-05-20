"""Tests for corpus frequency estimator."""

import json
import math
import tempfile
from pathlib import Path

from confabulation_scaling.index_builder import IndexBuilder
from confabulation_scaling.corpus import CorpusFrequencyEstimator


def _build_test_index(tmp_path: Path) -> Path:
    abstracts = [
        {"title": "Deep learning", "abstract": "Neural networks for vision."},
        {"title": "Reinforcement learning", "abstract": "Policy gradient methods."},
        {"title": "Deep reinforcement learning", "abstract": "Deep RL with neural nets."},
    ]
    wiki = [
        {"title": "Artificial intelligence", "text": "AI is intelligence demonstrated by machines."},
        {"title": "Computer science", "text": "The study of computation and information."},
    ]
    abs_path = tmp_path / "abs.json"
    abs_path.write_text(json.dumps(abstracts))
    wiki_path = tmp_path / "wiki.json"
    wiki_path.write_text(json.dumps(wiki))

    builder = IndexBuilder()
    index = builder.build(abs_path, wiki_path)
    idx_path = tmp_path / "index.json.gz"
    builder.save(index, idx_path)
    return idx_path


def test_estimate_returns_float(tmp_path):
    idx_path = _build_test_index(tmp_path)
    estimator = CorpusFrequencyEstimator(index_path=idx_path)
    result = estimator.estimate("deep learning")
    assert isinstance(result, float)
    assert result > 0


def test_estimate_unknown_topic_returns_log1(tmp_path):
    idx_path = _build_test_index(tmp_path)
    estimator = CorpusFrequencyEstimator(index_path=idx_path)
    result = estimator.estimate("xyzzy_nonexistent_topic_42")
    assert result == math.log10(0 + 1)  # log10(1) == 0


def test_higher_frequency_higher_score(tmp_path):
    idx_path = _build_test_index(tmp_path)
    estimator = CorpusFrequencyEstimator(index_path=idx_path)
    # "learning" should appear in more docs than "computation"
    score_common = estimator.estimate("learning")
    score_rare = estimator.estimate("computation")
    assert score_common >= score_rare
