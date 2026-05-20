"""Tests for sigmoid scaling law fitter."""

import math

from confabulation_scaling.sigmoid_fitter import SigmoidScalingLaw


def _make_synthetic_data() -> list[dict]:
    """Generate synthetic data with known scaling behavior."""
    records = []
    true_alpha, true_beta, true_gamma = 2.0, 1.5, -15.0

    for param_b in [0.1, 0.5, 1, 3, 7, 13, 70, 175, 540]:
        param_count = param_b * 1e9
        for freq in [1, 10, 100, 1000, 10000, 100000]:
            x1 = math.log10(param_count)
            x2 = math.log10(freq + 1)
            z = true_alpha * x1 + true_beta * x2 + true_gamma
            p = 1.0 / (1.0 + math.exp(-z))
            records.append(
                {
                    "param_count": param_count,
                    "doc_freq_raw": freq,
                    "recall_score": min(max(p, 0.01), 0.99),
                }
            )
    return records


def test_fit_returns_positive_coefficients():
    records = _make_synthetic_data()
    law = SigmoidScalingLaw()
    params = law.fit(records)
    assert params["alpha"] > 0
    assert params["beta"] > 0


def test_predict_monotone_in_params():
    records = _make_synthetic_data()
    law = SigmoidScalingLaw()
    law.fit(records)
    p_small = law.predict(1e8, 1000)
    p_large = law.predict(1e11, 1000)
    assert p_large > p_small


def test_predict_monotone_in_frequency():
    records = _make_synthetic_data()
    law = SigmoidScalingLaw()
    law.fit(records)
    p_rare = law.predict(7e9, 1)
    p_common = law.predict(7e9, 100000)
    assert p_common > p_rare


def test_predict_with_ci_contains_prediction():
    records = _make_synthetic_data()
    law = SigmoidScalingLaw()
    law.fit(records)
    result = law.predict_with_ci(7e9, 1000)
    assert result["ci_lower"] <= result["prediction"] <= result["ci_upper"]


def test_predict_with_ci_wider_at_extremes():
    records = _make_synthetic_data()
    law = SigmoidScalingLaw()
    law.fit(records)
    mid = law.predict_with_ci(7e9, 1000)
    mid_width = mid["ci_upper"] - mid["ci_lower"]
    assert mid_width >= 0


def test_fit_recovers_approximate_params():
    records = _make_synthetic_data()
    law = SigmoidScalingLaw()
    params = law.fit(records)
    assert abs(params["alpha"] - 2.0) < 0.5
    assert abs(params["beta"] - 1.5) < 0.5
