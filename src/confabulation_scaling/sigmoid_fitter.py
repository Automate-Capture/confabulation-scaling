"""Scaling law fitter: calibrated sigmoid over param count and topic frequency."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit


def _sigmoid(X: np.ndarray, alpha: float, beta: float, gamma: float) -> np.ndarray:
    z = alpha * X[0] + beta * X[1] + gamma
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class SigmoidScalingLaw:
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    covariance: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))

    def fit(self, records: list[dict]) -> dict:
        x1 = np.array([math.log10(r["param_count"]) for r in records])
        x2 = np.array([math.log10(r["doc_freq_raw"] + 1) for r in records])
        y = np.array([r["recall_score"] for r in records])

        X = np.vstack([x1, x2])

        popt, pcov = curve_fit(
            _sigmoid,
            X,
            y,
            p0=[1.0, 1.0, -5.0],
            bounds=([0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=10000,
        )

        self.alpha, self.beta, self.gamma = popt
        self.covariance = pcov

        return {
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "gamma": float(self.gamma),
        }

    def predict(self, param_count: float, doc_freq_raw: float) -> float:
        x1 = math.log10(param_count)
        x2 = math.log10(doc_freq_raw + 1)
        z = self.alpha * x1 + self.beta * x2 + self.gamma
        return 1.0 / (1.0 + math.exp(-z))

    def predict_with_ci(
        self, param_count: float, doc_freq_raw: float, z_score: float = 1.96
    ) -> dict:
        x1 = math.log10(param_count)
        x2 = math.log10(doc_freq_raw + 1)
        grad = np.array([x1, x2, 1.0])
        z = self.alpha * x1 + self.beta * x2 + self.gamma
        p = 1.0 / (1.0 + math.exp(-z))

        var_z = grad @ self.covariance @ grad
        se_z = math.sqrt(max(var_z, 0.0))
        z_lo = z - z_score * se_z
        z_hi = z + z_score * se_z

        p_lo = 1.0 / (1.0 + math.exp(-z_lo))
        p_hi = 1.0 / (1.0 + math.exp(-z_hi))

        return {"prediction": p, "ci_lower": p_lo, "ci_upper": p_hi}
