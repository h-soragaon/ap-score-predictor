"""Fit cutoff parameters to match an official score distribution."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .models import CutoffPriors, ScoreDistribution
from .priors import ordered_logit_probs


def fit_cutoffs_to_distribution(
    distribution: ScoreDistribution,
    initial_sigma: float = 0.08,
) -> CutoffPriors:
    """Find tau and sigma values that best reproduce an official score distribution.

    Assumes a uniform distribution of student composites in [0,1] and finds
    the ordered logit parameters that minimize squared error against the
    official score percentages.
    """
    target = np.array([distribution.distribution[str(k)] for k in range(1, 6)])

    def objective(params):
        tau = sorted(params[:4].tolist())
        sigma = max(params[4], 0.01)
        expected = np.zeros(5)
        x_vals = np.linspace(0, 1, 200)
        for x in x_vals:
            expected += ordered_logit_probs(x, tau, sigma)
        expected /= 200
        expected = np.clip(expected, 1e-10, 1.0)
        expected /= expected.sum()
        return float(np.sum((target - expected) ** 2))

    # Start with evenly spaced taus
    x0 = np.array([0.2, 0.4, 0.6, 0.8, initial_sigma])
    bounds = [(0.05, 0.95)] * 4 + [(0.01, 0.3)]

    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)

    new_tau = sorted(result.x[:4].tolist())
    new_sigma = max(result.x[4], 0.01)

    return CutoffPriors(tau=new_tau, sigma=new_sigma)
