"""Load cutoff priors and optionally constrain against score distributions."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .models import CutoffPriors, ScoreDistribution


def ordered_logit_probs(x: float, tau: list[float], sigma: float) -> np.ndarray:
    """Compute P(score=1..5) using ordered logit model.

    P(Y <= k | x) = logistic((tau_k - x) / sigma)
    """
    cumulative = np.array([expit((t - x) / sigma) for t in tau])
    probs = np.zeros(5)
    probs[0] = cumulative[0]
    for k in range(1, 4):
        probs[k] = cumulative[k] - cumulative[k - 1]
    probs[4] = 1.0 - cumulative[3]
    # Clip tiny negatives from numerical issues
    probs = np.clip(probs, 0.0, 1.0)
    probs /= probs.sum()
    return probs


def expected_distribution(tau: list[float], sigma: float, n_points: int = 200) -> dict[str, float]:
    """Compute the expected score distribution by integrating over uniform composite."""
    x_vals = np.linspace(0, 1, n_points)
    avg_probs = np.zeros(5)
    for x in x_vals:
        avg_probs += ordered_logit_probs(x, tau, sigma)
    avg_probs /= n_points
    return {str(k + 1): float(avg_probs[k]) for k in range(5)}


def constrain_priors(
    priors: CutoffPriors,
    distribution: Optional[ScoreDistribution] = None,
) -> CutoffPriors:
    """Optionally refine cutoff priors to better match an official score distribution.

    Uses L-BFGS-B to minimize KL divergence between the expected distribution
    (assuming uniform composite) and the official distribution.
    """
    if distribution is None:
        return priors

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
        # KL divergence: sum(target * log(target / expected))
        kl = np.sum(target * np.log(target / expected))
        return kl

    x0 = np.array(priors.tau + [priors.sigma])
    bounds = [(0.05, 0.95)] * 4 + [(0.01, 0.3)]
    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)

    new_tau = sorted(result.x[:4].tolist())
    new_sigma = max(result.x[4], 0.01)
    return CutoffPriors(tau=new_tau, sigma=new_sigma)
