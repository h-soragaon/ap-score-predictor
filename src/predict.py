"""Main prediction pipeline: validate -> composite -> cutoffs -> predict."""

from __future__ import annotations

import numpy as np

from .composite import compute_weighted_composite
from .course_registry import CourseRegistry
from .difficulty import compute_difficulty_adjustment
from .models import ConfidenceBand, PredictionInput, PredictionOutput, ScoreProbabilities
from .priors import constrain_priors, ordered_logit_probs


DIFFICULTY_LAMBDA = 1.0  # scaling factor for difficulty adjustment


def predict(inp: PredictionInput, registry: CourseRegistry) -> PredictionOutput:
    """Full prediction pipeline for a single student."""
    # 1. Validate input and get config
    config = registry.validate_input(inp)

    # 2. Compute weighted composite
    composite = compute_weighted_composite(inp, config)

    # 3. Get cutoff priors, optionally constrained by score distribution
    priors = registry.get_cutoff_priors(inp.course)
    dist = registry.get_score_distribution(inp.course, inp.exam_year)
    priors = constrain_priors(priors, dist)

    # 4. Compute difficulty adjustment
    stats = registry.get_scoring_statistics(inp.course, inp.exam_year)
    diff_adj = compute_difficulty_adjustment(inp, config, stats)

    # 5. Adjusted composite
    x_adj = float(np.clip(composite + DIFFICULTY_LAMBDA * diff_adj, 0.0, 1.0))

    # 6. Ordered logit probabilities
    probs = ordered_logit_probs(x_adj, priors.tau, priors.sigma)

    # 7. Build probability distribution
    prob_dict = {str(k + 1): round(float(probs[k]), 4) for k in range(5)}

    # 8. Most likely and expected score
    most_likely = int(np.argmax(probs)) + 1
    expected = float(np.sum(probs * np.arange(1, 6)))

    # 9. Confidence band (p10 and p90)
    cum_probs = np.cumsum(probs)
    p10 = int(np.searchsorted(cum_probs, 0.10)) + 1
    p90 = int(np.searchsorted(cum_probs, 0.90)) + 1
    p10 = max(1, min(5, p10))
    p90 = max(1, min(5, p90))

    # 10. Generate explanations
    explanations = _generate_explanations(composite, diff_adj, probs, most_likely, config)

    return PredictionOutput(
        course=inp.course,
        exam_year=inp.exam_year,
        predicted_distribution=prob_dict,
        most_likely_score=most_likely,
        expected_score=round(expected, 2),
        weighted_composite=round(composite, 4),
        difficulty_adjustment=round(diff_adj, 4),
        confidence_band=ConfidenceBand(p10=p10, p90=p90),
        explanations=explanations,
    )


def _generate_explanations(
    composite: float,
    diff_adj: float,
    probs: np.ndarray,
    most_likely: int,
    config,
) -> list[str]:
    """Generate human-readable explanations for the prediction."""
    explanations = []

    # Composite strength
    if composite >= 0.75:
        explanations.append("Strong overall performance across both MCQ and FRQ sections.")
    elif composite >= 0.5:
        explanations.append("Moderate overall performance; composite is near the course median.")
    else:
        explanations.append("Below-average composite score relative to typical performance ranges.")

    # Difficulty adjustment
    if abs(diff_adj) > 0.005:
        if diff_adj > 0:
            explanations.append(
                "FRQ performance was above the national means on available questions, "
                "providing a positive adjustment."
            )
        else:
            explanations.append(
                "FRQ performance was below the national means on available questions, "
                "providing a negative adjustment."
            )
    else:
        explanations.append(
            "No significant difficulty adjustment (no scoring statistics available "
            "or performance was near national averages)."
        )

    # Uncertainty
    max_prob = float(probs.max())
    if max_prob < 0.4:
        explanations.append(
            f"Result is near a score boundary — uncertainty is high. "
            f"The most likely score of {most_likely} has only {max_prob:.0%} probability."
        )
    elif max_prob > 0.7:
        explanations.append(
            f"The prediction is fairly confident: score {most_likely} "
            f"has {max_prob:.0%} probability."
        )

    return explanations
