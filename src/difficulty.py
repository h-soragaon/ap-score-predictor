"""FRQ difficulty adjustment using scoring statistics."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .models import CourseConfig, PredictionInput, ScoringStatistics


def compute_difficulty_adjustment(
    inp: PredictionInput,
    config: CourseConfig,
    stats: Optional[ScoringStatistics],
    shrink_factor: float = 0.7,
) -> float:
    """Compute difficulty adjustment from FRQ scoring statistics.

    For each FRQ question with available statistics:
    - Compute student proportion p_q = score_q / max_q
    - Compute national proportion mu_q / max_q
    - Compute z-score: z_q = (p_q - mu_q/max_q) / (sd_q/max_q)
    - Shrink extreme z values
    - Aggregate by section weight

    Returns 0.0 if no stats are available.
    """
    if stats is None:
        return 0.0

    # Index stats by (section, question_number)
    stats_by_key: dict[tuple[str, int], tuple[float, float, int]] = {}
    for q in stats.questions:
        stats_by_key[(q.section, q.question)] = (q.mean, q.sd, q.max_points)

    section_by_name = {s.name: s for s in inp.frq_sections}
    weighted_z_sum = 0.0
    total_weight = 0.0

    for section_config in config.frq_sections:
        section_input = section_by_name[section_config.name]
        for i, (score, max_pts, q_weight) in enumerate(
            zip(section_input.scores, section_config.question_max, section_config.question_weights)
        ):
            key = (section_config.name, i + 1)
            if key not in stats_by_key:
                continue

            mean, sd, stat_max = stats_by_key[key]
            if sd <= 0 or max_pts <= 0:
                continue

            p_student = score / max_pts
            p_national = mean / stat_max
            sd_proportion = sd / stat_max

            z = (p_student - p_national) / sd_proportion

            # Shrink extremes: tanh-based soft clipping
            z_shrunk = np.tanh(z * shrink_factor) / shrink_factor

            weighted_z_sum += z_shrunk * q_weight
            total_weight += q_weight

    if total_weight == 0:
        return 0.0

    # Normalize by total weight and scale to a small adjustment
    raw_adjustment = weighted_z_sum / total_weight
    # Scale so a 1-sigma national advantage gives ~0.03 composite adjustment
    return float(raw_adjustment * 0.03)
