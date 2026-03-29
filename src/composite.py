"""Weighted composite calculator for all AP course archetypes."""

from __future__ import annotations

from .models import CourseConfig, PredictionInput


def compute_weighted_composite(inp: PredictionInput, config: CourseConfig) -> float:
    """Compute a normalized weighted composite in [0, 1].

    Generic loop over config.frq_sections handles all course archetypes
    without course-specific branching.
    """
    # MCQ component: (correct / total) * mcq_weight
    mcq_proportion = inp.mcq_correct / config.mcq_total
    composite = mcq_proportion * config.mcq_weight

    # FRQ components: for each section, for each question,
    # (score / max) * question_weight
    section_by_name = {s.name: s for s in inp.frq_sections}

    for section_config in config.frq_sections:
        section_input = section_by_name[section_config.name]
        for score, max_pts, q_weight in zip(
            section_input.scores,
            section_config.question_max,
            section_config.question_weights,
        ):
            proportion = score / max_pts if max_pts > 0 else 0.0
            composite += proportion * q_weight

    return max(0.0, min(1.0, composite))


def compute_ab_subscore_composite(
    inp: PredictionInput,
    config: CourseConfig,
    ab_mcq_correct: int,
    ab_mcq_total: int,
    ab_frq_indices: list[int],
) -> float:
    """Compute the AB subscore composite for AP Calculus BC.

    This requires knowing which MCQ and FRQ questions map to AB topics.
    Since the tagging is beyond simple total inputs, this takes explicit
    AB-tagged counts.
    """
    section_input = inp.frq_sections[0]  # BC has one FRQ section
    section_config = config.frq_sections[0]

    # AB MCQ component (using same 50/50 weighting as main BC exam)
    mcq_proportion = ab_mcq_correct / ab_mcq_total if ab_mcq_total > 0 else 0.0
    composite = mcq_proportion * config.mcq_weight

    # AB FRQ component: only the tagged questions
    frq_composite = 0.0
    total_ab_max = sum(section_config.question_max[i] for i in ab_frq_indices)
    if total_ab_max > 0:
        for i in ab_frq_indices:
            proportion = section_input.scores[i] / section_config.question_max[i]
            frq_composite += proportion * section_config.question_max[i]
        frq_composite /= total_ab_max
    composite += frq_composite * (1.0 - config.mcq_weight)

    return max(0.0, min(1.0, composite))
