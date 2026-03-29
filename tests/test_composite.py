"""Tests for weighted composite calculator."""

import pytest

from src.composite import compute_weighted_composite
from src.course_registry import CourseRegistry
from src.models import FRQSectionInput, PredictionInput

ALL_COURSES = [
    "ap_biology",
    "ap_calculus_ab",
    "ap_calculus_bc",
    "ap_chemistry",
    "ap_computer_science_a",
    "ap_english_language_and_composition",
    "ap_english_literature_and_composition",
    "ap_environmental_science",
    "ap_human_geography",
    "ap_microeconomics",
    "ap_music_theory",
    "ap_physics_1",
    "ap_physics_2",
    "ap_psychology",
    "ap_statistics",
    "ap_us_government_and_politics",
    "ap_us_history",
    "ap_world_history_modern",
]


@pytest.fixture
def registry():
    return CourseRegistry()


def make_perfect_input(config) -> PredictionInput:
    """Create a perfect-score input for any course."""
    sections = []
    for s in config.frq_sections:
        sections.append(FRQSectionInput(name=s.name, scores=[float(m) for m in s.question_max]))
    return PredictionInput(
        course=config.key,
        mcq_correct=config.mcq_total,
        frq_sections=sections,
    )


def make_zero_input(config) -> PredictionInput:
    """Create a zero-score input for any course."""
    sections = []
    for s in config.frq_sections:
        sections.append(FRQSectionInput(name=s.name, scores=[0.0] * len(s.question_max)))
    return PredictionInput(
        course=config.key,
        mcq_correct=0,
        frq_sections=sections,
    )


def make_half_input(config) -> PredictionInput:
    """Create a 50% input for any course."""
    sections = []
    for s in config.frq_sections:
        sections.append(
            FRQSectionInput(name=s.name, scores=[m / 2.0 for m in s.question_max])
        )
    return PredictionInput(
        course=config.key,
        mcq_correct=config.mcq_total // 2,
        frq_sections=sections,
    )


class TestPerfectScore:
    @pytest.mark.parametrize("course_key", ALL_COURSES)
    def test_perfect_equals_one(self, registry, course_key):
        config = registry.get_config(course_key)
        inp = make_perfect_input(config)
        registry.validate_input(inp)
        composite = compute_weighted_composite(inp, config)
        assert abs(composite - 1.0) < 0.001, f"{course_key}: perfect={composite}"


class TestZeroScore:
    @pytest.mark.parametrize("course_key", ALL_COURSES)
    def test_zero_equals_zero(self, registry, course_key):
        config = registry.get_config(course_key)
        inp = make_zero_input(config)
        registry.validate_input(inp)
        composite = compute_weighted_composite(inp, config)
        assert abs(composite) < 0.001, f"{course_key}: zero={composite}"


class TestHalfScore:
    @pytest.mark.parametrize("course_key", ALL_COURSES)
    def test_half_near_half(self, registry, course_key):
        config = registry.get_config(course_key)
        inp = make_half_input(config)
        registry.validate_input(inp)
        composite = compute_weighted_composite(inp, config)
        # Half scores should give approximately 0.5
        # (not exact due to integer division on mcq_total)
        assert 0.45 < composite < 0.55, f"{course_key}: half={composite}"


class TestBiologyManualCalc:
    def test_biology_example(self, registry):
        """Manual verification: AP Bio with specific scores.
        MCQ: 42/60 = 0.7 * 0.5 = 0.35
        FRQ: [7,8,3,4,2,3] with max [9,9,4,4,4,4]
          Q1: 7/9 * 0.1324 = 0.1030
          Q2: 8/9 * 0.1324 = 0.1177
          Q3: 3/4 * 0.0588 = 0.0441
          Q4: 4/4 * 0.0588 = 0.0588
          Q5: 2/4 * 0.0588 = 0.0294
          Q6: 3/4 * 0.0588 = 0.0441
        FRQ total = 0.3971
        Total = 0.35 + 0.3971 = 0.7471
        """
        config = registry.get_config("ap_biology")
        inp = PredictionInput(
            course="ap_biology",
            mcq_correct=42,
            frq_scores=[7, 8, 3, 4, 2, 3],
        )
        registry.validate_input(inp)
        composite = compute_weighted_composite(inp, config)
        assert 0.74 < composite < 0.76, f"Biology manual calc: {composite}"


class TestMultiSectionCourses:
    def test_apush_composite(self, registry):
        config = registry.get_config("ap_us_history")
        inp = PredictionInput(
            course="ap_us_history",
            mcq_correct=40,
            frq_sections=[
                FRQSectionInput(name="saq", scores=[3, 3, 2]),
                FRQSectionInput(name="dbq", scores=[5]),
                FRQSectionInput(name="leq", scores=[4]),
            ],
        )
        registry.validate_input(inp)
        composite = compute_weighted_composite(inp, config)
        assert 0.0 < composite < 1.0

    def test_music_theory_composite(self, registry):
        config = registry.get_config("ap_music_theory")
        inp = PredictionInput(
            course="ap_music_theory",
            mcq_correct=60,
            frq_sections=[
                FRQSectionInput(
                    name="written_frq",
                    scores=[7, 7, 18, 18, 20, 14, 7],
                ),
                FRQSectionInput(name="sight_singing", scores=[7, 7]),
            ],
        )
        registry.validate_input(inp)
        composite = compute_weighted_composite(inp, config)
        assert 0.0 < composite < 1.0

    def test_world_history_composite(self, registry):
        config = registry.get_config("ap_world_history_modern")
        inp = PredictionInput(
            course="ap_world_history_modern",
            mcq_correct=35,
            frq_sections=[
                FRQSectionInput(name="saq", scores=[2, 2, 1]),
                FRQSectionInput(name="dbq", scores=[4]),
                FRQSectionInput(name="leq", scores=[3]),
            ],
        )
        registry.validate_input(inp)
        composite = compute_weighted_composite(inp, config)
        assert 0.0 < composite < 1.0
