"""Tests for course registry and data models."""

import pytest

from src.course_registry import CourseRegistry
from src.models import FRQSectionInput, PredictionInput


@pytest.fixture
def registry():
    return CourseRegistry()


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


class TestCourseLoading:
    def test_all_18_courses_loaded(self, registry):
        assert len(registry.courses) == 18

    @pytest.mark.parametrize("course_key", ALL_COURSES)
    def test_course_exists(self, registry, course_key):
        config = registry.get_config(course_key)
        assert config.key == course_key

    @pytest.mark.parametrize("course_key", ALL_COURSES)
    def test_weights_sum_to_one(self, registry, course_key):
        config = registry.get_config(course_key)
        frq_weight = sum(s.weight for s in config.frq_sections)
        total = config.mcq_weight + frq_weight
        assert abs(total - 1.0) < 0.01, f"{course_key}: weights sum to {total}"

    @pytest.mark.parametrize("course_key", ALL_COURSES)
    def test_question_weights_sum_to_section_weight(self, registry, course_key):
        config = registry.get_config(course_key)
        for section in config.frq_sections:
            qw_sum = sum(section.question_weights)
            assert abs(qw_sum - section.weight) < 0.01, (
                f"{course_key}/{section.name}: question weights sum to "
                f"{qw_sum}, expected {section.weight}"
            )


class TestCutoffPriors:
    @pytest.mark.parametrize("course_key", ALL_COURSES)
    def test_cutoff_priors_exist(self, registry, course_key):
        priors = registry.get_cutoff_priors(course_key)
        assert len(priors.tau) == 4
        assert priors.sigma > 0

    @pytest.mark.parametrize("course_key", ALL_COURSES)
    def test_tau_monotonic(self, registry, course_key):
        priors = registry.get_cutoff_priors(course_key)
        for i in range(3):
            assert priors.tau[i] < priors.tau[i + 1]


class TestInputValidation:
    def test_valid_biology_input(self, registry):
        inp = PredictionInput(
            course="ap_biology",
            mcq_correct=42,
            frq_scores=[7, 8, 3, 4, 2, 3],
        )
        config = registry.validate_input(inp)
        assert config.key == "ap_biology"
        assert inp.frq_sections is not None

    def test_invalid_course_rejected(self, registry):
        inp = PredictionInput(
            course="ap_fake_course",
            mcq_correct=30,
            frq_scores=[5],
        )
        with pytest.raises(ValueError, match="Unknown course"):
            registry.validate_input(inp)

    def test_mcq_out_of_range_rejected(self, registry):
        inp = PredictionInput(
            course="ap_biology",
            mcq_correct=999,
            frq_scores=[7, 8, 3, 4, 2, 3],
        )
        with pytest.raises(ValueError, match="mcq_correct"):
            registry.validate_input(inp)

    def test_wrong_frq_count_rejected(self, registry):
        inp = PredictionInput(
            course="ap_biology",
            mcq_correct=42,
            frq_scores=[7, 8, 3],  # only 3 instead of 6
        )
        with pytest.raises(ValueError, match="frq_scores length"):
            registry.validate_input(inp)

    def test_frq_score_over_max_rejected(self, registry):
        inp = PredictionInput(
            course="ap_biology",
            mcq_correct=42,
            frq_scores=[10, 8, 3, 4, 2, 3],  # Q1 max is 9
        )
        with pytest.raises(ValueError, match="out of range"):
            registry.validate_input(inp)

    def test_multi_section_requires_frq_sections(self, registry):
        inp = PredictionInput(
            course="ap_us_history",
            mcq_correct=40,
            frq_scores=[3, 3, 3, 5, 4],  # flat scores for multi-section
        )
        with pytest.raises(ValueError, match="frq_sections"):
            registry.validate_input(inp)

    def test_multi_section_valid_input(self, registry):
        inp = PredictionInput(
            course="ap_us_history",
            mcq_correct=40,
            frq_sections=[
                FRQSectionInput(name="saq", scores=[3, 3, 2]),
                FRQSectionInput(name="dbq", scores=[5]),
                FRQSectionInput(name="leq", scores=[4]),
            ],
        )
        config = registry.validate_input(inp)
        assert config.key == "ap_us_history"

    def test_music_theory_multi_section(self, registry):
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
        config = registry.validate_input(inp)
        assert config.key == "ap_music_theory"
