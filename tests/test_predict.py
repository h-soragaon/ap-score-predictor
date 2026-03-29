"""Tests for the prediction engine."""

import pytest
import numpy as np

from src.course_registry import CourseRegistry
from src.models import (
    CutoffPriors,
    FRQSectionInput,
    PredictionInput,
    ScoreDistribution,
    ScoringStatistics,
    ScoringStatisticsQuestion,
)
from src.predict import predict
from src.priors import ordered_logit_probs
from src.fit_cutoffs import fit_cutoffs_to_distribution
from src.difficulty import compute_difficulty_adjustment


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


def make_input(config, fraction: float) -> PredictionInput:
    """Create input at a given fraction of max score."""
    sections = []
    for s in config.frq_sections:
        scores = [round(m * fraction) for m in s.question_max]
        sections.append(FRQSectionInput(name=s.name, scores=[float(x) for x in scores]))
    mcq = round(config.mcq_total * fraction)
    return PredictionInput(
        course=config.key,
        mcq_correct=mcq,
        frq_sections=sections,
    )


class TestMathCorrectness:
    def test_probabilities_sum_to_one(self):
        tau = [0.25, 0.42, 0.60, 0.78]
        sigma = 0.08
        for x in np.linspace(0, 1, 20):
            probs = ordered_logit_probs(x, tau, sigma)
            assert abs(probs.sum() - 1.0) < 1e-6

    def test_probabilities_non_negative(self):
        tau = [0.25, 0.42, 0.60, 0.78]
        sigma = 0.08
        for x in np.linspace(0, 1, 20):
            probs = ordered_logit_probs(x, tau, sigma)
            assert np.all(probs >= 0)

    def test_monotonicity_score5(self):
        """Higher composite should give higher P(score=5)."""
        tau = [0.25, 0.42, 0.60, 0.78]
        sigma = 0.08
        prev_p5 = 0.0
        for x in np.linspace(0.3, 1.0, 15):
            probs = ordered_logit_probs(x, tau, sigma)
            assert probs[4] >= prev_p5 - 1e-6  # allow tiny numerical noise
            prev_p5 = probs[4]

    def test_monotonicity_score1(self):
        """Higher composite should give lower P(score=1)."""
        tau = [0.25, 0.42, 0.60, 0.78]
        sigma = 0.08
        prev_p1 = 1.0
        for x in np.linspace(0.0, 0.7, 15):
            probs = ordered_logit_probs(x, tau, sigma)
            assert probs[0] <= prev_p1 + 1e-6
            prev_p1 = probs[0]

    def test_symmetry_at_tau(self):
        """At tau_k, P(Y<=k) should be ~0.5."""
        tau = [0.25, 0.42, 0.60, 0.78]
        sigma = 0.08
        # At x=tau[0], P(Y<=1) = logistic(0) = 0.5
        probs = ordered_logit_probs(tau[0], tau, sigma)
        assert abs(probs[0] - 0.5) < 0.01


class TestEndToEndPredictions:
    @pytest.mark.parametrize("course_key", ALL_COURSES)
    def test_low_score_prediction(self, registry, course_key):
        config = registry.get_config(course_key)
        inp = make_input(config, 0.15)
        result = predict(inp, registry)
        assert sum(result.predicted_distribution.values()) > 0.99
        assert result.most_likely_score in [1, 2]

    @pytest.mark.parametrize("course_key", ALL_COURSES)
    def test_mid_score_prediction(self, registry, course_key):
        config = registry.get_config(course_key)
        inp = make_input(config, 0.55)
        result = predict(inp, registry)
        assert sum(result.predicted_distribution.values()) > 0.99
        assert result.most_likely_score in [2, 3, 4]

    @pytest.mark.parametrize("course_key", ALL_COURSES)
    def test_high_score_prediction(self, registry, course_key):
        config = registry.get_config(course_key)
        inp = make_input(config, 0.90)
        result = predict(inp, registry)
        assert sum(result.predicted_distribution.values()) > 0.99
        assert result.most_likely_score in [4, 5]

    @pytest.mark.parametrize("course_key", ALL_COURSES)
    def test_prediction_has_explanations(self, registry, course_key):
        config = registry.get_config(course_key)
        inp = make_input(config, 0.50)
        result = predict(inp, registry)
        assert len(result.explanations) >= 2

    @pytest.mark.parametrize("course_key", ALL_COURSES)
    def test_confidence_band_valid(self, registry, course_key):
        config = registry.get_config(course_key)
        inp = make_input(config, 0.50)
        result = predict(inp, registry)
        assert 1 <= result.confidence_band.p10 <= 5
        assert 1 <= result.confidence_band.p90 <= 5
        assert result.confidence_band.p10 <= result.confidence_band.p90


class TestDifficultyAdjustment:
    def test_no_stats_returns_zero(self, registry):
        config = registry.get_config("ap_calculus_ab")
        inp = PredictionInput(
            course="ap_calculus_ab",
            mcq_correct=30,
            frq_scores=[5, 5, 5, 5, 5, 5],
        )
        registry.validate_input(inp)
        adj = compute_difficulty_adjustment(inp, config, None)
        assert adj == 0.0

    def test_above_average_positive_adjustment(self, registry):
        config = registry.get_config("ap_biology")
        inp = PredictionInput(
            course="ap_biology",
            mcq_correct=42,
            frq_scores=[9, 9, 4, 4, 4, 4],  # perfect FRQ
        )
        registry.validate_input(inp)
        stats = ScoringStatistics(
            course="ap_biology",
            year=2026,
            questions=[
                ScoringStatisticsQuestion(question=i + 1, section="frq", max_points=m, mean=m * 0.4, sd=m * 0.2)
                for i, m in enumerate([9, 9, 4, 4, 4, 4])
            ],
        )
        adj = compute_difficulty_adjustment(inp, config, stats)
        assert adj > 0

    def test_below_average_negative_adjustment(self, registry):
        config = registry.get_config("ap_biology")
        inp = PredictionInput(
            course="ap_biology",
            mcq_correct=42,
            frq_scores=[1, 1, 0, 0, 0, 0],  # very low FRQ
        )
        registry.validate_input(inp)
        stats = ScoringStatistics(
            course="ap_biology",
            year=2026,
            questions=[
                ScoringStatisticsQuestion(question=i + 1, section="frq", max_points=m, mean=m * 0.5, sd=m * 0.2)
                for i, m in enumerate([9, 9, 4, 4, 4, 4])
            ],
        )
        adj = compute_difficulty_adjustment(inp, config, stats)
        assert adj < 0


class TestCutoffFitting:
    def test_round_trip(self):
        """Fit cutoffs to a distribution, then verify they reproduce it."""
        dist = ScoreDistribution(
            course="test",
            year=2024,
            distribution={"1": 0.10, "2": 0.20, "3": 0.30, "4": 0.25, "5": 0.15},
        )
        fitted = fit_cutoffs_to_distribution(dist)
        assert len(fitted.tau) == 4
        assert fitted.sigma > 0
        # Verify monotonicity
        for i in range(3):
            assert fitted.tau[i] < fitted.tau[i + 1]

    def test_fitted_reproduces_distribution(self):
        """Fitted cutoffs should approximately reproduce the target distribution."""
        dist = ScoreDistribution(
            course="test",
            year=2024,
            distribution={"1": 0.10, "2": 0.20, "3": 0.30, "4": 0.25, "5": 0.15},
        )
        fitted = fit_cutoffs_to_distribution(dist)

        # Compute expected distribution
        expected = np.zeros(5)
        x_vals = np.linspace(0, 1, 200)
        for x in x_vals:
            expected += ordered_logit_probs(x, fitted.tau, fitted.sigma)
        expected /= 200

        target = np.array([0.10, 0.20, 0.30, 0.25, 0.15])
        # Should match within a few percent
        assert np.all(np.abs(expected - target) < 0.05)


class TestSpecialCases:
    def test_calculus_bc(self, registry):
        config = registry.get_config("ap_calculus_bc")
        inp = PredictionInput(
            course="ap_calculus_bc",
            mcq_correct=35,
            frq_scores=[7, 7, 7, 7, 7, 7],
        )
        result = predict(inp, registry)
        assert result.course == "ap_calculus_bc"
        assert sum(result.predicted_distribution.values()) > 0.99

    def test_music_theory(self, registry):
        config = registry.get_config("ap_music_theory")
        inp = PredictionInput(
            course="ap_music_theory",
            mcq_correct=60,
            frq_sections=[
                FRQSectionInput(name="written_frq", scores=[7, 7, 18, 18, 20, 14, 7]),
                FRQSectionInput(name="sight_singing", scores=[7, 7]),
            ],
        )
        result = predict(inp, registry)
        assert result.course == "ap_music_theory"
        assert sum(result.predicted_distribution.values()) > 0.99

    def test_apush(self, registry):
        inp = PredictionInput(
            course="ap_us_history",
            mcq_correct=40,
            frq_sections=[
                FRQSectionInput(name="saq", scores=[3, 3, 2]),
                FRQSectionInput(name="dbq", scores=[5]),
                FRQSectionInput(name="leq", scores=[4]),
            ],
        )
        result = predict(inp, registry)
        assert result.course == "ap_us_history"
        assert sum(result.predicted_distribution.values()) > 0.99

    def test_world_history(self, registry):
        inp = PredictionInput(
            course="ap_world_history_modern",
            mcq_correct=35,
            frq_sections=[
                FRQSectionInput(name="saq", scores=[2, 2, 1]),
                FRQSectionInput(name="dbq", scores=[4]),
                FRQSectionInput(name="leq", scores=[3]),
            ],
        )
        result = predict(inp, registry)
        assert result.course == "ap_world_history_modern"

    def test_microeconomics(self, registry):
        inp = PredictionInput(
            course="ap_microeconomics",
            mcq_correct=45,
            frq_scores=[8, 4, 4],
        )
        result = predict(inp, registry)
        assert result.course == "ap_microeconomics"
        assert sum(result.predicted_distribution.values()) > 0.99
