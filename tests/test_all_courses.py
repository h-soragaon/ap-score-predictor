"""Comprehensive parametrized tests across all 18 AP courses."""

import pytest

from src.composite import compute_weighted_composite
from src.predict import predict
from src.models import FRQSectionInput, PredictionInput

from .conftest import (
    ALL_COURSES,
    make_perfect_input,
    make_zero_input,
    make_mid_input,
    make_input_at,
)


class TestCompositeBounds:
    @pytest.mark.parametrize("course_key", ALL_COURSES)
    def test_perfect_composite_is_one(self, registry, course_key):
        config = registry.get_config(course_key)
        inp = make_perfect_input(config)
        registry.validate_input(inp)
        c = compute_weighted_composite(inp, config)
        assert abs(c - 1.0) < 0.001

    @pytest.mark.parametrize("course_key", ALL_COURSES)
    def test_zero_composite_is_zero(self, registry, course_key):
        config = registry.get_config(course_key)
        inp = make_zero_input(config)
        registry.validate_input(inp)
        c = compute_weighted_composite(inp, config)
        assert abs(c) < 0.001

    @pytest.mark.parametrize("course_key", ALL_COURSES)
    def test_mid_composite_near_half(self, registry, course_key):
        config = registry.get_config(course_key)
        inp = make_mid_input(config)
        registry.validate_input(inp)
        c = compute_weighted_composite(inp, config)
        assert 0.40 < c < 0.60


class TestDistributionValidity:
    @pytest.mark.parametrize("course_key", ALL_COURSES)
    def test_probs_sum_to_one(self, registry, course_key):
        config = registry.get_config(course_key)
        inp = make_mid_input(config)
        result = predict(inp, registry)
        total = sum(result.predicted_distribution.values())
        assert abs(total - 1.0) < 0.01

    @pytest.mark.parametrize("course_key", ALL_COURSES)
    def test_all_probs_non_negative(self, registry, course_key):
        config = registry.get_config(course_key)
        inp = make_mid_input(config)
        result = predict(inp, registry)
        for prob in result.predicted_distribution.values():
            assert prob >= 0

    @pytest.mark.parametrize("course_key", ALL_COURSES)
    def test_most_likely_in_range(self, registry, course_key):
        config = registry.get_config(course_key)
        inp = make_mid_input(config)
        result = predict(inp, registry)
        assert 1 <= result.most_likely_score <= 5

    @pytest.mark.parametrize("course_key", ALL_COURSES)
    def test_expected_score_in_range(self, registry, course_key):
        config = registry.get_config(course_key)
        inp = make_mid_input(config)
        result = predict(inp, registry)
        assert 1.0 <= result.expected_score <= 5.0


class TestMonotonicity:
    @pytest.mark.parametrize("course_key", ALL_COURSES)
    def test_higher_composite_higher_expected(self, registry, course_key):
        """A student scoring higher overall should have a higher expected score."""
        config = registry.get_config(course_key)
        low_inp = make_input_at(config, 0.2)
        high_inp = make_input_at(config, 0.8)
        low_result = predict(low_inp, registry)
        high_result = predict(high_inp, registry)
        assert high_result.expected_score > low_result.expected_score

    @pytest.mark.parametrize("course_key", ALL_COURSES)
    def test_higher_composite_higher_p5(self, registry, course_key):
        """Higher performance should give higher P(score=5)."""
        config = registry.get_config(course_key)
        low_inp = make_input_at(config, 0.2)
        high_inp = make_input_at(config, 0.9)
        low_result = predict(low_inp, registry)
        high_result = predict(high_inp, registry)
        assert float(high_result.predicted_distribution["5"]) >= float(low_result.predicted_distribution["5"])


class TestInputValidationRejection:
    @pytest.mark.parametrize("course_key", ALL_COURSES)
    def test_negative_mcq_rejected(self, registry, course_key):
        config = registry.get_config(course_key)
        sections = [
            FRQSectionInput(name=s.name, scores=[0.0] * len(s.question_max))
            for s in config.frq_sections
        ]
        inp = PredictionInput(
            course=course_key,
            mcq_correct=-1,
            frq_sections=sections,
        )
        with pytest.raises(ValueError):
            registry.validate_input(inp)

    @pytest.mark.parametrize("course_key", ALL_COURSES)
    def test_mcq_over_max_rejected(self, registry, course_key):
        config = registry.get_config(course_key)
        sections = [
            FRQSectionInput(name=s.name, scores=[0.0] * len(s.question_max))
            for s in config.frq_sections
        ]
        inp = PredictionInput(
            course=course_key,
            mcq_correct=config.mcq_total + 1,
            frq_sections=sections,
        )
        with pytest.raises(ValueError):
            registry.validate_input(inp)

    @pytest.mark.parametrize("course_key", ALL_COURSES)
    def test_frq_over_max_rejected(self, registry, course_key):
        config = registry.get_config(course_key)
        sections = []
        for s in config.frq_sections:
            # Set first question to max+1
            scores = [0.0] * len(s.question_max)
            scores[0] = float(s.question_max[0] + 1)
            sections.append(FRQSectionInput(name=s.name, scores=scores))
        inp = PredictionInput(
            course=course_key,
            mcq_correct=0,
            frq_sections=sections,
        )
        with pytest.raises(ValueError):
            registry.validate_input(inp)
