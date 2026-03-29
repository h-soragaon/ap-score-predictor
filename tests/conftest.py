"""Shared test fixtures for the AP Score Predictor test suite."""

import pytest

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


@pytest.fixture(scope="session")
def registry():
    return CourseRegistry()


def make_perfect_input(config) -> PredictionInput:
    sections = []
    for s in config.frq_sections:
        sections.append(FRQSectionInput(name=s.name, scores=[float(m) for m in s.question_max]))
    return PredictionInput(
        course=config.key,
        mcq_correct=config.mcq_total,
        frq_sections=sections,
    )


def make_zero_input(config) -> PredictionInput:
    sections = []
    for s in config.frq_sections:
        sections.append(FRQSectionInput(name=s.name, scores=[0.0] * len(s.question_max)))
    return PredictionInput(
        course=config.key,
        mcq_correct=0,
        frq_sections=sections,
    )


def make_mid_input(config) -> PredictionInput:
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


def make_input_at(config, fraction: float) -> PredictionInput:
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
