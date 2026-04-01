"""Pydantic data models for the AP Score Predictor."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class FRQSectionConfig(BaseModel):
    """Configuration for one FRQ section of a course."""

    name: str
    weight: float
    question_max: list[int]
    question_weights: list[float]

    @field_validator("question_weights")
    @classmethod
    def weights_match_questions(cls, v: list[float], info) -> list[float]:
        if "question_max" in info.data and len(v) != len(info.data["question_max"]):
            raise ValueError("question_weights length must match question_max length")
        return v


class CourseConfig(BaseModel):
    """Full configuration for one AP course."""

    key: str = ""
    title: str
    mcq_total: int
    mcq_weight: float
    frq_sections: list[FRQSectionConfig]
    special: Optional[str] = None
    mcq_point_weight: float
    total_weight: float

    @field_validator("total_weight")
    @classmethod
    def total_weight_approx_one(cls, v: float) -> float:
        if abs(v - 1.0) > 0.01:
            raise ValueError(f"total_weight must be ~1.0, got {v}")
        return v


class FRQSectionInput(BaseModel):
    """Scores for one named FRQ section."""

    name: str
    scores: list[float]


class PredictionInput(BaseModel):
    """Input for a single prediction request."""

    course: str
    exam_year: int = 2025
    mcq_correct: int
    mcq_total: Optional[int] = None
    frq_scores: Optional[list[float]] = None
    frq_sections: Optional[list[FRQSectionInput]] = None

    @model_validator(mode="after")
    def validate_frq_input(self) -> "PredictionInput":
        if self.frq_scores is None and self.frq_sections is None:
            raise ValueError("Must provide either frq_scores or frq_sections")
        return self


class ScoreProbabilities(BaseModel):
    """Probability distribution across AP scores 1-5."""

    score_1: float = Field(ge=0, le=1)
    score_2: float = Field(ge=0, le=1)
    score_3: float = Field(ge=0, le=1)
    score_4: float = Field(ge=0, le=1)
    score_5: float = Field(ge=0, le=1)

    def as_dict(self) -> dict[str, float]:
        return {
            "1": self.score_1,
            "2": self.score_2,
            "3": self.score_3,
            "4": self.score_4,
            "5": self.score_5,
        }

    @model_validator(mode="after")
    def probs_sum_to_one(self) -> "ScoreProbabilities":
        total = self.score_1 + self.score_2 + self.score_3 + self.score_4 + self.score_5
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Probabilities must sum to ~1.0, got {total}")
        return self


class ConfidenceBand(BaseModel):
    """Percentile-based confidence band."""

    p10: int = Field(ge=1, le=5)
    p90: int = Field(ge=1, le=5)


class PredictionOutput(BaseModel):
    """Full output from a prediction."""

    course: str
    exam_year: int
    predicted_distribution: dict[str, float]
    most_likely_score: int = Field(ge=1, le=5)
    expected_score: float
    weighted_composite: float
    difficulty_adjustment: float
    confidence_band: ConfidenceBand
    explanations: list[str]


class CutoffPriors(BaseModel):
    """Cutoff priors for one course's ordered logit model."""

    tau: list[float] = Field(min_length=4, max_length=4)
    sigma: float = Field(gt=0)

    @field_validator("tau")
    @classmethod
    def tau_monotonic(cls, v: list[float]) -> list[float]:
        for i in range(len(v) - 1):
            if v[i] >= v[i + 1]:
                raise ValueError(f"tau must be strictly increasing: {v}")
        return v


class ScoringStatisticsQuestion(BaseModel):
    """Scoring statistics for one FRQ question."""

    question: int
    section: str
    max_points: int
    mean: float
    sd: float


class ScoringStatistics(BaseModel):
    """Scoring statistics for a course/year."""

    course: str
    year: int
    questions: list[ScoringStatisticsQuestion]


class ScoreDistribution(BaseModel):
    """Official score distribution for a course/year."""

    course: str
    year: int
    total_students: Optional[int] = None
    distribution: dict[str, float]

    @field_validator("distribution")
    @classmethod
    def valid_distribution(cls, v: dict[str, float]) -> dict[str, float]:
        if set(v.keys()) != {"1", "2", "3", "4", "5"}:
            raise ValueError("Distribution must have keys '1' through '5'")
        total = sum(v.values())
        if abs(total - 1.0) > 0.02:
            raise ValueError(f"Distribution must sum to ~1.0, got {total}")
        return v
