"""Course registry: loads configs, cutoff priors, and data files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .models import (
    CourseConfig,
    CutoffPriors,
    FRQSectionConfig,
    FRQSectionInput,
    PredictionInput,
    ScoreDistribution,
    ScoringStatistics,
)

DATA_DIR = Path(__file__).parent.parent / "data"


class CourseRegistry:
    """Loads and validates all course configurations and associated data."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self.courses: dict[str, CourseConfig] = {}
        self.cutoff_priors: dict[str, CutoffPriors] = {}
        self.score_distributions: dict[str, dict[int, ScoreDistribution]] = {}
        self.scoring_statistics: dict[str, dict[int, ScoringStatistics]] = {}
        self._load_courses()
        self._load_cutoff_priors()
        self._load_score_distributions()
        self._load_scoring_statistics()

    def _load_courses(self) -> None:
        config_path = self.data_dir / "course_config.json"
        with open(config_path) as f:
            raw = json.load(f)
        for key, data in raw.items():
            data["key"] = key
            sections = []
            for s in data["frq_sections"]:
                sections.append(FRQSectionConfig(**s))
            data["frq_sections"] = sections
            self.courses[key] = CourseConfig(**data)

    def _load_cutoff_priors(self) -> None:
        priors_path = self.data_dir / "cutoff_priors.json"
        if not priors_path.exists():
            return
        with open(priors_path) as f:
            raw = json.load(f)
        for key, data in raw.items():
            if key.startswith("_"):
                continue
            self.cutoff_priors[key] = CutoffPriors(**data)

    def _load_score_distributions(self) -> None:
        dist_dir = self.data_dir / "score_distributions"
        if not dist_dir.exists():
            return
        for path in dist_dir.glob("*.json"):
            with open(path) as f:
                raw = json.load(f)
            if raw.get("_description"):
                raw.pop("_description", None)
            dist = ScoreDistribution(**raw)
            self.score_distributions.setdefault(dist.course, {})[dist.year] = dist

    def _load_scoring_statistics(self) -> None:
        stats_dir = self.data_dir / "scoring_statistics"
        if not stats_dir.exists():
            return
        for path in stats_dir.glob("*.json"):
            with open(path) as f:
                raw = json.load(f)
            if raw.get("_description"):
                raw.pop("_description", None)
            stats = ScoringStatistics(**raw)
            self.scoring_statistics.setdefault(stats.course, {})[stats.year] = stats

    def get_config(self, course_key: str) -> CourseConfig:
        if course_key not in self.courses:
            raise ValueError(f"Unknown course: {course_key}. Available: {list(self.courses.keys())}")
        return self.courses[course_key]

    def get_cutoff_priors(self, course_key: str) -> CutoffPriors:
        if course_key not in self.cutoff_priors:
            raise ValueError(f"No cutoff priors for course: {course_key}")
        return self.cutoff_priors[course_key]

    def get_score_distribution(self, course_key: str, year: int) -> Optional[ScoreDistribution]:
        return self.score_distributions.get(course_key, {}).get(year)

    def get_scoring_statistics(self, course_key: str, year: int) -> Optional[ScoringStatistics]:
        return self.scoring_statistics.get(course_key, {}).get(year)

    def list_courses(self) -> list[dict[str, str]]:
        return [{"key": k, "title": v.title} for k, v in sorted(self.courses.items())]

    def validate_input(self, inp: PredictionInput) -> CourseConfig:
        """Validate a prediction input against course config. Returns the config."""
        config = self.get_config(inp.course)

        if inp.mcq_total is not None and inp.mcq_total != config.mcq_total:
            raise ValueError(
                f"mcq_total mismatch: input has {inp.mcq_total}, "
                f"config expects {config.mcq_total}"
            )

        if inp.mcq_correct < 0 or inp.mcq_correct > config.mcq_total:
            raise ValueError(
                f"mcq_correct must be 0..{config.mcq_total}, got {inp.mcq_correct}"
            )

        # Auto-convert flat frq_scores to frq_sections for single-section courses
        if inp.frq_scores is not None and inp.frq_sections is None:
            if len(config.frq_sections) == 1:
                section = config.frq_sections[0]
                if len(inp.frq_scores) != len(section.question_max):
                    raise ValueError(
                        f"frq_scores length {len(inp.frq_scores)} doesn't match "
                        f"expected {len(section.question_max)} questions"
                    )
                inp.frq_sections = [
                    FRQSectionInput(name=section.name, scores=inp.frq_scores)
                ]
            else:
                raise ValueError(
                    f"Course {inp.course} has {len(config.frq_sections)} FRQ sections. "
                    f"Use frq_sections instead of flat frq_scores."
                )

        # Validate each FRQ section
        if inp.frq_sections is not None:
            section_names = {s.name for s in config.frq_sections}
            input_names = {s.name for s in inp.frq_sections}
            if input_names != section_names:
                raise ValueError(
                    f"FRQ section names mismatch. Expected {section_names}, got {input_names}"
                )

            config_by_name = {s.name: s for s in config.frq_sections}
            for section_input in inp.frq_sections:
                section_config = config_by_name[section_input.name]
                if len(section_input.scores) != len(section_config.question_max):
                    raise ValueError(
                        f"Section '{section_input.name}': expected "
                        f"{len(section_config.question_max)} scores, "
                        f"got {len(section_input.scores)}"
                    )
                for i, (score, max_pts) in enumerate(
                    zip(section_input.scores, section_config.question_max)
                ):
                    if score < 0 or score > max_pts:
                        raise ValueError(
                            f"Section '{section_input.name}' Q{i+1}: "
                            f"score {score} out of range 0..{max_pts}"
                        )

        return config
