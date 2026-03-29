"""CLI tool for AP Score Predictor."""

from __future__ import annotations

import argparse
import csv
import json
import sys

from .course_registry import CourseRegistry
from .models import FRQSectionInput, PredictionInput
from .predict import predict


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ap-predict",
        description="AP Score Probability Predictor",
    )
    sub = parser.add_subparsers(dest="command")

    # courses command
    sub.add_parser("courses", help="List all supported courses")

    # predict command
    pred = sub.add_parser("predict", help="Predict score for a single student")
    pred.add_argument("--course", required=True, help="Course key (e.g. ap_biology)")
    pred.add_argument("--mcq", type=int, required=True, help="MCQ correct count")
    pred.add_argument("--frq", type=str, required=True, help="Comma-separated FRQ scores (e.g. '7,8,3,4,2,3')")
    pred.add_argument("--year", type=int, default=2026, help="Exam year")
    pred.add_argument("--section", type=str, action="append", default=None,
                       help="Section scores: 'name:s1,s2,...' (for multi-section courses)")

    # batch command
    batch = sub.add_parser("batch", help="Batch predict from CSV")
    batch.add_argument("--input", required=True, help="Input CSV path")
    batch.add_argument("--output", required=True, help="Output CSV path")

    return parser


def cmd_courses(registry: CourseRegistry) -> None:
    for c in registry.list_courses():
        print(f"  {c['key']:45s} {c['title']}")


def cmd_predict(args, registry: CourseRegistry) -> None:
    frq_sections = None
    frq_scores = None

    if args.section:
        # Multi-section input: --section saq:3,3,2 --section dbq:5 --section leq:4
        frq_sections = []
        for s in args.section:
            name, scores_str = s.split(":", 1)
            scores = [float(x) for x in scores_str.split(",")]
            frq_sections.append(FRQSectionInput(name=name, scores=scores))
    else:
        frq_scores = [float(x) for x in args.frq.split(",")]

    inp = PredictionInput(
        course=args.course,
        exam_year=args.year,
        mcq_correct=args.mcq,
        frq_scores=frq_scores,
        frq_sections=frq_sections,
    )

    result = predict(inp, registry)

    print(f"\n  Course:       {result.course}")
    print(f"  Exam Year:    {result.exam_year}")
    print(f"  Composite:    {result.weighted_composite:.4f}")
    print(f"  Difficulty:   {result.difficulty_adjustment:+.4f}")
    print(f"  Expected:     {result.expected_score:.2f}")
    print(f"  Most Likely:  {result.most_likely_score}")
    print(f"  Confidence:   {result.confidence_band.p10}-{result.confidence_band.p90}")
    print()
    print("  Score Distribution:")
    for score, prob in sorted(result.predicted_distribution.items()):
        bar = "#" * int(prob * 50)
        print(f"    Score {score}: {prob:6.1%}  {bar}")
    print()
    for expl in result.explanations:
        print(f"  - {expl}")
    print()


def cmd_batch(args, registry: CourseRegistry) -> None:
    with open(args.input, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    results = []
    for i, row in enumerate(rows):
        course = row["course"]
        config = registry.get_config(course)

        mcq_correct = int(row["mcq_correct"])

        # Build FRQ sections from CSV columns
        frq_sections = []
        for section in config.frq_sections:
            scores = []
            for q_idx in range(len(section.question_max)):
                col = f"{section.name}_{q_idx + 1}"
                if col in row:
                    scores.append(float(row[col]))
                else:
                    # Fallback: frq_1, frq_2, ... for single-section
                    alt_col = f"frq_{q_idx + 1}"
                    if alt_col in row:
                        scores.append(float(row[alt_col]))
                    else:
                        raise ValueError(f"Row {i}: missing column {col} or {alt_col}")
            frq_sections.append(FRQSectionInput(name=section.name, scores=scores))

        year = int(row.get("exam_year", 2026))
        inp = PredictionInput(
            course=course,
            exam_year=year,
            mcq_correct=mcq_correct,
            frq_sections=frq_sections,
        )
        result = predict(inp, registry)
        results.append((row, result))

    with open(args.output, "w", newline="") as f:
        fieldnames = list(rows[0].keys()) + [
            "weighted_composite",
            "difficulty_adjustment",
            "expected_score",
            "most_likely_score",
            "p_1", "p_2", "p_3", "p_4", "p_5",
            "confidence_p10", "confidence_p90",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for orig_row, result in results:
            out = dict(orig_row)
            out["weighted_composite"] = f"{result.weighted_composite:.4f}"
            out["difficulty_adjustment"] = f"{result.difficulty_adjustment:.4f}"
            out["expected_score"] = f"{result.expected_score:.2f}"
            out["most_likely_score"] = result.most_likely_score
            for k in range(1, 6):
                out[f"p_{k}"] = f"{result.predicted_distribution[str(k)]:.4f}"
            out["confidence_p10"] = result.confidence_band.p10
            out["confidence_p90"] = result.confidence_band.p90
            writer.writerow(out)

    print(f"Processed {len(results)} students -> {args.output}")


def main(argv=None) -> None:
    parser = make_parser()
    args = parser.parse_args(argv)
    registry = CourseRegistry()

    if args.command == "courses":
        cmd_courses(registry)
    elif args.command == "predict":
        cmd_predict(args, registry)
    elif args.command == "batch":
        cmd_batch(args, registry)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
