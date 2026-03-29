"""Tests for the CLI tool."""

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parent.parent


class TestCLICourses:
    def test_list_courses(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "courses"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert "ap_biology" in result.stdout
        assert "ap_calculus_ab" in result.stdout
        # Should have 18 courses
        lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
        assert len(lines) == 18


class TestCLIPredict:
    def test_single_prediction(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "src.cli", "predict",
                "--course", "ap_biology",
                "--mcq", "42",
                "--frq", "7,8,3,4,2,3",
            ],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert "Score Distribution" in result.stdout
        assert "Most Likely" in result.stdout

    def test_multi_section_prediction(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "src.cli", "predict",
                "--course", "ap_us_history",
                "--mcq", "40",
                "--frq", "unused",
                "--section", "saq:3,3,2",
                "--section", "dbq:5",
                "--section", "leq:4",
            ],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert "ap_us_history" in result.stdout


class TestCLIBatch:
    def test_batch_csv(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as inp_file:
            writer = csv.DictWriter(inp_file, fieldnames=[
                "course", "mcq_correct", "frq_1", "frq_2", "frq_3",
                "frq_4", "frq_5", "frq_6",
            ])
            writer.writeheader()
            writer.writerow({
                "course": "ap_biology",
                "mcq_correct": "42",
                "frq_1": "7", "frq_2": "8", "frq_3": "3",
                "frq_4": "4", "frq_5": "2", "frq_6": "3",
            })
            writer.writerow({
                "course": "ap_biology",
                "mcq_correct": "30",
                "frq_1": "5", "frq_2": "5", "frq_3": "2",
                "frq_4": "2", "frq_5": "2", "frq_6": "2",
            })
            inp_path = inp_file.name

        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False
        ) as out_file:
            out_path = out_file.name

        result = subprocess.run(
            [
                sys.executable, "-m", "src.cli", "batch",
                "--input", inp_path,
                "--output", out_path,
            ],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert "Processed 2 students" in result.stdout

        # Verify output CSV
        with open(out_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert "expected_score" in rows[0]
        assert "most_likely_score" in rows[0]
        assert "p_1" in rows[0]

        # Cleanup
        Path(inp_path).unlink()
        Path(out_path).unlink()
