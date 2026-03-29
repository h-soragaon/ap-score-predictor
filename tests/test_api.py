"""Tests for the FastAPI backend."""

import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture
def client():
    return TestClient(app)


class TestListCourses:
    def test_returns_18_courses(self, client):
        resp = client.get("/api/courses")
        assert resp.status_code == 200
        courses = resp.json()
        assert len(courses) == 18

    def test_courses_have_key_and_title(self, client):
        resp = client.get("/api/courses")
        for c in resp.json():
            assert "key" in c
            assert "title" in c


class TestGetCourse:
    def test_valid_course(self, client):
        resp = client.get("/api/courses/ap_biology")
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "AP Biology"
        assert data["mcq_total"] == 60

    def test_invalid_course(self, client):
        resp = client.get("/api/courses/ap_fake")
        assert resp.status_code == 404


class TestPredict:
    def test_valid_biology_prediction(self, client):
        resp = client.post("/api/predict", json={
            "course": "ap_biology",
            "mcq_correct": 42,
            "frq_scores": [7, 8, 3, 4, 2, 3],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_distribution" in data
        assert "most_likely_score" in data
        assert "expected_score" in data
        assert "weighted_composite" in data
        assert "confidence_band" in data
        assert "explanations" in data

    def test_invalid_course_predict(self, client):
        resp = client.post("/api/predict", json={
            "course": "ap_fake",
            "mcq_correct": 30,
            "frq_scores": [5],
        })
        assert resp.status_code == 422

    def test_mcq_out_of_range(self, client):
        resp = client.post("/api/predict", json={
            "course": "ap_biology",
            "mcq_correct": 999,
            "frq_scores": [7, 8, 3, 4, 2, 3],
        })
        assert resp.status_code == 422

    def test_multi_section_apush(self, client):
        resp = client.post("/api/predict", json={
            "course": "ap_us_history",
            "mcq_correct": 40,
            "frq_sections": [
                {"name": "saq", "scores": [3, 3, 2]},
                {"name": "dbq", "scores": [5]},
                {"name": "leq", "scores": [4]},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["course"] == "ap_us_history"


class TestBatch:
    def test_batch_predict(self, client):
        resp = client.post("/api/predict/batch", json={
            "inputs": [
                {
                    "course": "ap_biology",
                    "mcq_correct": 42,
                    "frq_scores": [7, 8, 3, 4, 2, 3],
                },
                {
                    "course": "ap_calculus_ab",
                    "mcq_correct": 30,
                    "frq_scores": [5, 5, 5, 5, 5, 5],
                },
            ]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2
        assert len(data["errors"]) == 0

    def test_batch_with_errors(self, client):
        resp = client.post("/api/predict/batch", json={
            "inputs": [
                {
                    "course": "ap_biology",
                    "mcq_correct": 42,
                    "frq_scores": [7, 8, 3, 4, 2, 3],
                },
                {
                    "course": "ap_biology",
                    "mcq_correct": 999,
                    "frq_scores": [7, 8, 3, 4, 2, 3],
                },
            ]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert len(data["errors"]) == 1
