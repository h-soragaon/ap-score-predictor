"""FastAPI backend for the AP Score Predictor."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .course_registry import CourseRegistry
from .models import PredictionInput, PredictionOutput
from .predict import predict

app = FastAPI(title="AP Score Predictor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = CourseRegistry()


@app.get("/api/courses")
def list_courses():
    return registry.list_courses()


@app.get("/api/courses/{course_key}")
def get_course(course_key: str):
    try:
        config = registry.get_config(course_key)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return config.model_dump()


@app.post("/api/predict", response_model=PredictionOutput)
def predict_score(inp: PredictionInput):
    try:
        return predict(inp, registry)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


class BatchRequest(BaseModel):
    inputs: list[PredictionInput]


class BatchResponse(BaseModel):
    results: list[PredictionOutput]
    errors: list[dict]


@app.post("/api/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest):
    results = []
    errors = []
    for i, inp in enumerate(batch.inputs):
        try:
            results.append(predict(inp, registry))
        except ValueError as e:
            errors.append({"index": i, "error": str(e)})
    return BatchResponse(results=results, errors=errors)


# Serve static frontend
APP_DIR = Path(__file__).parent.parent / "app"
if APP_DIR.exists():
    @app.get("/")
    def serve_index():
        return FileResponse(APP_DIR / "index.html")

    app.mount("/app", StaticFiles(directory=str(APP_DIR)), name="static")
