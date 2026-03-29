# AP Score Predictor

A Bayesian AP score probability predictor for 18 courses. Unlike deterministic public calculators that output a single score, this returns `P(score=1..5)` using an ordered logit model calibrated against public College Board data.

## Supported Courses

AP Biology, AP Calculus AB, AP Calculus BC, AP Chemistry, AP Computer Science A, AP English Language and Composition, AP English Literature and Composition, AP Environmental Science, AP Human Geography, AP Microeconomics, AP Music Theory, AP Physics 1, AP Physics 2, AP Psychology, AP Statistics, AP U.S. Government and Politics, AP U.S. History, AP World History: Modern.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Web UI

```bash
uvicorn src.api:app --reload
```

Open http://localhost:8000 — select a course, enter scores, get probability distributions.

### API

```bash
# List courses
curl http://localhost:8000/api/courses

# Get course config
curl http://localhost:8000/api/courses/ap_biology

# Single prediction
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"course":"ap_biology","mcq_correct":42,"frq_scores":[7,8,3,4,2,3]}'

# Batch prediction
curl -X POST http://localhost:8000/api/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"inputs":[{"course":"ap_biology","mcq_correct":42,"frq_scores":[7,8,3,4,2,3]}]}'
```

### CLI

```bash
# List courses
python -m src.cli courses

# Single prediction
python -m src.cli predict --course ap_biology --mcq 42 --frq 7,8,3,4,2,3

# Multi-section course (APUSH)
python -m src.cli predict --course ap_us_history --mcq 40 \
  --frq unused --section saq:3,3,2 --section dbq:5 --section leq:4

# Batch CSV
python -m src.cli batch --input input.csv --output output.csv
```

### CSV Format

For single-section courses, use columns: `course, mcq_correct, frq_1, frq_2, ...`

For multi-section courses (APUSH, World History), use: `course, mcq_correct, saq_1, saq_2, saq_3, dbq_1, leq_1`

## Model

The prediction pipeline:

1. **Weighted composite**: Normalize raw inputs to `[0, 1]` using official section weights
2. **Difficulty adjustment**: Z-score each FRQ question against national means/SDs from scoring statistics (when available), shrink extremes, aggregate by section weight
3. **Ordered logit**: `P(Y <= k | x) = logistic((tau_k - x_adj) / sigma)` where `x_adj = clip(composite + lambda * difficulty_adjustment, 0, 1)`
4. **Probability distribution**: Derive `P(score=k)` from cumulative probabilities
5. **Confidence band**: 10th and 90th percentile scores

The cutoff parameters (`tau`, `sigma`) are pre-seeded estimates that can be refined by fitting against official score distributions using KL-divergence minimization.

## Data Refresh Guide

To update for a new exam year:

1. **Score distributions**: Add a JSON file to `data/score_distributions/` with the official percentages:
   ```json
   {"course": "ap_biology", "year": 2025, "distribution": {"1": 0.08, "2": 0.20, "3": 0.25, "4": 0.27, "5": 0.20}}
   ```

2. **Scoring statistics**: Add a JSON file to `data/scoring_statistics/` with FRQ question-level means and SDs from College Board:
   ```json
   {"course": "ap_biology", "year": 2025, "questions": [{"question": 1, "section": "frq", "max_points": 9, "mean": 4.2, "sd": 2.1}, ...]}
   ```

3. **Cutoff priors**: Optionally update `data/cutoff_priors.json` or let the model auto-constrain priors against score distributions.

## Limitations

- Cutoff parameters are pre-seeded estimates, not exact College Board values
- Without student outcome data, the model cannot be locally calibrated
- The difficulty adjustment only activates when scoring statistics are available
- Assumes a uniform distribution of student composites when fitting cutoffs (a simplifying assumption)
- Practice test scores may differ from live exam conditions

## Testing

```bash
pytest tests/ -v
```

## Project Structure

```
data/
  course_config.json          # Course structure configs for all 18 courses
  cutoff_priors.json          # Pre-seeded ordered logit parameters
  score_distributions/        # Official yearly score distributions
  scoring_statistics/         # Official FRQ scoring statistics
src/
  models.py                   # Pydantic data models
  course_registry.py          # Config loader and validator
  composite.py                # Weighted composite calculator
  priors.py                   # Ordered logit + prior constraining
  difficulty.py               # FRQ difficulty adjustment
  fit_cutoffs.py              # Fit cutoffs to score distributions
  predict.py                  # Main prediction pipeline
  api.py                      # FastAPI backend
  cli.py                      # CLI tool
app/
  index.html                  # Frontend
  app.js                      # Frontend logic
  style.css                   # Frontend styles
tests/
  conftest.py                 # Shared fixtures
  test_registry.py            # Registry and validation tests
  test_composite.py           # Composite calculator tests
  test_predict.py             # Prediction engine tests
  test_api.py                 # API endpoint tests
  test_cli.py                 # CLI tests
  test_all_courses.py         # Parametrized cross-course tests
  fixtures/                   # Sample inputs and outputs
```
