# AP Score Predictor Research Pack

Prepared for building a course-specific AP score probability predictor using public College Board data and open calculator signals.

## Executive summary

- Best current public calculators appear to use deterministic or near-deterministic weighted composites plus approximate cutoff ranges.

- College Board does **not** publicly publish the exact annual composite-to-1/5 cut points for all exams, but it does publicly describe the score-setting process and publish enough auxiliary data to build a stronger probabilistic estimator.

- The highest-value official data are: current exam blueprints, yearly subject score distributions, and yearly FRQ scoring statistics.

- Recommended first release: a **year-aware Bayesian cutoff model** that returns `P(score = 1..5)` instead of a single hard score.

- Recommended second release (after your school accumulates actual AP outcomes): local recalibration on your own students.


## Public data sources to harvest

1. **Current course exam pages (AP Central / AP Students)** for exact section counts, timing, section weights, and special score structures.

2. **AP Data and Research** for current score distributions by subject and the 2021-2024 archive.

3. **Course-specific score distribution PDFs** for each year you can recover (2021-2025 are easiest; 2018-2019 are also discoverable for many subjects).

4. **Course-specific scoring statistics PDFs** for each year, which give FRQ question means, SDs, and max points.

5. **Scoring guidelines / rubric PDFs** for question semantics and question-level max points if a scoring-statistics file is missing.

6. **Open calculators** (Albert, Fiveable, Collegize, etc.) as noisy priors on cutoff locations—not as ground truth.


## Model recommendation

### Why not a single fixed cutoff table?

- Current open calculators publicly admit that cutoffs are approximate and vary by year.

- College Board uses evidence-based standard setting and statistical translation, not a simple fixed curve.

- Subject means and pass rates move over time, so a static table will drift.


### Recommended predictive model

Use a **course-specific ordered probability model** on a normalized composite scale.


For a student and course/year:

1. Convert raw inputs to a normalized weighted composite using official section weights.

2. Use FRQ question-level official scoring statistics to adjust for question difficulty and question reliability.

3. Combine multiple public cutoff sources into a prior distribution for each score boundary.

4. Constrain those cutoff priors using official yearly score distributions.

5. Add measurement error for practice-test-to-live-exam variation.

6. Return probabilities for scores 1-5 instead of a single deterministic point.


### Practical formula

Let `x` be the weighted composite in `[0, 1]`.

For ordered cutoffs `tau_2 < tau_3 < tau_4 < tau_5` and course-specific uncertainty `sigma`:


```text

P(Y <= k | x) = logistic((tau_k - x) / sigma)

P(Y = 1) = P(Y <= 1)

P(Y = 2) = P(Y <= 2) - P(Y <= 1)

...

P(Y = 5) = 1 - P(Y <= 4)

```


Where:

- `tau_k` are not fixed constants; they are fitted per course with optional year effects.

- `sigma` encodes uncertainty from yearly cutoff movement and practice-form mismatch.

- The posterior over `tau` is informed by open calculators + official score distributions + FRQ scoring statistics.


### Difficulty adjustment from FRQ scoring statistics

For each FRQ question `q` with max points `M_q`, official mean `mu_q`, and SD `sd_q`:

- compute student proportion `p_q = score_q / M_q`

- compute standardized performance `z_q = (p_q - mu_q/M_q) / (sd_q/M_q)`

- shrink extreme `z_q` values to avoid overreaction on small-rubric questions

- aggregate by section weight to get a `difficulty_adjustment`

- final latent composite can be `x_adj = clip(x + lambda * difficulty_adjustment, 0, 1)`


This uses information that most public calculators ignore.


## Validation reality check

Without your own students' eventual official AP scores, you can build a stronger open-data estimator, but you **cannot honestly prove** it beats the best public calculator on your student population. The correct benchmark plan is:

1. Backtest against held-out historical public cutoff heuristics.

2. Compare against deterministic public calculators.

3. As soon as official scores arrive for your students, run real calibration and error analysis.


## Claude Code build plan

### Suggested repo structure

```text

ap-score-predictor/

  data/

    raw/

    processed/

    course_config.json

  harvest/

    harvest_exam_pages.py

    harvest_score_distributions.py

    harvest_scoring_statistics.py

    harvest_open_calculators.py

  src/

    course_registry.py

    scoring_inputs.py

    composite.py

    priors.py

    fit_cutoffs.py

    predict.py

    calibration.py

    api.py

  app/

    streamlit_app.py   # or Next.js frontend if preferred

  tests/

  README.md

```


### Inputs schema

```json

{

  "course": "ap_biology",

  "exam_year": 2026,

  "mcq_correct": 42,

  "mcq_total": 60,

  "frq_scores": [7, 8, 3, 4, 2, 3],

  "practice_form_id": "bio_form_a_optional",

  "cohort_context": null

}

```


### Output schema

```json

{

  "course": "ap_biology",

  "exam_year": 2026,

  "predicted_distribution": {"1": 0.03, "2": 0.09, "3": 0.31, "4": 0.40, "5": 0.17},

  "most_likely_score": 4,

  "expected_score": 3.59,

  "weighted_composite": 0.683,

  "difficulty_adjustment": 0.017,

  "confidence_band": {"p10": 3, "p90": 5},

  "explanations": [

    "MCQ performance is above the course median proxy.",

    "Long FRQ scores were stronger than the public national means.",

    "Result is near the 4/5 boundary, so uncertainty remains material."

  ]

}

```


### Shipping order

1. Build deterministic weighted-composite calculator from `course_config`.

2. Add open-calculator cutoff scraping and consensus cutoffs.

3. Add official score-distribution constraints.

4. Add FRQ question-level difficulty adjustment from scoring statistics.

5. Add Bayesian / ordered-logit probability outputs.

6. Add CSV batch API and simple UI.

7. Add local recalibration once official student outcomes exist.


## Course configuration summary

### AP Biology

- MCQ: 60 questions, weight 0.5000

- frq: weight 0.5000, question max points [9, 9, 4, 4, 4, 4]


### AP Calculus AB

- MCQ: 45 questions, weight 0.5000

- frq: weight 0.5000, question max points [9, 9, 9, 9, 9, 9]


### AP Calculus BC

- MCQ: 45 questions, weight 0.5000

- frq: weight 0.5000, question max points [9, 9, 9, 9, 9, 9]

- Special handling: Calculus AB subscore optional with additional tagged inputs


### AP Chemistry

- MCQ: 60 questions, weight 0.5000

- frq: weight 0.5000, question max points [10, 10, 10, 4, 4, 4, 4]


### AP Computer Science A

- MCQ: 42 questions, weight 0.5500

- frq: weight 0.4500, question max points [9, 9, 9, 9]


### AP English Language and Composition

- MCQ: 45 questions, weight 0.4500

- essays: weight 0.5500, question max points [6, 6, 6]


### AP English Literature and Composition

- MCQ: 55 questions, weight 0.4500

- essays: weight 0.5500, question max points [6, 6, 6]


### AP Environmental Science

- MCQ: 80 questions, weight 0.6000

- frq: weight 0.4000, question max points [10, 10, 10]


### AP Human Geography

- MCQ: 60 questions, weight 0.5000

- frq: weight 0.5000, question max points [7, 7, 7]


### AP Microeconomics

- MCQ: 60 questions, weight 0.6667

- frq: weight 0.3333, question max points [10, 5, 5]


### AP Music Theory

- MCQ: 75 questions, weight 0.4500

- written_frq: weight 0.4500, question max points [9, 9, 24, 24, 25, 18, 9]

- sight_singing: weight 0.1000, question max points [9, 9]

- Special handling: Aural and nonaural subscores optional with component-level MCQ inputs


### AP Physics 1: Algebra-Based

- MCQ: 40 questions, weight 0.5000

- frq: weight 0.5000, question max points [10, 12, 10, 8]


### AP Physics 2: Algebra-Based

- MCQ: 40 questions, weight 0.5000

- frq: weight 0.5000, question max points [10, 12, 10, 8]


### AP Psychology

- MCQ: 75 questions, weight 0.6667

- frq: weight 0.3333, question max points [7, 7]


### AP Statistics

- MCQ: 40 questions, weight 0.5000

- frq: weight 0.5000, question max points [4, 4, 4, 4, 4, 4]


### AP U.S. Government and Politics

- MCQ: 55 questions, weight 0.5000

- frq: weight 0.5000, question max points [3, 4, 4, 6]


### AP U.S. History

- MCQ: 55 questions, weight 0.4000

- saq: weight 0.2000, question max points [3, 3, 3]

- dbq: weight 0.2500, question max points [7]

- leq: weight 0.1500, question max points [6]

- Special handling: Last SAQ is choice-based; store as saq_q3_or_q4


### AP World History: Modern

- MCQ: 55 questions, weight 0.4000

- saq: weight 0.2000, question max points [3, 3, 3]

- dbq: weight 0.2500, question max points [7]

- leq: weight 0.1500, question max points [6]

- Special handling: Last SAQ is choice-based; store as saq_q3_or_q4



## Notes on special cases

- **AP Calculus BC**: overall BC score can be predicted from total BC inputs, but AB subscore requires item/topic tagging beyond simple total MCQ correct.

- **AP Music Theory**: overall score can be predicted from total MCQ + written FRQ + sight-singing inputs, but aural/nonaural subscores require splitting MCQ into aural and nonaural components.

- **AP U.S. History / AP World History: Modern**: do not collapse all written responses into one FRQ bucket; keep SAQ, DBQ, and LEQ as separate weighted components.

- **AP Microeconomics**: keep the long FRQ separate from the short FRQs because the long question is worth half of the FRQ section.


## Data-harvest rules for Claude Code

- Always version data by course and exam year.

- Prefer official College Board sources when they exist.

- If multiple public calculators disagree, treat their cutoffs as observations with uncertainty, not facts.

- Never hard-code one set of cutoffs forever.

- Keep model outputs probabilistic and expose calibration metadata.
