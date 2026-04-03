"""Pre-compute constrained priors for the static GitHub Pages site.

Runs constrain_priors() for each course/year and saves the results
so the browser never needs to run an optimizer.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.course_registry import CourseRegistry
from src.priors import constrain_priors


def main():
    registry = CourseRegistry()
    output = {}

    for key in sorted(registry.courses.keys()):
        priors = registry.get_cutoff_priors(key)

        # Try each available year
        years_available = registry.score_distributions.get(key, {})
        for year in sorted(years_available.keys()):
            dist = registry.get_score_distribution(key, year)
            constrained = constrain_priors(priors, dist)
            entry_key = f"{key}_{year}"
            output[entry_key] = {
                "tau": [round(t, 6) for t in constrained.tau],
                "sigma": round(constrained.sigma, 6),
            }

        # Also store the raw (unconstrained) priors as fallback
        output[key] = {
            "tau": [round(t, 6) for t in priors.tau],
            "sigma": round(priors.sigma, 6),
        }

    out_path = Path(__file__).parent.parent / "docs" / "data" / "constrained_priors.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {len(output)} entries to {out_path}")


if __name__ == "__main__":
    main()
