"""
Branch D, Phase 3 -- agreement. See docs/branch_d_guideline.md sections 3
and 8.

For each of the 27 attributes, correlates Phase 1's elicited ordinal rating
against the measured value branch_d_features.py computed from the joint
data, across every clip where both exist. Spearman rank correlation is
used, not Pearson: the elicited side is a 3-point ordinal scale, not a
continuous measurement, and rank correlation needs no unit conversion
between the measured side's pixel/frame-count units and the model's 0/1/2
ratings.

A low or negative correlation on an attribute the model rated confidently
(few nulls) is the hallucination signal guideline section 3 describes: the
model claimed to see something the joint data does not support.

Does not run until Phase 1's output (branch_d_elicit.py, run on Kaggle) has
been produced; branch_d_measured_features.json already exists (run
branch_d_features.py locally, no GPU needed).
"""
import json
import math
from pathlib import Path

from scipy.stats import spearmanr

from branch_d_elicit import ATTRIBUTE_GROUPS

ELICITATION_PATH = Path(__file__).resolve().parent.parent / "results" / "branch_d_elicitation.json"
FEATURES_PATH = Path(__file__).resolve().parent.parent / "results" / "branch_d_measured_features.json"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "results" / "branch_d_agreement.json"

ATTRIBUTE_KEYS = [key for group in ATTRIBUTE_GROUPS.values() for key, _, _ in group]
MIN_PAIRS = 5  # fewer paired observations than this and a correlation is too noisy to report as a number


def load_paired_values(elicitation: list[dict], features_by_instance: dict, key: str) -> tuple[list[int], list[float]]:
    """(elicited ratings, measured values) for every clip where the model
    rated this attribute (not null, not invalid) and the measured value is
    not nan."""
    ratings, measured = [], []
    for record in elicitation:
        rating = record["ratings"].get(key)
        value = features_by_instance.get(record["instance"], {}).get(key)
        if rating is None or value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        ratings.append(rating)
        measured.append(value)
    return ratings, measured


def elicitability_summary(elicitation: list[dict], key: str) -> dict:
    """Per-attribute breakdown of rated / null / invalid across all clips
    -- guideline section 6 treats this as a finding in its own right, not
    only a filter for the correlation step."""
    counts = {"rated": 0, "null": 0, "invalid": 0}
    for record in elicitation:
        counts[record["status"].get(key, "invalid")] += 1
    return counts


def run(elicitation_path: Path = ELICITATION_PATH, features_path: Path = FEATURES_PATH,
        output_path: Path = OUTPUT_PATH) -> dict:
    elicitation = json.loads(elicitation_path.read_text())
    features_by_instance = {r["instance"]: r["features"] for r in json.loads(features_path.read_text())}

    per_attribute = {}
    for key in ATTRIBUTE_KEYS:
        ratings, measured = load_paired_values(elicitation, features_by_instance, key)
        status_counts = elicitability_summary(elicitation, key)

        if len(ratings) < MIN_PAIRS or len(set(ratings)) < 2:
            correlation, p_value = None, None
        else:
            r, p = spearmanr(ratings, measured)
            correlation, p_value = float(r), float(p)

        per_attribute[key] = {
            "n_pairs": len(ratings),
            "status_counts": status_counts,
            "spearman_r": correlation,
            "p_value": p_value,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(per_attribute, indent=2))

    print(f"{'attribute':<32}{'n':>5}{'rated':>7}{'null':>6}{'r':>8}{'p':>8}")
    for key, stats in per_attribute.items():
        r = f"{stats['spearman_r']:.2f}" if stats["spearman_r"] is not None else "n/a"
        p = f"{stats['p_value']:.3f}" if stats["p_value"] is not None else "n/a"
        print(f"{key:<32}{stats['n_pairs']:>5}{stats['status_counts']['rated']:>7}"
              f"{stats['status_counts']['null']:>6}{r:>8}{p:>8}")
    print(f"Wrote {output_path}")
    return per_attribute


if __name__ == "__main__":
    run()
