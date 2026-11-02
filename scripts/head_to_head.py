"""
Table 6 on identical clips: scores the pose models' clip-level predictions on the VLM held-out clips, beside
the Gemma prompts and the majority baseline, with the same metrics (macro-F1, accuracy, balanced accuracy,
per-class recall), 95% intervals and paired tests.

Input from the pose strand: a CSV with columns  model, instance, pred  (pred = 0, 1 or 2; one row per model and
held-out instance id from samples.HELD_OUT_INSTANCES). Clips missing for a model are reported, not guessed, and
every model is scored on the clips all of them have (clip 1094 is not in the pose data).

Intervals: 95% percentile bootstrap of macro-F1 (10,000 resamples, seed 27) over clips and over participants
(all of a person's clips together), as for Table 5. Paired tests: exact McNemar on correct/incorrect per clip.

Run from scripts/:  python head_to_head.py path/to/pose_heldout_predictions.csv
Writes report/head_to_head.csv and report/head_to_head_mcnemar.csv.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from labels import load_labels
from macro_f1_table import scores
from report_stats import mcnemar
from report_vlm_numbers import BOOTSTRAP_RESAMPLES
from samples import CLASS_OF, HELD_OUT_INSTANCES

ROOT = Path(__file__).resolve().parent.parent
GEMMA_HELD_OUT = ROOT / "results" / "gemma_heldout_classification.jsonl"
BASELINE = "Majority baseline (always normal)"


def gemma_predictions() -> pd.DataFrame:
    rows = [json.loads(line) for line in GEMMA_HELD_OUT.read_text().splitlines()]
    frame = pd.DataFrame(rows)
    frame = frame[frame["rep"] == "montage"]
    return frame.assign(model="Gemma " + frame["prompt"])[["model", "instance", "pred"]]


def interval(true: list, pred: list, groups: list, seed: int = 27) -> tuple[float, float]:
    """95% percentile interval of macro-F1, resampling the units in groups with replacement."""
    units = sorted(set(groups))
    members = {u: [k for k, g in enumerate(groups) if g == u] for u in units}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        idx = [k for u in rng.choice(units, len(units)) for k in members[u]]
        draws.append(scores([true[k] for k in idx], [pred[k] for k in idx])["macro_f1"])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def shared_predictions(pose_csv: Path) -> pd.DataFrame:
    """One row per model and clip, restricted to the held-out clips every model answered."""
    pose = pd.read_csv(pose_csv)[["model", "instance", "pred"]]
    both = pd.concat([pose, gemma_predictions()])
    both = both[both["instance"].isin(HELD_OUT_INSTANCES)]
    for model, g in both.groupby("model"):
        missing = sorted(set(HELD_OUT_INSTANCES) - set(g["instance"]))
        if missing:
            print(f"{model}: {len(missing)} held-out clips missing: {missing}")
    common = sorted(set.intersection(*(set(g["instance"]) for _, g in both.groupby("model"))))
    print(f"Scoring every model on the {len(common)} clips all of them have")
    baseline = pd.DataFrame({"model": BASELINE, "instance": common, "pred": 0})
    both = pd.concat([both[both["instance"].isin(common)], baseline])
    return both.sort_values(["model", "instance"]).reset_index(drop=True)


def main(pose_csv: Path) -> None:
    both = shared_predictions(pose_csv)
    person = load_labels(str(ROOT / "All Joints" / "Absolute_Data.csv"))["person_id"]
    rows, correct = [], {}
    for model, g in both.groupby("model"):
        clips = g["instance"].tolist()
        true, pred = [CLASS_OF[i] for i in clips], g["pred"].tolist()
        low_c, high_c = interval(true, pred, clips)
        low_p, high_p = interval(true, pred, [int(person[i]) for i in clips])
        rows.append({"model": model, "clips": len(clips), **scores(true, pred),
                     "ci_clips_low": low_c, "ci_clips_high": high_c, "ci_people_low": low_p, "ci_people_high": high_p})
        correct[model] = pd.Series([t == p for t, p in zip(true, pred)], index=clips)
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "report" / "head_to_head.csv", index=False)
    print(table.round(3).to_string(index=False))

    tests = []
    for a, b in combinations([m for m in correct if m != BASELINE], 2):
        only_a, only_b, p = mcnemar(correct[a], correct[b])
        tests.append({"a": a, "b": b, "only_a_right": only_a, "only_b_right": only_b, "mcnemar_p": p})
    paired = pd.DataFrame(tests)
    paired.to_csv(ROOT / "report" / "head_to_head_mcnemar.csv", index=False)
    print(paired.round(4).to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
