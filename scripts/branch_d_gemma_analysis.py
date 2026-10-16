"""
Branch D with Gemma, analysis (local, no GPU). Reads the Gemma ratings (branch_d_gemma.jsonl,
one line per clip, input and attribute group) and the 27 measured values of the same 36 dev
clips (results/measured_dev36.csv, from branch_d_measured_dev.py), and reports for each input:

  - per attribute: how many clips were rated, null, invalid, and how many distinct ratings
  - per attribute: Spearman rank correlation with the measured value, over the clips where the
    rating is 0, 1 or 2 and the measured value exists, only when the ratings take at least two
    values and there are at least MIN_PAIRS clips
  - a classifier on the ratings alone (leave-one-participant-out logistic regression, three
    classes: all clips of a participant are held out together) against the 12-of-36 always-class-0
    baseline

Run from scripts/:  python branch_d_gemma_analysis.py ../results/branch_d_gemma.jsonl
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from branch_d_elicit import ATTRIBUTE_GROUPS
from samples import INSTANCES_PER_PARTICIPANT

ATTRIBUTES = [key for group in ATTRIBUTE_GROUPS.values() for key, _, _ in group]
MEASURED_PATH = Path(__file__).resolve().parent.parent / "results" / "measured_dev36.csv"
MIN_PAIRS = 8


def load_ratings(path: Path) -> pd.DataFrame:
    """One row per (rep, instance): the 27 ratings (NaN where null or invalid) and statuses."""
    rows = {}
    for record in map(json.loads, path.read_text().splitlines()):
        row = rows.setdefault((record["rep"], record["instance"]),
                              {"rep": record["rep"], "instance": record["instance"],
                               "weight_class": record["weight_class"]})
        for key, value in record["ratings"].items():
            row[key] = np.nan if value is None else value
            row[f"{key}__status"] = record["status"][key]
    return pd.DataFrame(rows.values())


def attribute_table(ratings: pd.DataFrame, measured: pd.DataFrame) -> pd.DataFrame:
    merged = ratings.merge(measured[["instance", *ATTRIBUTES]], on="instance", suffixes=("", "_measured"))
    out = []
    for key in ATTRIBUTES:
        status = merged[f"{key}__status"]
        rated = merged[key].notna() & merged[f"{key}_measured"].notna()
        values = merged.loc[rated, key]
        rho = p = np.nan
        if len(values) >= MIN_PAIRS and values.nunique() >= 2:
            rho, p = spearmanr(values, merged.loc[rated, f"{key}_measured"])
        out.append({"attribute": key, "rated": int((status == "rated").sum()),
                    "null": int((status == "null").sum()), "invalid": int((status == "invalid").sum()),
                    "distinct": int(merged[key].nunique()), "pairs": int(rated.sum()),
                    "spearman": rho, "p": p})
    return pd.DataFrame(out)


def classify_from_ratings(ratings: pd.DataFrame) -> dict:
    """Leave-one-participant-out logistic regression on the ratings (missing = 1, 'typical').
    A participant's clips never sit on both sides of the split."""
    x = ratings[ATTRIBUTES].fillna(1).to_numpy(dtype=float)
    y = ratings["weight_class"].to_numpy()
    groups = ratings["instance"].to_numpy() // INSTANCES_PER_PARTICIPANT
    if np.ptp(x, axis=0).sum() == 0:  # every rating identical: nothing to fit
        return {"correct": int((y == np.bincount(y).argmax()).sum()), "n": len(y), "informative": False}
    correct = 0
    for person in np.unique(groups):
        test = groups == person
        model = LogisticRegression(max_iter=1000).fit(x[~test], y[~test])
        correct += int((model.predict(x[test]) == y[test]).sum())
    return {"correct": correct, "n": len(y), "informative": True}


def main(path: Path) -> None:
    ratings, measured = load_ratings(path), pd.read_csv(MEASURED_PATH)
    for rep, group in ratings.groupby("rep"):
        table = attribute_table(group, measured)
        computable = int((table["spearman"].notna()).sum())
        print(f"\n== {rep}: {group['instance'].nunique()} clips ==")
        print(table.round(3).to_string(index=False))
        print(f"attributes with a computable correlation: {computable}/27; "
              f"attributes with one distinct rating: {int((table['distinct'] <= 1).sum())}/27; "
              f"null share {table['null'].sum() / (27 * group['instance'].nunique()):.2f}")
        print("classifier on ratings:", classify_from_ratings(group))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
