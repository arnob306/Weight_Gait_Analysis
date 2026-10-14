"""
Which of the 27 measured gait attributes differ between the three weight classes,
on the participants that are in neither the development nor the held-out set
(the "training" participants, free to use for choosing what the prompt asks about).

For every clip of those participants (12 people x 60 clips) it computes the 27
attributes of docs/branch_d_guideline.md section 7 with branch_d_features, over the
same walking window the v2 runs use (clip_meta.walking_window). Each person walks
all three classes, so the main test is within person: per-person median per class,
then a Friedman test across the three classes (n = number of people). A clip-level
Kruskal-Wallis p is shown for reference only (clips from one person are not
independent, so it is optimistic). p-values across the 27 attributes are
Benjamini-Hochberg adjusted.

Run from scripts/:  python measured_by_class.py
Writes results/measured_by_class.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, kruskal

import branch_d_features as bdf
import clip_meta as cm
import skeleton_utils as su
from labels import load_labels
from run_kaggle_qwen import INSTANCE_META
from samples import CLIPS_PER_PARTICIPANT_CLASS, DEV_PERSONS, HELD_OUT_PERSONS, N_PARTICIPANTS, instance_id

OUT_PATH = Path(__file__).resolve().parent.parent / "results" / "measured_by_class.csv"
CLASSES = (0, 1, 2)
MIN_PEOPLE = 5  # fewest people with all three class medians before a Friedman test is run


def training_persons() -> list[int]:
    used = set(DEV_PERSONS) | set(HELD_OUT_PERSONS)
    return [p for p in range(N_PARTICIPANTS) if p not in used]


def cache_joint_table() -> None:
    """Read the 41 MB CSV once; the loaders below then slice it instead of re-reading it."""
    table = pd.read_csv(bdf.CSV_PATH, header=None)
    by_instance = {k: g.sort_values(by=1) for k, g in table.groupby(0)}
    su.load_instance_joints = lambda csv_path, instance: by_instance[instance]


def clip_features(instance: int, weight_class: int) -> dict:
    frames = su.load_instance_joints(str(bdf.CSV_PATH), instance)
    per_frame = [su.joints_from_row(row) for _, row in frames.iterrows()]
    start, end = cm.walking_window(per_frame, su.active_frame_range(frames))
    INSTANCE_META[instance] = {"start": start, "end": end, "weight_class": weight_class}
    return bdf.compute_instance_features(bdf.CSV_PATH, instance)


def collect(persons: list[int], labels: pd.DataFrame, event_free: bool = False) -> pd.DataFrame:
    """Features for every clip of these people; event_free keeps only clips with neither a
    freeze nor an obstacle step (the stops and the high step-overs would otherwise look
    like gait differences)."""
    rows = []
    for p in persons:
        for c in CLASSES:
            for k in range(CLIPS_PER_PARTICIPANT_CLASS):
                i = instance_id(p, c, k)
                if int(labels.loc[i, "weight_class"]) != c or int(labels.loc[i, "person_id"]) != p:
                    raise ValueError(f"instance {i} does not match person {p}, class {c}")
                if event_free and (labels.loc[i, "gait_freeze"] or labels.loc[i, "obstacle"]):
                    continue
                rows.append({"person": p, "weight_class": c, "instance": i, **clip_features(i, c)})
    return pd.DataFrame(rows)


def benjamini_hochberg(p: list[float]) -> list[float]:
    """BH-adjusted p-values; NaN stays NaN."""
    arr = np.array(p, dtype=float)
    ok = ~np.isnan(arr)
    adj = np.full_like(arr, np.nan)
    order = np.argsort(arr[ok])
    ranked = arr[ok][order] * ok.sum() / (np.arange(ok.sum()) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(ok.sum())
    out[order] = np.minimum(ranked, 1.0)
    adj[ok] = out
    return adj.tolist()


def test_attribute(df: pd.DataFrame, name: str) -> dict:
    per_person = df.groupby(["person", "weight_class"])[name].median().unstack()
    per_person = per_person.dropna()
    friedman = float("nan")
    if len(per_person) >= MIN_PEOPLE:
        friedman = friedmanchisquare(*[per_person[c] for c in CLASSES]).pvalue
    groups = [df.loc[df.weight_class == c, name].dropna() for c in CLASSES]
    clip_p = kruskal(*groups).pvalue if all(len(g) > 1 for g in groups) else float("nan")
    return {
        "attribute": name,
        "median_class0": groups[0].median(), "median_class1": groups[1].median(),
        "median_class2": groups[2].median(),
        "people_used": len(per_person), "missing_share": float(df[name].isna().mean()),
        "people_c1_above_c0": int((per_person[1] > per_person[0]).sum()),
        "people_c2_above_c0": int((per_person[2] > per_person[0]).sum()),
        "friedman_p": friedman, "clip_level_p": clip_p,
    }


def analyse(df: pd.DataFrame) -> pd.DataFrame:
    names = [c for c in df.columns if c not in ("person", "weight_class", "instance")]
    table = pd.DataFrame([test_attribute(df, n) for n in names])
    table["friedman_p_bh"] = benjamini_hochberg(table["friedman_p"].tolist())
    return table.sort_values("friedman_p").reset_index(drop=True)


def main(event_free: bool = False) -> pd.DataFrame:
    cache_joint_table()
    labels = load_labels(str(bdf.CSV_PATH))
    persons = training_persons()
    print(f"training participants: {persons}; event-free clips only: {event_free}")
    df = collect(persons, labels, event_free)
    table = analyse(df)
    suffix = "_event_free" if event_free else ""
    out_path = OUT_PATH.with_name(f"measured_by_class{suffix}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    df.to_csv(out_path.with_name(f"measured_clip_features{suffix}.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3g}".format):
        print(table.to_string(index=False))
    print(f"\n{len(df)} clips; wrote {out_path}")
    return table


if __name__ == "__main__":
    import sys
    main(event_free="--event-free" in sys.argv)
