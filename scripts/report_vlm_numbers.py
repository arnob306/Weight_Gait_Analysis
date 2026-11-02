"""
Every VLM number the report quotes beyond macro-F1, from the saved result files (no GPU).

  vlm_bootstrap.csv       held-out macro-F1 per prompt (montage) with a 95% bootstrap interval over clips
  vlm_bootstrap_people.csv  the same interval, resampling the 10 held-out participants instead of clips
  vlm_random_guess.csv    macro-F1 of uniform random guessing on 36 balanced clips (chance level for macro-F1)
  vlm_confusion.csv       held-out confusion counts (true class x predicted class) for Z, R, S on montage
  vlm_pred_counts.csv     dev: how often each prompt x input predicted each class (invalid = -1)
  vlm_ankle_cut.csv       dev clips run both ways, R prompt, raw video vs ankles blacked out: predictions per class
  vlm_weights_probe.csv   "is anything strapped to either leg?" per input and true class: any leg flagged,
                          the right legs flagged
  vlm_confidence.csv      R on raw video at temperature 0.7, three seeds, 15 clips: share 'high',
                          accuracy at 'high', clips with the same answer in all three runs
  vlm_pairs.csv           held-out yes/no pairs per topic: both answers right, same answer to both,
                          and false 'yes' to the question whose condition is absent
  vlm_error_overlap.csv   dev, R prompt: Jaccard overlap of the clips each input gets wrong

Run from scripts/:  python report_vlm_numbers.py
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from labels import load_labels
from macro_f1_table import CLASSES, INPUTS, scores

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
REPORT = ROOT / "report"
INVALID = -1
BOOTSTRAP_RESAMPLES = 10_000


def load(*names: str) -> pd.DataFrame:
    rows = [json.loads(line) for name in names for line in (RESULTS / name).read_text().splitlines()]
    frame = pd.DataFrame(rows)
    if "pred" in frame:
        frame["pred"] = frame["pred"].map(lambda p: int(p) if p in CLASSES else INVALID)
    return frame


def confusion() -> pd.DataFrame:
    held = load("gemma_heldout_classification.jsonl")
    held = held[held["rep"] == "montage"]
    return (held.groupby(["prompt", "true_class", "pred"]).size()
            .unstack("pred", fill_value=0).reset_index())


def pred_counts(dev: pd.DataFrame) -> pd.DataFrame:
    framed = dev[dev["rep"].isin(INPUTS)]
    return (framed.groupby(["prompt", "rep", "pred"]).size()
            .unstack("pred", fill_value=0).reset_index())


def ankle_cut(dev: pd.DataFrame) -> pd.DataFrame:
    cut = load("gemma_ankle_cut_and_2fps_R.jsonl")
    cut = cut[(cut["rep"] == "raw_ankle_cut") & (cut["fps"] == 1)]
    raw = dev[(dev["prompt"] == "R_rubric") & (dev["rep"] == "raw_video")]
    clips = set(cut["instance"]) & set(raw["instance"])  # compare on the clips run both ways
    both = pd.concat([raw[raw["instance"].isin(clips)].assign(input="raw_video"),
                      cut[cut["instance"].isin(clips)].assign(input="raw_ankle_cut")])
    table = both.groupby(["input", "true_class", "pred"]).size().unstack("pred", fill_value=0).reset_index()
    table["n_clips"] = len(clips)
    return table


def weights_probe() -> pd.DataFrame:
    probe = load("gemma_probe_weights.jsonl", "gemma_new_dev_probes.jsonl")
    truth = {0: ("no", "no"), 1: ("yes", "no"), 2: ("yes", "yes")}  # class 1 = left ankle weighted
    probe["any_flagged"] = (probe["left_attach"] == "yes") | (probe["right_attach"] == "yes")
    probe["legs_right"] = [(l, r) == truth[c] for l, r, c in
                           zip(probe["left_attach"], probe["right_attach"], probe["true_class"])]
    return (probe.groupby(["rep", "true_class"])
            .agg(clips=("instance", "nunique"), any_flagged=("any_flagged", "sum"),
                 legs_right=("legs_right", "sum")).reset_index())


def confidence() -> pd.DataFrame:
    runs = load("gemma_sampled_conf.jsonl")
    high = runs[runs["confidence"] == "high"]
    same = runs.groupby("instance")["pred"].nunique().eq(1).sum()
    return pd.DataFrame([{"answers": len(runs), "clips": runs["instance"].nunique(),
                          "high": len(high), "high_share": len(high) / len(runs),
                          "correct_at_high": int((high["pred"] == high["true_class"]).sum()),
                          "accuracy_at_high": float((high["pred"] == high["true_class"]).mean()),
                          "accuracy_all": float((runs["pred"] == runs["true_class"]).mean()),
                          "clips_same_answer_all_runs": int(same)}])


def pairs() -> pd.DataFrame:
    probes = load("gemma_heldout_pairs.jsonl")
    out = []
    for topic, g in probes.groupby("topic", sort=False):
        side = g.pivot(index="instance", columns="side", values="answer")
        ok = g.pivot(index="instance", columns="side", values="correct")
        present = g.groupby("instance")["present"].first()
        # the question whose condition is absent: A ("is it present?") when absent, B ("is it absent?") when present
        absent_q = [side.loc[i, "A"] if not present[i] else side.loc[i, "B"] for i in side.index]
        out.append({"topic": topic, "clips": len(side), "condition_present": int(present.sum()),
                    "both_right": int((ok["A"] & ok["B"]).sum()),
                    "same_answer_to_both": int((side["A"] == side["B"]).sum()),
                    "no_to_both": int(((side["A"] == "no") & (side["B"] == "no")).sum()),
                    "false_yes": int(sum(a == "yes" for a in absent_q))})
    table = pd.DataFrame(out)
    table["hallucination_rate"] = 1 - table["both_right"] / table["clips"]
    return table


def error_overlap(dev: pd.DataFrame) -> pd.DataFrame:
    r = dev[(dev["prompt"] == "R_rubric") & dev["rep"].isin(INPUTS)]
    wrong = {rep: set(g.loc[g["pred"] != g["true_class"], "instance"]) for rep, g in r.groupby("rep")}
    rows = [{"a": a, "b": b, "jaccard": len(wrong[a] & wrong[b]) / len(wrong[a] | wrong[b]),
             "wrong_a": len(wrong[a]), "wrong_b": len(wrong[b]), "both_wrong": len(wrong[a] & wrong[b])}
            for a, b in combinations(INPUTS, 2)]
    rows.append({"a": "all four", "b": "", "both_wrong": len(set.intersection(*wrong.values()))})
    return pd.DataFrame(rows)


def bootstrap(n_boot: int = BOOTSTRAP_RESAMPLES, seed: int = 27) -> pd.DataFrame:
    """95% percentile interval of held-out macro-F1 per prompt, resampling the 36 clips with replacement."""
    held = load("gemma_heldout_classification.jsonl")
    held = held[held["rep"] == "montage"]
    rng = np.random.default_rng(seed)
    rows = []
    for prompt, g in held.groupby("prompt"):
        true, pred = g["true_class"].to_numpy(), g["pred"].to_numpy()
        draws = [scores(list(true[idx]), list(pred[idx]))["macro_f1"]
                 for idx in (rng.integers(0, len(true), len(true)) for _ in range(n_boot))]
        rows.append({"prompt": prompt, "macro_f1": scores(list(true), list(pred))["macro_f1"],
                     "ci_low": float(np.percentile(draws, 2.5)), "ci_high": float(np.percentile(draws, 97.5))})
    return pd.DataFrame(rows)


def bootstrap_people(n_boot: int = BOOTSTRAP_RESAMPLES, seed: int = 27) -> pd.DataFrame:
    """As bootstrap(), but resampling the 10 held-out participants (all of a person's clips together), the unit
    the pose strand resamples."""
    held = load("gemma_heldout_classification.jsonl")
    held = held[held["rep"] == "montage"]
    person = load_labels(str(ROOT / "All Joints" / "Absolute_Data.csv"))["person_id"]
    held = held.assign(person=held["instance"].map(person))
    people = held["person"].unique()
    rng = np.random.default_rng(seed)
    rows = []
    for prompt, g in held.groupby("prompt"):
        by_person = {p: (list(c["true_class"]), list(c["pred"])) for p, c in g.groupby("person")}
        draws = []
        for _ in range(n_boot):
            picked = rng.choice(people, len(people))
            true = [t for p in picked for t in by_person[p][0]]
            pred = [q for p in picked for q in by_person[p][1]]
            draws.append(scores(true, pred)["macro_f1"])
        rows.append({"prompt": prompt, "n_people": len(people),
                     "macro_f1": scores(list(g["true_class"]), list(g["pred"]))["macro_f1"],
                     "ci_low": float(np.percentile(draws, 2.5)), "ci_high": float(np.percentile(draws, 97.5))})
    return pd.DataFrame(rows)


def random_guess(n_sims: int = 20_000, seed: int = 27) -> pd.DataFrame:
    """Macro-F1 of uniform random guessing on 36 balanced clips, the chance level for macro-F1
    (the always-0 baseline is lower, 0.167). p_at_least_S: share of guessers reaching S's held-out score."""
    held = load("gemma_heldout_classification.jsonl")
    s = held[(held["rep"] == "montage") & (held["prompt"] == "S_few_shot")]
    s_f1 = scores(list(s["true_class"]), list(s["pred"]))["macro_f1"]
    true = [c for c in CLASSES for _ in range(12)]
    rng = np.random.default_rng(seed)
    draws = np.array([scores(true, list(rng.integers(0, 3, len(true))))["macro_f1"] for _ in range(n_sims)])
    return pd.DataFrame([{"sims": n_sims, "mean_macro_f1": draws.mean(),
                          "p95_macro_f1": float(np.percentile(draws, 95)),
                          "S_heldout_macro_f1": s_f1, "p_at_least_S": float((draws >= s_f1).mean())}])


def main() -> None:
    REPORT.mkdir(exist_ok=True)
    dev = load("gemma_dev36_R_Z.jsonl", "gemma_dev36_S.jsonl")
    tables = {"vlm_bootstrap": bootstrap(), "vlm_bootstrap_people": bootstrap_people(),
              "vlm_random_guess": random_guess(),
              "vlm_confusion": confusion(), "vlm_pred_counts": pred_counts(dev),
              "vlm_ankle_cut": ankle_cut(dev), "vlm_weights_probe": weights_probe(),
              "vlm_confidence": confidence(), "vlm_pairs": pairs(), "vlm_error_overlap": error_overlap(dev)}
    for name, table in tables.items():
        table.to_csv(REPORT / f"{name}.csv", index=False)
        print(f"\n== {name} ==\n{table.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
