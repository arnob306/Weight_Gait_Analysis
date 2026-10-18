"""
Classification statistics for the report, from the saved result files (no GPU).

  dev        12 settings (3 prompts x 4 inputs) on the 36 dev clips: correct, exact one-sided binomial
             p against chance 1/3, Holm-adjusted p over the 12 settings, predicted-class counts,
             nine-class correct (3 x gait_class + event), and the always-"0" baseline.
  held-out   Z, R, S on montage: the same columns plus paired exact McNemar tests between prompts.

An invalid answer (no usable class) counts as wrong. The blind calls are reported separately.

Run from scripts/:  python report_stats.py
Writes report/dev_settings.csv, report/held_out.csv and report/held_out_pairs.csv.
"""
import json
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
REPORT = ROOT / "report"
DEV_FILES = ("gemma_dev36_R_Z.jsonl", "gemma_dev36_S.jsonl")
HELD_OUT_FILE = "gemma_heldout_classification.jsonl"
PROMPTS = ("Z_zero_shot", "R_rubric", "S_few_shot")
INPUTS = ("raw_video", "skeleton", "overlay", "montage")
CHANCE = 1 / 3
CLASSES = (0, 1, 2)


def load(names) -> pd.DataFrame:
    rows = [json.loads(line) for name in names for line in (RESULTS / name).read_text().splitlines()]
    return pd.DataFrame(rows)


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted, running = [0.0] * len(p_values), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def setting_row(group: pd.DataFrame) -> dict:
    correct = group["pred"] == group["true_class"]
    n = len(group)
    return {"n": n, "correct": int(correct.sum()),
            "p_vs_third": binomtest(int(correct.sum()), n, CHANCE, alternative="greater").pvalue,
            "nine_class_correct": int((group["nine_pred"] == group["nine_true"]).sum()),
            **{f"pred_{c}": int((group["pred"] == c).sum()) for c in CLASSES},
            **{f"hit_{c}": int((correct & (group["true_class"] == c)).sum()) for c in CLASSES}}


def dev_table(rows: pd.DataFrame) -> pd.DataFrame:
    framed = rows[rows["rep"].isin(INPUTS)]
    table = pd.DataFrame([{"prompt": p, "input": r, **setting_row(g)}
                          for (p, r), g in framed.groupby(["prompt", "rep"], sort=False)])
    table["p_holm"] = holm(table["p_vs_third"].tolist())
    table["always_0"] = int((framed[framed["rep"] == "montage"].drop_duplicates("instance")["true_class"] == 0).sum())
    return table


def mcnemar(a: pd.Series, b: pd.Series) -> tuple[int, int, float]:
    only_a, only_b = int((a & ~b).sum()), int((b & ~a).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return only_a, only_b, p


def held_out_table(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    montage = rows[rows["rep"] == "montage"]
    table = pd.DataFrame([{"prompt": p, **setting_row(montage[montage["prompt"] == p])} for p in PROMPTS])
    ok = {}
    for p in PROMPTS:
        clips = montage[montage["prompt"] == p].set_index("instance")
        ok[p] = clips["pred"] == clips["true_class"]
    pairs = pd.DataFrame([{"a": a, "b": b, **dict(zip(("only_a", "only_b", "p"), mcnemar(ok[a], ok[b])))}
                          for a, b in combinations(PROMPTS, 2)])
    return table, pairs


def main() -> None:
    REPORT.mkdir(exist_ok=True)
    dev = dev_table(load(DEV_FILES))
    held_table, held_pairs = held_out_table(load([HELD_OUT_FILE]))
    dev.to_csv(REPORT / "dev_settings.csv", index=False)
    held_table.to_csv(REPORT / "held_out.csv", index=False)
    held_pairs.to_csv(REPORT / "held_out_pairs.csv", index=False)
    print("DEV (always-0 baseline = %d)" % dev["always_0"].iloc[0])
    print(dev.drop(columns="always_0").round(3).to_string(index=False))
    print("\nHELD-OUT, montage")
    print(held_table.round(3).to_string(index=False))
    print(held_pairs.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
