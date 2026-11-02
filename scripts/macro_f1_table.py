"""
Macro-F1, accuracy and balanced accuracy for every Gemma and Qwen setting, from saved results (no GPU).

  Gemma dev       12 settings (3 prompts x 4 inputs), 36 clips             results/gemma_dev36_*.jsonl
  Gemma held-out  prompts Z, R, S (montage) and any control rows, 36 clips  results/gemma_heldout_classification.jsonl
  Qwen dev        4 inputs (rubric prompt) and zero-shot / few-shot (skeleton), 30 clips, 8 frames
                  printed output of notebooks/notebook1f2b9c0b76.ipynb

Same metrics as the classical branches: macro-F1 (mean of the three per-class F1), accuracy and balanced
accuracy, plus recall per class. An invalid answer (no usable class) is never a hit, so it lowers recall and adds no false positive.
The majority-class baseline is "always predict 0".

Run from scripts/:  python macro_f1_table.py
Writes report/macro_f1_summary.csv.
"""
import ast
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
REPORT = ROOT / "report"
QWEN_NOTEBOOK = ROOT / "notebooks" / "notebook1f2b9c0b76.ipynb"
GEMMA_DEV_FILES = ("gemma_dev36_R_Z.jsonl", "gemma_dev36_S.jsonl")
GEMMA_HELD_OUT_FILE = "gemma_heldout_classification.jsonl"
INPUTS = ("raw_video", "skeleton", "overlay", "montage")
CLASSES = (0, 1, 2)
QWEN_INPUT_LINE = re.compile(r"\[(\d+)\]\[(\w+)\] gt=(\d) pred=(\w+)")
QWEN_PROMPT_LINE = re.compile(r"\[(\d+)\] gt=(\d) preds_by_prompt=(\{.*\})")


def scores(true: list, pred: list) -> dict:
    """Accuracy, macro-F1 and balanced accuracy over the three classes."""
    f1s, recalls = [], []
    for c in CLASSES:
        tp = sum(t == c and p == c for t, p in zip(true, pred))
        fp = sum(t != c and p == c for t, p in zip(true, pred))
        fn = sum(t == c and p != c for t, p in zip(true, pred))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
        recalls.append(tp / (tp + fn) if tp + fn else 0.0)
    correct = sum(t == p for t, p in zip(true, pred))
    return {"n": len(true), "correct": correct, "accuracy": correct / len(true),
            "macro_f1": sum(f1s) / 3, "balanced_accuracy": sum(recalls) / 3,
            **{f"recall_{c}": r for c, r in zip(CLASSES, recalls)}}


def row(model: str, data_set: str, setting: str, true: list, pred: list) -> dict:
    return {"model": model, "set": data_set, "setting": setting, **scores(true, pred)}


def load_jsonl(names) -> pd.DataFrame:
    return pd.DataFrame([json.loads(line) for name in names
                         for line in (RESULTS / name).read_text().splitlines()])


def gemma_rows() -> list[dict]:
    rows = []
    dev = load_jsonl(GEMMA_DEV_FILES)
    dev = dev[dev["rep"].isin(INPUTS)]
    for (prompt, rep), g in dev.groupby(["prompt", "rep"], sort=False):
        rows.append(row("Gemma", "dev", f"{prompt} / {rep}", g["true_class"].tolist(), g["pred"].tolist()))
    held = load_jsonl([GEMMA_HELD_OUT_FILE])
    for (prompt, rep), g in held.groupby(["prompt", "rep"], sort=False):
        if rep == "blind":  # no images: one call per prompt, the same answer for every clip
            true = held[(held["prompt"] == prompt) & (held["rep"] == "montage")]["true_class"].tolist()
            rows.append(row("Gemma", "held-out", f"{prompt} / blind", true, g["pred"].tolist() * len(true)))
        else:
            rows.append(row("Gemma", "held-out", f"{prompt} / {rep}", g["true_class"].tolist(), g["pred"].tolist()))
    return rows


def notebook_text() -> str:
    cells = json.loads(QWEN_NOTEBOOK.read_text(encoding="utf8"))["cells"]
    parts = []
    for cell in cells:
        for out in cell.get("outputs", []):
            text = out.get("text") or ""
            parts.append("".join(text) if isinstance(text, list) else text)
    return "\n".join(parts)


def qwen_rows() -> list[dict]:
    text = notebook_text()
    by_input: dict[str, dict[int, tuple[int, str]]] = {}
    for inst, rep, gt, pred in QWEN_INPUT_LINE.findall(text):
        by_input.setdefault(rep, {})[int(inst)] = (int(gt), pred)  # a rerun overwrites, last run wins
    rows = []
    for rep in INPUTS:
        pairs = by_input.get(rep, {})
        true = [g for g, _ in pairs.values()]
        pred = [int(p) if p in ("0", "1", "2") else None for _, p in pairs.values()]
        rows.append(row("Qwen", "dev", f"rubric / {rep}", true, pred))
    sweep: dict[str, dict[int, tuple[int, int | None]]] = {}
    for inst, gt, preds in QWEN_PROMPT_LINE.findall(text):
        for name, pred in ast.literal_eval(preds).items():
            sweep.setdefault(name, {})[int(inst)] = (int(gt), pred)
    for name, pairs in sweep.items():
        true = [g for g, _ in pairs.values()]
        rows.append(row("Qwen", "dev", f"{name} / skeleton", true, [p for _, p in pairs.values()]))
    return rows


def baseline_rows() -> list[dict]:
    return [row(model, data_set, "always predict 0", [0] * third + [1] * third + [2] * third, [0] * n)
            for model, data_set, n in (("both", "36 clips", 36), ("Qwen", "30 clips", 30))
            for third in (n // 3,)]


def main() -> None:
    REPORT.mkdir(exist_ok=True)
    table = pd.DataFrame(gemma_rows() + qwen_rows() + baseline_rows())
    table.to_csv(REPORT / "macro_f1_summary.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
