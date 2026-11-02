"""
Report figures for the VLM sections, in the navy and orange palette of the pose figures.

  fig1_study_design.png          Figure 1: the shared data feeding the pose and VLM strands and the head-to-head
  fig2_inputs.png                Figure 2: clip 343 (both ankles weighted) as the model sees it in each input,
                                 three frames per input, plus the ankle-cut control
  fig4_errors_hallucination.png  Figure 4: (a) Jaccard overlap of the dev clips each input gets wrong (R prompt);
                                 (b) held-out yes/no pairs: share of clips where the two answers contradict
                                 the truth (hallucination rate) per topic

Inputs: the frame renders from presentation/make_figures.py and the CSVs from report_vlm_numbers.py.
Run from scripts/ after those two:  python make_report_figures.py
"""
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "presentation" / "figs"
REPORT = ROOT / "report"
OUT = REPORT / "figures"
PALETTE = tuple(matplotlib.colors.to_rgb(h) for h in ("#1f4e79", "#e08a1e"))  # navy and orange, as in the pose figures
INPUT_LABELS = {"raw_video": "RGB", "skeleton": "Skeleton", "overlay": "Overlay",
                "raw_ankle_cut": "RGB, ankles masked (control)"}  # text before the comma labels Figure 4a
TOPIC_LABELS = {"freeze": "Freeze", "obstacle": "Obstacle", "left_leg": "Left leg", "right_leg": "Right leg"}
FRAME_PICKS = (0, 2, 4)
DPI = 300

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "font.family": "DejaVu Sans"})


def rgb(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def figure_inputs() -> None:
    """Four frame strips in a 2 x 2 block on the left, the montage spanning both rows on the right."""
    fig = plt.figure(figsize=(7.2, 2.9))
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.95])
    strips = [fig.add_subplot(grid[r, c]) for r in range(2) for c in range(2)]
    for ax, (rep, label) in zip(strips, INPUT_LABELS.items()):
        ax.imshow(np.hstack([rgb(FRAMES / f"frames_{rep}_{k}.png") for k in FRAME_PICKS]))
        ax.set_title(label)
    montage = fig.add_subplot(grid[:, 2])
    montage.imshow(rgb(FRAMES / "montage_343.png"))
    montage.set_title("Montage (one image)")
    for ax in [*strips, montage]:
        ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig2_inputs.png", dpi=DPI)
    plt.close(fig)


def overlap_matrix() -> tuple[np.ndarray, list[str]]:
    table = pd.read_csv(REPORT / "vlm_error_overlap.csv").dropna(subset=["jaccard"])
    reps = list(dict.fromkeys(table["a"].tolist() + table["b"].tolist()))
    matrix = np.eye(len(reps))
    for a, b, j in table[["a", "b", "jaccard"]].itertuples(index=False):
        i, k = reps.index(a), reps.index(b)
        matrix[i, k] = matrix[k, i] = j
    return matrix, reps


def figure_errors_hallucination() -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1, 1.25]})
    matrix, reps = overlap_matrix()
    blues = plt.get_cmap("Blues")
    left.imshow(matrix, cmap=blues, vmin=0, vmax=1.5)
    names = [INPUT_LABELS.get(r, r.capitalize()).split(",")[0] for r in reps]
    left.set_xticks(range(len(reps)), names, rotation=30, ha="right")
    left.set_yticks(range(len(reps)), names)
    for i in range(len(reps)):
        for k in range(len(reps)):
            left.text(k, i, f"{matrix[i, k]:.2f}", ha="center", va="center",
                      color="white" if matrix[i, k] > 0.7 else "black")
    left.set_title("(a) Overlap of wrong clips across inputs\n(Jaccard, development set, rubric prompt)")

    pairs = pd.read_csv(REPORT / "vlm_pairs.csv")
    labels = [TOPIC_LABELS[t] for t in pairs["topic"]]
    rates = pairs["hallucination_rate"] * 100
    bars = right.bar(labels, rates, color=[PALETTE[1] if "leg" in t else PALETTE[0] for t in pairs["topic"]])
    for bar, row in zip(bars, pairs.itertuples()):
        x = bar.get_x() + bar.get_width() / 2
        right.text(x, bar.get_height() + 2, f"{row.clips - row.both_right}/{row.clips}", ha="center", va="bottom")
        if row.no_to_both:
            right.text(x, bar.get_height() / 2, f"'no' to\nboth\nquestions:\n{row.no_to_both}",
                       ha="center", va="center", color="white")
    right.set_ylim(0, 125)
    right.set_yticks(range(0, 101, 25))
    right.set_ylabel("Clips with at least one wrong answer (%)")
    right.set_title("(b) Paired yes/no questions, held-out set")
    right.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig4_errors_hallucination.png", dpi=DPI)
    plt.close(fig)


STUDY_BOXES = {  # name: (x centre, y centre, text, colour); facts from Sections 3.1, 3.3, 4 and 5.1
    "data": (0.50, 0.90, "WeightGait: 32 participants, 3 classes (normal, limp, shuffle),\n"
                         "clean, freeze and obstacle walks, RGB video and 2D pose", None),
    "pose": (0.24, 0.63, "Pose strand: 5 participant-wise folds\n(7, 7, 7, 4, 7 people), 7-frame windows", 0),
    "vlm": (0.76, 0.63, "VLM strand: seeded draw (seed 27)\ndevelopment 36 clips / 10 people,\n"
                        "held-out 36 clips / 10 other people", 1),
    "cnn": (0.12, 0.34, "Branch A\n1D-CNN on raw pose", 0),
    "xgb": (0.36, 0.34, "Branch B\nXGBoost on 27 features,\nSHAP", 0),
    "gemma": (0.76, 0.34, "Gemma-4-E4B, no training:\nprompts Z, R, S;\nheld-out run on the montage", 1),
    "res_pose": (0.24, 0.08, "Section 6.1: fold-level results", 0),
    "res_vlm": (0.76, 0.08, "Section 6.2: held-out results", 1),
    "joint": (0.50, 0.08, "Section 6.3: side by side,\neach against its own baseline", None),
}
STUDY_ARROWS = [("data", "pose"), ("data", "vlm"), ("pose", "cnn"), ("pose", "xgb"), ("vlm", "gemma"),
                ("cnn", "res_pose"), ("xgb", "res_pose"), ("gemma", "res_vlm"), ("cnn", "joint"),
                ("xgb", "joint"), ("gemma", "joint")]


def figure_study_design() -> None:
    """Figure 1: the shared dataset feeding the pose and VLM strands, which meet in the head-to-head."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
    boxes = {}
    for name, (x, y, text, colour) in STUDY_BOXES.items():
        face = "#f2f2f2" if colour is None else (*PALETTE[colour], 0.15)
        edge = "#555555" if colour is None else PALETTE[colour]
        boxes[name] = ax.text(x, y, text, ha="center", va="center", linespacing=1.3,
                              bbox={"boxstyle": "round,pad=0.45", "facecolor": face, "edgecolor": edge,
                                    "linewidth": 0.9})
    for a, b in STUDY_ARROWS:
        (xa, ya, *_), (xb, yb, *_) = STUDY_BOXES[a], STUDY_BOXES[b]
        ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
                    arrowprops={"arrowstyle": "-|>", "color": "#777777", "lw": 0.8,
                                "linestyle": "--" if b == "joint" else "-", "shrinkA": 0, "shrinkB": 0,
                                "patchA": boxes[a].get_bbox_patch(), "patchB": boxes[b].get_bbox_patch()})
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "fig1_study_design.png", dpi=DPI)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure_study_design()
    figure_inputs()
    figure_errors_hallucination()
    print("figures in", OUT)


if __name__ == "__main__":
    main()
