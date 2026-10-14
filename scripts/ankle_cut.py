"""
Ankle cut: remove the ankle weights (and everything below) from a clip.

The class labels come from ankle weights, and raw video is the only input
that shows them. Cutting the frame at the ankle removes that cue, so a model
that still separates the classes is reading movement, not the weight device.

The cut is ONE fixed row per clip, not a line that follows each frame's ankle.
A line that moved with the ankle would put the foot's height onto the edge of
the black region, handing the model the very foot-clearance signal being
tested. The row sits just above the highest point either ankle reaches
anywhere in the clip, so no weight is visible in any frame.

Usage (from the repo root):
    python scripts/ankle_cut.py            # preview + cut-row summary
"""
from pathlib import Path

import cv2
import numpy as np

import branch_d_geometry as geo
import skeleton_utils as su

CSV_PATH = Path(__file__).resolve().parent.parent / "All Joints" / "Absolute_Data.csv"
PREVIEW_DIR = Path(__file__).resolve().parent.parent / "results" / "ankle_preview"
ANKLE_JOINTS = (15, 16)  # L and R ankle keypoints (named L_foot / R_foot in skeleton_utils)
MARGIN_PX = 6            # clearance above the highest ankle, to cover a weight sitting above it


def ankle_cut_row(csv_path: Path, instance: int, margin: int = MARGIN_PX) -> int:
    """Row at which to cut: the highest ankle position in the clip, minus margin.
    Missed detections (0, 0) are skipped, and outlier frames are dropped with
    the same median-based rule used for the kinematic measurements."""
    rows = su.load_instance_joints(str(csv_path), instance)
    ys = [
        float(p[1])
        for _, row in rows.iterrows()
        for p in (geo.joint_xy(row, j) for j in ANKLE_JOINTS)
        if p is not None
    ]
    if not ys:
        raise ValueError(f"instance {instance}: no ankle detections to place a cut")
    return max(1, int(min(geo.reject_outliers(ys))) - margin)


def apply_ankle_cut(frame: np.ndarray, cut_row: int, mode: str = "mask") -> np.ndarray:
    """mode='mask' blacks out rows from cut_row down, keeping the frame size;
    mode='crop' drops them, giving a shorter frame."""
    cut = int(np.clip(cut_row, 1, frame.shape[0]))
    if mode == "crop":
        return frame[:cut].copy()
    out = frame.copy()
    out[cut:] = 0
    return out


def build_preview(instances: list[int], windows: dict, source_dir: Path) -> None:
    """One image: per clip, mid-window frame with the cut row in red, the masked
    frame, and the cropped frame, side by side."""
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    panels = []
    for inst in instances:
        cap = cv2.VideoCapture(str(source_dir / f"Instance_{inst}.0.mp4"))
        start, end = windows[inst]
        cap.set(cv2.CAP_PROP_POS_FRAMES, (start + end) // 2)
        ok, frame = cap.read()
        cap.release()
        cut = ankle_cut_row(CSV_PATH, inst)
        marked = frame.copy()
        cv2.line(marked, (0, cut), (frame.shape[1], cut), (0, 0, 255), 1)
        cropped = apply_ankle_cut(frame, cut, "crop")
        pad = np.zeros_like(frame)
        pad[: cropped.shape[0]] = cropped
        panels.append(np.hstack([marked, apply_ankle_cut(frame, cut, "mask"), pad]))
    cv2.imwrite(str(PREVIEW_DIR / "preview.png"), np.vstack(panels))


if __name__ == "__main__":
    from run_kaggle_qwen import INSTANCE_META

    source = Path(__file__).resolve().parent.parent / "All RGB Videos"
    cuts = {i: ankle_cut_row(CSV_PATH, i) for i in sorted(INSTANCE_META)}
    removed = [100 * (240 - c) / 240 for c in cuts.values()]
    print(f"cut row over {len(cuts)} clips: min {min(cuts.values())}, "
          f"median {int(np.median(list(cuts.values())))}, max {max(cuts.values())} (frame height 240)")
    print(f"share of the frame removed: min {min(removed):.0f}%, median {np.median(removed):.0f}%, max {max(removed):.0f}%")
    windows = {i: (m["start"], m["end"]) for i, m in INSTANCE_META.items()}
    build_preview([785, 91, 825], windows, source)
    print(f"preview written to {PREVIEW_DIR / 'preview.png'}")
