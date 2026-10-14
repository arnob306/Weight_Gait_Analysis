"""
Branch D, Phase 3 -- measured side (a stand-in for Branch B; see
docs/branch_d_guideline.md section 9). Computes all 27 attributes from
docs/branch_d_guideline.md section 7 directly from the joint CSV, over the
same active frame range (INSTANCE_META start/end) the VLM was shown in
Branch C and Phase 1 elicitation. These are the values Phase 1's elicited
ratings get checked against (Spearman rank correlation, in
branch_d_agreement.py) -- rank correlation is unit-free, so pixel units and
frame-count units below are never converted to real-world scale.

Operationalization choices (the guideline's 27-row table names each
attribute but not a formula; these are the formulas used here):

- ROM-based asymmetries (knee/hip angle, foot clearance, arm swing) are
  |left ROM - right ROM|, not a per-frame difference. Per-frame L vs R
  comparison would need gait-cycle phase alignment (legs move in
  antiphase), which section 6 of the guideline already flags as
  unavailable without cycle segmentation.
- Height-based asymmetries (elbow, shoulder) are the mean absolute
  per-frame difference instead, since "height" is a single per-frame
  quantity with no ROM to compare.
- Step width is the median horizontal gap between the knees. In a
  monocular side-view clip, true (depth-axis) width is not directly
  visible; the horizontal gap is a proxy confounded with the walking-
  direction step motion, same limitation the original spec already
  labels "proxy" for step length and width.
- Peak knee flexion timing is the single global peak's position as a
  fraction of the active window, not a per-cycle average -- most clips
  are short enough to show at most one full stride (see stride_time
  below), so per-cycle averaging has nothing to average over.
- Stride time and step length variability need at least two detected
  cycle events (scipy.signal.find_peaks) and are nan otherwise. This is
  expected to be nan for many of the shorter clips -- itself a finding
  about how little a short clip constrains a temporal measure, worth
  reporting rather than working around.
- Every ROM is a robust (percentile-trimmed) range, not strict max - min
  -- see branch_d_geometry.rom for why: a single glitched detection would
  otherwise dominate the whole clip's reading, confirmed directly on this
  data (instance 91's right wrist). shoulder_rotation_rom additionally
  unwraps the angle series first, since the shoulder line sits near the
  +/-180 degree boundary for a side-on walker and would otherwise read a
  small real oscillation as a near-360-degree swing -- also confirmed
  directly (every instance's raw value was within a degree of 358 before
  this fix).
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

import branch_d_geometry as geo
import skeleton_utils as su
from run_kaggle_qwen import INSTANCE_META

CSV_PATH = Path(__file__).resolve().parent.parent / "All Joints" / "Absolute_Data.csv"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "results" / "branch_d_measured_features.json"

# JOINT_NAMES index shortcuts (skeleton_utils.JOINT_NAMES order)
(NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW,
 L_HAND, R_HAND, L_HIP, R_HIP, L_KNEE, R_KNEE, L_FOOT, R_FOOT) = range(17)

MIN_APEX_FRAMES = 4  # fewest valid frames before peak-finding is attempted at all


def load_active_rows(csv_path: Path, instance: int, start: int, end: int):
    """Joint rows for one instance, restricted to the positional [start, end]
    window -- the same window frame_sampling.sample_frame_indices draws
    from, so the measured side reflects the part of the clip the model
    actually saw."""
    df = su.load_instance_joints(str(csv_path), instance)
    return df.iloc[start:end + 1]


def joint_series(rows, idx: int) -> list:
    return [geo.joint_xy(row, idx) for _, row in rows.iterrows()]


def angle_series(rows, a_idx: int, b_idx: int, c_idx: int) -> list[float]:
    a, b, c = joint_series(rows, a_idx), joint_series(rows, b_idx), joint_series(rows, c_idx)
    return [geo.angle_deg(a[i], b[i], c[i]) for i in range(len(a))]


def vertical_series(rows, idx: int) -> list[float]:
    return [geo.vertical_pos(p) for p in joint_series(rows, idx)]


def horizontal_gap_series(rows, a_idx: int, b_idx: int) -> list[float]:
    a, b = joint_series(rows, a_idx), joint_series(rows, b_idx)
    return [geo.horizontal_gap(a[i], b[i]) for i in range(len(a))]


def rel_x_series(rows, moving_idx: int, anchor_idx: int) -> list[float]:
    """moving joint's x position minus anchor joint's x position, per frame
    -- isolates a limb's own excursion from the body's overall translation
    across the frame during walking."""
    moving, anchor = joint_series(rows, moving_idx), joint_series(rows, anchor_idx)
    return [
        (moving[i][0] - anchor[i][0]) if moving[i] is not None and anchor[i] is not None else float("nan")
        for i in range(len(moving))
    ]


def midpoint_x_series(rows, a_idx: int, b_idx: int) -> list[float]:
    a, b = joint_series(rows, a_idx), joint_series(rows, b_idx)
    return [
        (a[i][0] + b[i][0]) / 2 if a[i] is not None and b[i] is not None else float("nan")
        for i in range(len(a))
    ]


def shoulder_line_angle_series(rows) -> list[float]:
    """Angle of the L-shoulder -> R-shoulder line relative to image
    horizontal (the assumed walking direction in a side-view clip), per
    frame -- shoulder-pelvis counter-rotation shows up as this angle
    changing over the stride."""
    left, right = joint_series(rows, L_SHOULDER), joint_series(rows, R_SHOULDER)
    out = []
    for l, r in zip(left, right):
        if l is None or r is None:
            out.append(float("nan"))
        else:
            out.append(math.degrees(math.atan2(r[1] - l[1], r[0] - l[0])))
    return out


def stance_fraction(foot_y: list[float]) -> float:
    """Fraction of valid frames where the foot sits in the bottom quartile
    of its own vertical range for this clip (near the ground; larger y =
    lower in image coordinates). nan if the foot's range is degenerate."""
    clean = [v for v in foot_y if not math.isnan(v)]
    if len(clean) < 2:
        return float("nan")
    lo, hi = min(clean), max(clean)
    if hi == lo:
        return float("nan")
    threshold = hi - 0.25 * (hi - lo)
    return sum(v >= threshold for v in clean) / len(clean)


def swing_apex_frame_gap(foot_y: list[float]) -> float:
    """Median frame gap between swing-apex events (local minima in y, i.e.
    the foot at its highest point) -- a stride-time proxy in frame units.
    nan if fewer than two apex events are detected."""
    valid_idx = [i for i, v in enumerate(foot_y) if not math.isnan(v)]
    if len(valid_idx) < MIN_APEX_FRAMES:
        return float("nan")
    values = np.array([foot_y[i] for i in valid_idx])
    peaks, _ = find_peaks(-values)
    if len(peaks) < 2:
        return float("nan")
    frame_positions = np.array(valid_idx)[peaks]
    return float(np.median(np.diff(frame_positions)))


def peak_timing_fraction(angle_values: list[float]) -> float:
    """Position of the single largest value in the series, as a fraction of
    the active window (0 = start, 1 = end). nan if fewer than two valid
    values exist."""
    clean = [(i, v) for i, v in enumerate(angle_values) if not math.isnan(v)]
    if len(clean) < 2 or len(angle_values) < 2:
        return float("nan")
    peak_i = max(clean, key=lambda iv: iv[1])[0]
    return peak_i / (len(angle_values) - 1)


def step_peak_variability(foot_gap: list[float]) -> float:
    """Coefficient of variation across detected step-length peaks (local
    maxima of the foot-to-foot horizontal gap). nan if fewer than two
    steps are detected."""
    valid_idx = [i for i, v in enumerate(foot_gap) if not math.isnan(v)]
    if len(valid_idx) < MIN_APEX_FRAMES:
        return float("nan")
    values = np.array([foot_gap[i] for i in valid_idx])
    peaks, _ = find_peaks(values)
    if len(peaks) < 2:
        return float("nan")
    return geo.coefficient_of_variation(list(values[peaks]))


def compute_instance_features(csv_path: Path, instance: int) -> dict:
    meta = INSTANCE_META[instance]
    rows = load_active_rows(csv_path, instance, meta["start"], meta["end"])

    knee_L, knee_R = angle_series(rows, L_HIP, L_KNEE, L_FOOT), angle_series(rows, R_HIP, R_KNEE, R_FOOT)
    hip_L, hip_R = angle_series(rows, L_SHOULDER, L_HIP, L_KNEE), angle_series(rows, R_SHOULDER, R_HIP, R_KNEE)
    elbow_L, elbow_R = angle_series(rows, L_SHOULDER, L_ELBOW, L_HAND), angle_series(rows, R_SHOULDER, R_ELBOW, R_HAND)

    foot_gap = horizontal_gap_series(rows, L_FOOT, R_FOOT)
    knee_gap = horizontal_gap_series(rows, L_KNEE, R_KNEE)
    foot_y_L, foot_y_R = vertical_series(rows, L_FOOT), vertical_series(rows, R_FOOT)
    shoulder_y_L, shoulder_y_R = vertical_series(rows, L_SHOULDER), vertical_series(rows, R_SHOULDER)
    elbow_y_L, elbow_y_R = vertical_series(rows, L_ELBOW), vertical_series(rows, R_ELBOW)

    trunk_offset = [
        sm - hm if not (math.isnan(sm) or math.isnan(hm)) else float("nan")
        for sm, hm in zip(midpoint_x_series(rows, L_SHOULDER, R_SHOULDER), midpoint_x_series(rows, L_HIP, R_HIP))
    ]
    arm_L, arm_R = rel_x_series(rows, L_HAND, L_SHOULDER), rel_x_series(rows, R_HAND, R_SHOULDER)
    shoulder_rotation = shoulder_line_angle_series(rows)

    knee_rom_L, knee_rom_R = geo.rom(knee_L), geo.rom(knee_R)
    hip_rom_L, hip_rom_R = geo.rom(hip_L), geo.rom(hip_R)
    foot_rom_L, foot_rom_R = geo.rom(foot_y_L), geo.rom(foot_y_R)
    arm_rom_L, arm_rom_R = geo.rom(arm_L), geo.rom(arm_R)
    elbow_rom_L, elbow_rom_R = geo.rom(elbow_L), geo.rom(elbow_R)

    stride_time = swing_apex_frame_gap(foot_y_L)
    if math.isnan(stride_time):
        stride_time = swing_apex_frame_gap(foot_y_R)

    clean_foot_gap = [v for v in foot_gap if not math.isnan(v)]
    clean_knee_gap = [v for v in knee_gap if not math.isnan(v)]

    return {
        "knee_flexion_rom_left": knee_rom_L,
        "knee_flexion_rom_right": knee_rom_R,
        "hip_flexion_rom_left": hip_rom_L,
        "hip_flexion_rom_right": hip_rom_R,
        "knee_angle_asymmetry": geo.safe_abs_diff(knee_rom_L, knee_rom_R),
        "hip_angle_asymmetry": geo.safe_abs_diff(hip_rom_L, hip_rom_R),
        "step_length": max(clean_foot_gap, default=float("nan")),
        "step_width": float(np.median(clean_knee_gap)) if clean_knee_gap else float("nan"),
        "foot_vertical_rom_left": foot_rom_L,
        "foot_vertical_rom_right": foot_rom_R,
        "foot_clearance_asymmetry": geo.safe_abs_diff(foot_rom_L, foot_rom_R),
        "stance_fraction_left": stance_fraction(foot_y_L),
        "stance_fraction_right": stance_fraction(foot_y_R),
        "stance_swing_asymmetry": geo.safe_abs_diff(stance_fraction(foot_y_L), stance_fraction(foot_y_R)),
        "peak_knee_flexion_timing_left": peak_timing_fraction(knee_L),
        "peak_knee_flexion_timing_right": peak_timing_fraction(knee_R),
        "stride_time": stride_time,
        "step_length_variability": step_peak_variability(foot_gap),
        "knee_angle_variability": float(np.nanmean(
            [geo.mean_abs_frame_diff(knee_L), geo.mean_abs_frame_diff(knee_R)]
        )),
        "trunk_lean_rom": geo.rom(trunk_offset),
        "arm_swing_amplitude_left": arm_rom_L,
        "arm_swing_amplitude_right": arm_rom_R,
        "arm_swing_asymmetry": geo.safe_abs_diff(arm_rom_L, arm_rom_R),
        "elbow_flexion_rom": float(np.nanmean([elbow_rom_L, elbow_rom_R])),
        "elbow_height_difference": geo.mean_abs_series_diff(elbow_y_L, elbow_y_R),
        "shoulder_height_asymmetry": geo.mean_abs_series_diff(shoulder_y_L, shoulder_y_R),
        "shoulder_rotation_rom": geo.rom(geo.detrend_linear(geo.unwrap_axial_deg(shoulder_rotation))),
    }


def run(csv_path: Path = CSV_PATH, output_path: Path = OUTPUT_PATH) -> list[dict]:
    records = []
    for instance in sorted(INSTANCE_META):
        features = compute_instance_features(csv_path, instance)
        n_nan = sum(1 for v in features.values() if isinstance(v, float) and math.isnan(v))
        records.append({
            "instance": instance,
            "weight_class": INSTANCE_META[instance]["weight_class"],
            "features": features,
        })
        print(f"[{instance}] {len(features) - n_nan}/{len(features)} features computed")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2, allow_nan=True))
    print(f"Wrote {len(records)} instances to {output_path}")
    return records


if __name__ == "__main__":
    run()
