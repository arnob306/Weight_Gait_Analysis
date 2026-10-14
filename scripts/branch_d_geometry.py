"""
Joint-geometry helpers shared by branch_d_features.py. Pure functions over
the (x, y) pixel coordinates skeleton_utils.joints_from_row already extracts
-- no dependency on which instance or attribute is being computed.

The joint CSV is 2D (pixel row/col plus a detector confidence, not depth --
confirmed against a real row: skeleton_utils.parse_joint keeps only the
first two values). Every measurement below is therefore a 2D image-plane
proxy, same as what the vision-language model itself sees in a sampled
frame. That keeps the Phase 3 comparison fair: model and measurement are
both working from a 2D projection, not comparing a 2D guess against a 3D
ground truth it could never have recovered.
"""
import math

import numpy as np

import skeleton_utils as su

Point = tuple[int, int] | None


def joint_xy(row, idx: int) -> Point:
    """One joint's (x, y) from a CSV row, or None if the detector missed it."""
    joints, valid = su.joints_from_row(row)
    return joints[idx] if valid[idx] else None


def angle_deg(a: Point, b: Point, c: Point) -> float:
    """Angle at vertex b between rays b->a and b->c, in degrees. nan if any
    point is missing or degenerate (b coincides with a or c)."""
    if a is None or b is None or c is None:
        return float("nan")
    ba = np.array(a, dtype=float) - np.array(b, dtype=float)
    bc = np.array(c, dtype=float) - np.array(b, dtype=float)
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm == 0:
        return float("nan")
    cos_angle = np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def distance(a: Point, b: Point) -> float:
    """Euclidean pixel distance, nan if either point is missing."""
    if a is None or b is None:
        return float("nan")
    return float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))


def horizontal_gap(a: Point, b: Point) -> float:
    """Absolute horizontal (x) pixel distance, nan if either point is missing."""
    if a is None or b is None:
        return float("nan")
    return abs(a[0] - b[0])


def vertical_pos(a: Point) -> float:
    """Raw y pixel coordinate (image convention: larger y = lower/closer to
    the ground). nan if missing."""
    return float(a[1]) if a is not None else float("nan")


def reject_outliers(values: list[float], z_threshold: float = 3.5) -> list[float]:
    """Drops points whose modified z-score (median-based, robust to the
    outliers themselves) exceeds z_threshold. 3.5 is the standard Iglewicz
    & Hoaglin cutoff. Falls back to the untouched input if MAD is zero
    (fewer than two distinct values) or rejection would leave under two
    points, so a degenerate series never becomes empty."""
    if len(values) < 2:
        return values
    median = float(np.median(values))
    mad = float(np.median([abs(v - median) for v in values]))
    if mad == 0:
        return values
    kept = [v for v in values if 0.6745 * abs(v - median) / mad <= z_threshold]
    return kept if len(kept) >= 2 else values


def rom(series: list[float]) -> float:
    """Robust range of motion: max - min after dropping outlier frames, not
    strict max - min over everything.

    Pose-detector output has occasional garbage values (a hand or foot
    detected far from where it actually is for a few consecutive frames)
    -- confirmed directly in this dataset: instance 91's right wrist jumps
    to x = -435 relative to the shoulder for three consecutive frames
    (5.8% of that clip's 52 frames) while every neighbouring frame is
    within +/-50. A fixed percentile trim was tried first and failed here
    specifically because the glitch run (5.8%) exceeded a 5% trim, so the
    interpolated cutoff still landed inside the glitch; percentile
    trimming would also quietly shave real extremes off every clean clip
    that has no glitch at all. Median-based outlier rejection (reject_
    outliers) does not have either problem: it removes only points that
    are actually anomalous relative to the clip's own spread, regardless
    of how many frames the glitch spans, and leaves a glitch-free series
    untouched. nan if fewer than two valid values remain."""
    values = [v for v in series if not math.isnan(v)]
    if len(values) < 2:
        return float("nan")
    clean = reject_outliers(values)
    return float(max(clean) - min(clean))


def unwrap_deg(series: list[float]) -> list[float]:
    """Removes spurious +/-360 degree jumps from a series of angles in
    degrees (e.g. atan2 output), so a value oscillating across the +/-180
    boundary reads as continuous instead of swinging through the full
    circle. nan positions are preserved; unwrapping is applied only to the
    run of valid values between them (each run is unwrapped independently,
    since a gap of missing frames breaks any continuity assumption).

    Needed for shoulder_rotation_rom: the shoulder line sits near +/-180
    degrees for a side-on walker (L/R joint labels are anatomical, not
    screen-relative), so small real oscillations there would otherwise be
    read as a near-360-degree swing by rom() -- confirmed directly: every
    instance's raw shoulder_rotation_rom came out within a degree of 358
    before this fix, regardless of how the subject actually moved."""
    result = [float("nan")] * len(series)
    run_start = None
    for i, v in enumerate(series + [float("nan")]):  # sentinel flushes the final run
        if math.isnan(v):
            if run_start is not None:
                run = series[run_start:i]
                unwrapped = np.unwrap(np.radians(run))
                for j, u in enumerate(unwrapped):
                    result[run_start + j] = math.degrees(u)
                run_start = None
        elif run_start is None:
            run_start = i
    return result


def unwrap_axial_deg(series: list[float]) -> list[float]:
    """Like unwrap_deg, but for an axial (undirected line) angle that
    repeats every 180 degrees rather than 360 -- e.g. the orientation of
    the line through two joints, which is the same line whether read
    left-to-right or right-to-left.

    atan2 always returns a directed (360-periodic) angle. When the line
    it measures sits near the point where its direction label flips (the
    horizontal separation between the two points crosses zero), the
    reading jumps by ~180 degrees even though the physical line barely
    moved -- confirmed directly on this data: instance 189's shoulder
    line reads ...174.8, -26.6, 172.9, 163.3, 23.2... frame to frame,
    which a real shoulder line cannot do in one video frame. Plain
    unwrap_deg does not remove this: it only corrects jumps that exceed
    180 degrees, and a label-flip jump sits right at that boundary, so it
    is genuinely ambiguous between "the line rotated a lot" and "the same
    line, opposite label" without axial handling. The standard fix is
    used here: double the angle (turning a 180-degree label flip into a
    full 360-degree wrap), unwrap that as an ordinary circular quantity,
    then halve it back."""
    doubled = [float("nan") if math.isnan(v) else ((2 * v + 180) % 360) - 180 for v in series]
    return [float("nan") if math.isnan(v) else v / 2 for v in unwrap_deg(doubled)]


def detrend_linear(series: list[float]) -> list[float]:
    """Subtracts the best-fit line through the non-nan values, leaving only
    deviation from a steady drift. nan positions pass through unchanged.

    Needed only for shoulder_rotation: unlike every other angle or offset
    attribute here, it is measured against a fixed image axis (the
    assumed walking direction), not against another body landmark, so it
    is the one attribute exposed to the subject's overall orientation
    changing across the clip (e.g. walking somewhat toward the camera
    rather than exactly parallel to it) rather than only to the fast,
    within-stride shoulder-pelvis counter-rotation the attribute is meant
    to capture. Confirmed directly on this data: instance 209's shoulder
    line rotates smoothly and monotonically from 180 to -14 degrees over
    23 frames -- far too large and far too gradual to be stride-to-stride
    counter-rotation, and not a wraparound artifact either (no jump
    exceeds a few degrees between consecutive frames)."""
    idx = [i for i, v in enumerate(series) if not math.isnan(v)]
    if len(idx) < 2:
        return list(series)
    values = [series[i] for i in idx]
    slope, intercept = np.polyfit(idx, values, 1)
    result = list(series)
    for i in idx:
        result[i] = series[i] - (slope * i + intercept)
    return result


def coefficient_of_variation(values: list[float]) -> float:
    """std / mean over non-nan values. nan if fewer than two values or the
    mean is zero (undefined ratio)."""
    clean = [v for v in values if not math.isnan(v)]
    if len(clean) < 2 or np.mean(clean) == 0:
        return float("nan")
    return float(np.std(clean) / abs(np.mean(clean)))


def mean_abs_series_diff(series_a: list[float], series_b: list[float]) -> float:
    """Mean absolute difference between two same-length series at matching
    frame positions (e.g. left vs right joint height) -- a whole-clip
    symmetry statistic that needs no cycle alignment. nan if no frame has
    both values valid."""
    diffs = [
        abs(a - b)
        for a, b in zip(series_a, series_b)
        if not math.isnan(a) and not math.isnan(b)
    ]
    return float(np.mean(diffs)) if diffs else float("nan")


def safe_abs_diff(a: float, b: float) -> float:
    """abs(a - b), or nan if either is nan -- guards ROM-based asymmetry
    computations against propagating a missing value silently as 0."""
    return abs(a - b) if not (math.isnan(a) or math.isnan(b)) else float("nan")


def mean_abs_frame_diff(series: list[float]) -> float:
    """Mean absolute change between consecutive non-nan values -- a
    frame-to-frame variability/jerkiness measure that needs no cycle
    detection. nan if fewer than two consecutive valid pairs exist."""
    diffs = [
        abs(series[i + 1] - series[i])
        for i in range(len(series) - 1)
        if not math.isnan(series[i]) and not math.isnan(series[i + 1])
    ]
    return float(np.mean(diffs)) if diffs else float("nan")
