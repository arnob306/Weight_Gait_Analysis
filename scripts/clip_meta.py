"""
Per-clip facts the Gemma v2 prompts need, computed once locally from the joint CSV
and shipped to the notebook as clip_meta.json (the notebook has no 41 MB label CSV).

For each instance:
  window      [first, last] stored frames the v2 runs sample from: the active window
              from active_frame_range, except in the clips where the walker leaves the
              frame and returns inside it (8 of the 60 seeded clips); there it is the
              longest stretch where the hips are seen, i.e. one pass, no empty frames.
  direction  "left" or "right": the way the walker moves across the image, from
              the slope of the hip x position over the active window. None if the
              walker barely moves (under MIN_DRIFT_PX), so the prompt can leave the
              sentence out. The view is side-on and both directions occur.
  near_leg    the walker's leg nearer the camera: a walker moving toward the image
              right shows the camera their right side, toward the left their left
              side. Checked on class-1 clips, where the weighted (left) leg is the
              near leg in the clips that move left.
  cx          per stored frame, the x pixel of the walker's centre (mean of the
              valid joints), used to crop every frame around the walker.
  freeze, obstacle, weight_class   ground truth from the dataset labels.
"""
import numpy as np

HIP_JOINTS = (11, 12)  # L_hip, R_hip in skeleton_utils.JOINT_NAMES
MIN_DRIFT_PX = 30
FALLBACK_CX = 212  # image centre of the 424 px wide frames
MAX_GAP = 3        # missing frames bridged when looking for one continuous pass
MIN_COVERAGE = 0.9  # share of the active window with the hips seen, below which it is replaced


def walking_direction(hip_x: list[float | None], min_drift: float = MIN_DRIFT_PX) -> str | None:
    """'left' or 'right' from the slope of hip x over the frames where a hip was seen."""
    pts = [(k, x) for k, x in enumerate(hip_x) if x is not None]
    if len(pts) < 2:
        return None
    ks, xs = np.array(pts, dtype=float).T
    slope = np.polyfit(ks, xs, 1)[0]
    if abs(slope) * (ks[-1] - ks[0]) < min_drift:
        return None
    return "right" if slope > 0 else "left"


def near_leg_for(direction: str | None) -> str | None:
    return direction  # moving toward the image right shows the walker's right side, and likewise for left


def centre_series(per_frame_joints: list[tuple[list[tuple[int, int]], list[bool]]]) -> list[int]:
    """x of the walker's centre per frame (mean of valid joints), carried forward
    over frames with no detection, FALLBACK_CX before the first detection."""
    out, last = [], FALLBACK_CX
    for joints, valid in per_frame_joints:
        xs = [x for (x, _), ok in zip(joints, valid) if ok]
        last = int(round(float(np.mean(xs)))) if xs else last
        out.append(last)
    return out


def hip_series(per_frame_joints, window: tuple[int, int]) -> list[float | None]:
    start, end = window
    out = []
    for joints, valid in per_frame_joints[start:end + 1]:
        xs = [joints[j][0] for j in HIP_JOINTS if valid[j]]
        out.append(float(np.mean(xs)) if xs else None)
    return out


def longest_run(present: list[bool], max_gap: int = MAX_GAP) -> tuple[int, int]:
    """(first, last) frame of the longest stretch where the walker is seen, bridging
    gaps of up to max_gap missing frames. Whole range if the walker is never seen."""
    best, start, last = None, None, None
    for k, seen in enumerate(present + [False] * (max_gap + 2)):
        if seen:
            start = k if start is None else start
            last = k
        elif start is not None and k - last > max_gap:
            if best is None or last - start > best[1] - best[0]:
                best = (start, last)
            start = None
    return best if best else (0, max(len(present) - 1, 0))


def walking_window(per_frame_joints, active: tuple[int, int]) -> tuple[int, int]:
    """The active window, unless the walker leaves the frame and comes back inside it
    (hips seen in under MIN_COVERAGE of its frames); then the longest stretch where
    the hips are seen, so the images show one pass and no empty frames."""
    hips = hip_series(per_frame_joints, (0, len(per_frame_joints) - 1))
    start, end = active
    seen = sum(h is not None for h in hips[start:end + 1])
    if seen >= MIN_COVERAGE * (end - start + 1):
        return active
    return longest_run([h is not None for h in hips])


def build_clip_meta(instance: int, csv_path: str, labels) -> dict:
    """The record for one instance (needs the joint CSV, so it runs locally)."""
    import skeleton_utils as su
    from labels import get_instance_label

    frames = su.load_instance_joints(csv_path, instance)
    per_frame = [su.joints_from_row(row) for _, row in frames.iterrows()]
    window = walking_window(per_frame, su.active_frame_range(frames))
    direction = walking_direction(hip_series(per_frame, window))
    label = get_instance_label(labels, instance)
    return {
        "window": list(window),
        "direction": direction,
        "near_leg": near_leg_for(direction),
        "cx": centre_series(per_frame),
        "weight_class": label["weight_class"],
        "freeze": label["gait_freeze"],
        "obstacle": label["obstacle"],
    }
