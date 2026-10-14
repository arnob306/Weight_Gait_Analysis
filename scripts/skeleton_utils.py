import cv2
import numpy as np
import pandas as pd

JOINT_NAMES = ['Nose', 'L_eye', 'R_eye', 'L_ear', 'R_ear', 'L_shoulder', 'R_shoulder',
               'L_elbow', 'R_elbow', 'L_hand', 'R_hand', 'L_hip', 'R_hip',
               'L_knee', 'R_knee', 'L_foot', 'R_foot']

SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

NUM_JOINTS = len(JOINT_NAMES)


def parse_joint(cell: str) -> tuple[int, int]:
    """Parse a "[row, col, conf]" CSV cell into an (x, y) pixel coordinate."""
    vals = cell.strip("[]").split(",")
    row, col = float(vals[0]), float(vals[1])
    return int(col), int(row)


def load_instance_joints(csv_path: str, instance: int) -> pd.DataFrame:
    """Load and frame-order the joint rows for one video instance."""
    df = pd.read_csv(csv_path, header=None)
    return df[df[0] == instance].sort_values(by=1)


def joints_from_row(row: pd.Series) -> tuple[list[tuple[int, int]], list[bool]]:
    """Extract per-joint (x, y) coordinates and validity flags from one CSV row."""
    joints = [parse_joint(row[6 + j]) for j in range(NUM_JOINTS)]
    valid = [not (x == 0 and y == 0) for (x, y) in joints]
    return joints, valid


def active_frame_range(instance_joints: pd.DataFrame) -> tuple[int, int]:
    """Returns the (first, last) 0-indexed frame positions where the subject
    has at least one detected joint. Many clips have the person in frame for
    only part of the recording (walking in from off-camera, or leaving frame
    before the clip ends) -- sampling should stay within this range."""
    detected = [any(joints_from_row(row)[1]) for _, row in instance_joints.iterrows()]
    positions = [i for i, ok in enumerate(detected) if ok]
    return (positions[0], positions[-1]) if positions else (0, len(instance_joints) - 1)


BLUE, RED, GREY = (255, 80, 0), (0, 0, 255), (170, 170, 170)  # BGR: left limbs, right limbs, head and trunk
_LEFT_LIMB = {7, 9, 13, 15}   # left elbow, hand, knee, foot
_RIGHT_LIMB = {8, 10, 14, 16}
_LEFT_EDGES = {(5, 7), (7, 9), (11, 13), (13, 15)}
_RIGHT_EDGES = {(6, 8), (8, 10), (12, 14), (14, 16)}


def draw_skeleton(canvas: np.ndarray, joints: list[tuple[int, int]], valid: list[bool],
                  by_side: bool = False) -> np.ndarray:
    """Returns a new array with skeleton edges and joint markers drawn over canvas.

    by_side=False is the original white-and-orange drawing. by_side=True draws the
    walker's left arm and leg in blue, the right arm and leg in red, and the head
    and trunk in grey (the colours the Gemma prompts describe); the shoulders and
    hips count as trunk, so each limb's colour starts at its elbow or knee."""
    canvas = canvas.copy()
    for a, b in SKELETON_EDGES:
        if valid[a] and valid[b]:
            color = (255, 255, 255)
            if by_side:
                color = BLUE if (a, b) in _LEFT_EDGES else RED if (a, b) in _RIGHT_EDGES else GREY
            cv2.line(canvas, joints[a], joints[b], color, 2)
    for k, ((x, y), ok) in enumerate(zip(joints, valid)):
        if ok:
            color = (0, 200, 255)
            if by_side:
                color = BLUE if k in _LEFT_LIMB else RED if k in _RIGHT_LIMB else GREY
            cv2.circle(canvas, (x, y), 4, color, -1)
    return canvas
