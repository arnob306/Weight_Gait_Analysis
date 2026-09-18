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


def draw_skeleton(canvas: np.ndarray, joints: list[tuple[int, int]], valid: list[bool]) -> None:
    """Draw skeleton edges and joint markers onto canvas in place."""
    for a, b in SKELETON_EDGES:
        if valid[a] and valid[b]:
            cv2.line(canvas, joints[a], joints[b], (255, 255, 255), 2)
    for (x, y), ok in zip(joints, valid):
        if ok:
            cv2.circle(canvas, (x, y), 4, (0, 200, 255), -1)
