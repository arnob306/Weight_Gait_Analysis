import cv2
import numpy as np

from frame_sampling import extract_frames, sample_frame_indices

INSTANCE = 12
RAW_VIDEO_PATH = f"All RGB Videos/Instance_{INSTANCE}.0.mp4"
OUTPUT_PATH = f"montage_Instance_{INSTANCE}.0.png"
NUM_TILES = 8
GRID_COLS = 4


def build_montage(frames: list[np.ndarray], cols: int) -> np.ndarray:
    """Tile frames left-to-right, top-to-bottom into a single grid image."""
    rows = (len(frames) + cols - 1) // cols
    h, w = frames[0].shape[:2]
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = frame
    return canvas


cap = cv2.VideoCapture(RAW_VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

indices = sample_frame_indices(total_frames, NUM_TILES)
frames = extract_frames(RAW_VIDEO_PATH, indices)
montage = build_montage(frames, GRID_COLS)
cv2.imwrite(OUTPUT_PATH, montage)
print(f"Saved gait-cycle montage ({len(frames)} frames, indices {indices}) to {OUTPUT_PATH}")
