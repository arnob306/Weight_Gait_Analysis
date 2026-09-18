import cv2
import numpy as np


def sample_frame_indices(total_frames: int, n: int) -> list[int]:
    """Uniformly pick n frame indices spanning [0, total_frames)."""
    if n >= total_frames:
        return list(range(total_frames))
    return [int(i) for i in np.linspace(0, total_frames - 1, n)]


def extract_frames(video_path: str, indices: list[int]) -> list[np.ndarray]:
    """Read the frames at the given indices from a video file, in order."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames
