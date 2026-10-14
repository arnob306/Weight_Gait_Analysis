import cv2
import numpy as np


def sample_frame_indices(total_frames: int, n: int, start: int = 0, end: int | None = None) -> list[int]:
    """Uniformly pick n frame indices spanning [start, end] (end defaults to total_frames - 1).

    Restricting to [start, end] matters when the subject isn't in frame for
    the whole clip -- sampling the full range wastes most of the budget on
    empty frames instead of the walk itself.
    """
    end = total_frames - 1 if end is None else end
    span = end - start + 1
    if n >= span:
        return list(range(start, end + 1))
    return [start + int(i) for i in np.linspace(0, span - 1, n)]


def sample_frame_indices_fps(
    native_fps: float, target_fps: float, start: int, end: int, min_frames: int = 2
) -> list[int]:
    """Equally spaced frame indices at target_fps over [start, end], on the
    stored video's own timeline (these clips are stored at 7 fps).

    Samples sit step = native_fps / target_fps source frames apart. When that
    is fractional (3 fps on a 7 fps clip -> 2.33), gaps alternate between two
    neighbouring integers, so spacing is equal to within one source frame and
    never consecutive. Windows too short to yield min_frames fall back to
    min_frames evenly spread across the window so every clip gets a sequence.
    """
    if target_fps > native_fps:
        raise ValueError(f"target_fps {target_fps} exceeds the stored {native_fps} fps; frames would repeat")
    span = end - start + 1
    step = native_fps / target_fps
    count = int((span - 1) / step) + 1
    if count < min_frames:
        n = min(min_frames, span)
        return [start + int(round(i)) for i in np.linspace(0, span - 1, n)]
    return [start + int(round(k * step)) for k in range(count)]


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
