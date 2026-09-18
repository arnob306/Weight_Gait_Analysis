from pathlib import Path

import cv2
import numpy as np

from frame_sampling import extract_frames, sample_frame_indices
from skeleton_utils import draw_skeleton, joints_from_row, load_instance_joints

CSV_PATH = "All Joints/Absolute_Data.csv"


def raw_video_path(instance: int) -> str:
    return f"All RGB Videos/Instance_{instance}.0.mp4"


def render_skeleton_video(instance: int, output_path: str) -> None:
    cap = cv2.VideoCapture(raw_video_path(instance))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    inst = load_instance_joints(CSV_PATH, instance)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for _, row in inst.iterrows():
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        joints, valid = joints_from_row(row)
        draw_skeleton(canvas, joints, valid)
        writer.write(canvas)
    writer.release()


def render_overlay_video(instance: int, output_path: str) -> None:
    cap = cv2.VideoCapture(raw_video_path(instance))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    inst = load_instance_joints(CSV_PATH, instance)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for _, row in inst.iterrows():
        ok, frame = cap.read()
        if not ok:
            break
        joints, valid = joints_from_row(row)
        draw_skeleton(frame, joints, valid)
        writer.write(frame)
    cap.release()
    writer.release()


def tile_frames(frames: list[np.ndarray], cols: int) -> np.ndarray:
    rows = (len(frames) + cols - 1) // cols
    h, w = frames[0].shape[:2]
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = frame
    return canvas


def render_montage_image(instance: int, output_path: str, num_tiles: int = 8, cols: int = 4) -> None:
    video_path = raw_video_path(instance)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    indices = sample_frame_indices(total_frames, num_tiles)
    frames = extract_frames(video_path, indices)
    montage = tile_frames(frames, cols)
    cv2.imwrite(output_path, montage)


def build_all_representations(instance: int, out_dir: str) -> dict[str, str]:
    """Renders skeleton/overlay/montage for one instance under out_dir and
    returns a name -> file_path map covering all four representations."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    paths = {
        "raw_video": raw_video_path(instance),
        "skeleton": f"{out_dir}/skeleton.mp4",
        "overlay": f"{out_dir}/overlay.mp4",
        "montage": f"{out_dir}/montage.png",
    }
    render_skeleton_video(instance, paths["skeleton"])
    render_overlay_video(instance, paths["overlay"])
    render_montage_image(instance, paths["montage"])
    return paths
