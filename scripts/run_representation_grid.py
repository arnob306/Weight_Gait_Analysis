import os
from pathlib import Path

import cv2
from dotenv import load_dotenv
from google import genai

from experiment import NUM_SAMPLED_FRAMES, PROMPT, log_result
from frame_sampling import extract_frames, sample_frame_indices
from harness import classify_clip, classify_frames_gemini
from labels import get_instance_label, load_labels
from skeleton_utils import active_frame_range, load_instance_joints

load_dotenv()

INSTANCE = 12
MODEL = "gemini-3.5-flash"
CSV_PATH = "All Joints/Absolute_Data.csv"
RESULTS_DIR = Path("results/raw_responses")

# raw_video/skeleton/overlay are frame sequences -- sampled to NUM_SAMPLED_FRAMES
# so Gemini sees the same frame budget as the Kaggle/Qwen backend, not the whole clip.
VIDEO_REPRESENTATIONS = {
    "raw_video": f"All RGB Videos/Instance_{INSTANCE}.0.mp4",
    "skeleton": f"skeleton_Instance_{INSTANCE}.0.mp4",
    "overlay": f"overlay_Instance_{INSTANCE}.0.mp4",
}
# montage is already a single fixed image (8 tiled frames), sent as-is.
IMAGE_REPRESENTATIONS = {
    "montage": f"montage_Instance_{INSTANCE}.0.png",
}


def report(representation: str, file_path: str, result: dict, ground_truth: dict) -> None:
    log_result(RESULTS_DIR, MODEL, INSTANCE, representation, file_path, result, ground_truth)
    if result["success"]:
        predicted = result["parsed"]["weight_class"]
        mark = "correct" if predicted == ground_truth["weight_class"] else "WRONG"
        print(f"[{representation}] predicted={predicted} ({mark}) -- {result['parsed']['reasoning']}")
    else:
        print(f"[{representation}] FAILED -- {result['error']}")
        print(f"    raw response: {result['raw_response']!r}")


def main() -> None:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    labels = load_labels(CSV_PATH)
    ground_truth = get_instance_label(labels, INSTANCE)
    print(f"Instance {INSTANCE} ground truth: {ground_truth}")

    instance_joints = load_instance_joints(CSV_PATH, INSTANCE)
    start, end = active_frame_range(instance_joints)

    for representation, file_path in VIDEO_REPRESENTATIONS.items():
        if not Path(file_path).exists():
            print(f"[{representation}] SKIPPED -- file not found: {file_path}")
            continue

        cap = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        indices = sample_frame_indices(total_frames, NUM_SAMPLED_FRAMES, start=start, end=end)
        frames = extract_frames(file_path, indices)

        result = classify_frames_gemini(client, frames, PROMPT, model=MODEL)
        report(representation, file_path, result, ground_truth)

    for representation, file_path in IMAGE_REPRESENTATIONS.items():
        if not Path(file_path).exists():
            print(f"[{representation}] SKIPPED -- file not found: {file_path}")
            continue

        result = classify_clip(client, file_path, PROMPT, model=MODEL)
        report(representation, file_path, result, ground_truth)


if __name__ == "__main__":
    main()
