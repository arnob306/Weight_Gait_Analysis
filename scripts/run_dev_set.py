import json
import os
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv
from google import genai

from experiment import NUM_SAMPLED_FRAMES, PROMPT, log_result
from frame_sampling import extract_frames, sample_frame_indices
from harness import classify_clip, classify_frames_gemini
from labels import get_instance_label, load_labels, select_dev_set
from representations import build_all_representations
from skeleton_utils import active_frame_range, load_instance_joints

load_dotenv()

CSV_PATH = "All Joints/Absolute_Data.csv"
MODEL = "gemini-3.8-flash"
# gemini-3.5-flash's free-tier daily quota (20 requests/day on this key) was
# exhausted by this same dev-set run -- confirmed via the API's own 429
# RESOURCE_EXHAUSTED response, not assumed. gemini-3.8-flash is Google's own
# suggested replacement (returned in a 404 message from the now-deprecated
# gemini-2.5-flash) and has its own separate, untouched daily quota, since
# Google scopes this quota per model, not per API key.
N_PER_CLASS = 2
SEED = 42
RENDER_DIR = "renders"
RESULTS_DIR = Path("results/raw_responses")
SUMMARY_PATH = Path("results/dev_set_summary.json")
# Conservative pacing carried over from gemini-3.5-flash's free tier (5
# requests/minute); gemini-3.8-flash's exact rate limit hasn't been confirmed,
# so this stays as a safe default rather than a guessed faster pace.
SECONDS_BETWEEN_CALLS = 13


def classify_representation(client, file_path: str, model: str, start: int, end: int) -> dict:
    """Montage is already a single fixed image, sent as-is. The three video
    representations are sampled to NUM_SAMPLED_FRAMES within [start, end] --
    the range where the subject is actually in frame -- so Gemini sees the
    same frame budget as the Kaggle/Qwen backend, not the whole clip (most of
    which can be empty before/after the walk)."""
    if file_path.endswith(".png"):
        return classify_clip(client, file_path, PROMPT, model=model)

    cap = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    indices = sample_frame_indices(total_frames, NUM_SAMPLED_FRAMES, start=start, end=end)
    frames = extract_frames(file_path, indices)
    return classify_frames_gemini(client, frames, PROMPT, model=model)


def summarize(records: list[dict], representations: list[str]) -> dict:
    summary = {}
    for representation in representations:
        scored = [r for r in records if r["representation"] == representation and r["correct"] is not None]
        n_correct = sum(r["correct"] for r in scored)
        summary[representation] = {
            "n": len(scored),
            "correct": n_correct,
            "accuracy": n_correct / len(scored) if scored else None,
        }
    return summary


def main() -> None:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    labels = load_labels(CSV_PATH)
    dev_instances = select_dev_set(labels, n_per_class=N_PER_CLASS, seed=SEED)
    print(f"Dev set ({len(dev_instances)} instances, {N_PER_CLASS} per class): {dev_instances}")

    all_records = []
    representations = []
    for instance in dev_instances:
        ground_truth = get_instance_label(labels, instance)
        instance_joints = load_instance_joints(CSV_PATH, instance)
        start, end = active_frame_range(instance_joints)
        paths = build_all_representations(instance, f"{RENDER_DIR}/{instance}", instance_joints=instance_joints)
        representations = list(paths.keys())

        for representation, file_path in paths.items():
            result = classify_representation(client, file_path, MODEL, start, end)
            record = log_result(RESULTS_DIR, MODEL, instance, representation, file_path, result, ground_truth)
            all_records.append(record)

            predicted = record["parsed"]["weight_class"] if record["success"] else None
            mark = "OK" if record["correct"] else "WRONG" if record["correct"] is not None else "FAILED"
            print(f"[{instance}][{representation}] gt={ground_truth['weight_class']} pred={predicted} ({mark})")
            time.sleep(SECONDS_BETWEEN_CALLS)

    summary = summarize(all_records, representations)
    print("\nAccuracy by representation:")
    for representation, stats in summary.items():
        if stats["accuracy"] is None:
            print(f"  {representation}: no scored results")
        else:
            print(f"  {representation}: {stats['correct']}/{stats['n']} = {stats['accuracy']:.2f}")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps({"dev_instances": dev_instances, "accuracy_by_representation": summary}, indent=2))


if __name__ == "__main__":
    main()
