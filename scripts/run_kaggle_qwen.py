"""
Driver for the full Kaggle/Qwen3-VL-4B run over the 30-instance, balanced
evaluation set (10 per weight_class), 4 representations each -- raw_video,
skeleton, overlay, montage. Not runnable in this repo's environment; it
needs the model weights and a GPU. Meant to run inside a Kaggle notebook.

Kaggle setup:
1. Settings > Accelerator > GPU T4 x2 (or P100).
2. Upload the `kaggle_representations/` folder (30 instance subfolders, each
   containing raw_video.mp4, skeleton.mp4, overlay.mp4, montage.png) as a
   Kaggle Dataset, e.g. named "gait-kaggle-representations".
3. Upload frame_sampling.py, harness.py, experiment.py, kaggle_qwen_backend.py,
   and this file as a second Dataset (or a Kaggle Utility Script), e.g.
   "gait-harness-code".
4. Add both datasets to the notebook, then run:
       !pip install -q transformers accelerate qwen-vl-utils
       import sys
       sys.path.insert(0, "/kaggle/input/gait-harness-code")
       import run_kaggle_qwen
       run_kaggle_qwen.main()

Ground truth (weight_class) and each clip's active frame range (start, end --
the frame positions where the subject is actually in shot) are baked in below
so this script has no dependency on the 41 MB label CSV or skeleton_utils.py;
both were precomputed locally from All Joints/Absolute_Data.csv.
"""
import json
import time
from pathlib import Path

import cv2

from experiment import PROMPT, NUM_SAMPLED_FRAMES, log_result
from frame_sampling import extract_frames, sample_frame_indices
from harness import classify_frames_local

INPUT_DIR = Path("/kaggle/input/gait-kaggle-representations")
RESULTS_DIR = Path("/kaggle/working/results/raw_responses")
SUMMARY_PATH = Path("/kaggle/working/results/kaggle_qwen_summary.json")
MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"

# instance -> {weight_class, start, end}, precomputed from Absolute_Data.csv
INSTANCE_META = {
    71: {"weight_class": 0, "start": 5, "end": 38},
    91: {"weight_class": 1, "start": 16, "end": 67},
    111: {"weight_class": 2, "start": 5, "end": 41},
    189: {"weight_class": 0, "start": 5, "end": 25},
    209: {"weight_class": 1, "start": 5, "end": 27},
    229: {"weight_class": 2, "start": 5, "end": 34},
    438: {"weight_class": 0, "start": 5, "end": 22},
    458: {"weight_class": 1, "start": 5, "end": 23},
    478: {"weight_class": 2, "start": 5, "end": 24},
    494: {"weight_class": 0, "start": 5, "end": 32},
    514: {"weight_class": 1, "start": 6, "end": 34},
    534: {"weight_class": 2, "start": 5, "end": 78},
    727: {"weight_class": 0, "start": 5, "end": 21},
    747: {"weight_class": 1, "start": 5, "end": 25},
    767: {"weight_class": 2, "start": 5, "end": 24},
    785: {"weight_class": 0, "start": 5, "end": 28},
    795: {"weight_class": 0, "start": 5, "end": 28},
    805: {"weight_class": 1, "start": 5, "end": 26},
    815: {"weight_class": 1, "start": 5, "end": 28},
    825: {"weight_class": 2, "start": 5, "end": 27},
    835: {"weight_class": 2, "start": 5, "end": 29},
    851: {"weight_class": 0, "start": 5, "end": 31},
    871: {"weight_class": 1, "start": 5, "end": 35},
    891: {"weight_class": 2, "start": 5, "end": 40},
    1690: {"weight_class": 0, "start": 5, "end": 67},
    1710: {"weight_class": 1, "start": 5, "end": 35},
    1730: {"weight_class": 2, "start": 5, "end": 41},
    1757: {"weight_class": 0, "start": 5, "end": 79},
    1777: {"weight_class": 1, "start": 5, "end": 37},
    1797: {"weight_class": 2, "start": 5, "end": 36},
}

REPRESENTATIONS = ["raw_video", "skeleton", "overlay", "montage"]


def classify_representation(instance: int, representation: str, start: int, end: int) -> tuple[str, dict]:
    """montage.png is already a fixed single image, sent as one frame. The
    three video representations are sampled to NUM_SAMPLED_FRAMES within
    [start, end] -- matching exactly what the Gemini dev-set run used, so
    the two models see the same frame budget per clip."""
    from kaggle_qwen_backend import generate_fn  # local import: needs torch, only required when this runs on Kaggle

    instance_dir = INPUT_DIR / str(instance)

    if representation == "montage":
        file_path = instance_dir / "montage.png"
        frame = cv2.imread(str(file_path))
        return str(file_path), classify_frames_local(generate_fn, [frame], PROMPT)

    file_path = instance_dir / f"{representation}.mp4"
    cap = cv2.VideoCapture(str(file_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    indices = sample_frame_indices(total_frames, NUM_SAMPLED_FRAMES, start=start, end=end)
    frames = extract_frames(str(file_path), indices)
    return str(file_path), classify_frames_local(generate_fn, frames, PROMPT)


def summarize(records: list[dict]) -> dict:
    summary = {}
    for representation in REPRESENTATIONS:
        scored = [r for r in records if r["representation"] == representation and r["correct"] is not None]
        n_correct = sum(r["correct"] for r in scored)
        summary[representation] = {
            "n": len(scored),
            "correct": n_correct,
            "accuracy": n_correct / len(scored) if scored else None,
        }
    return summary


def main() -> None:
    all_records = []
    instances = sorted(INSTANCE_META)
    print(f"Running {MODEL_NAME} over {len(instances)} instances x {len(REPRESENTATIONS)} representations")

    for i, instance in enumerate(instances):
        meta = INSTANCE_META[instance]
        ground_truth = {"weight_class": meta["weight_class"]}

        for representation in REPRESENTATIONS:
            t0 = time.time()
            file_path, result = classify_representation(instance, representation, meta["start"], meta["end"])
            record = log_result(RESULTS_DIR, MODEL_NAME, instance, representation, file_path, result, ground_truth)
            all_records.append(record)

            predicted = record["parsed"]["weight_class"] if record["success"] else None
            mark = "OK" if record["correct"] else "WRONG" if record["correct"] is not None else "FAILED"
            elapsed = time.time() - t0
            print(f"[{i+1}/{len(instances)}][{instance}][{representation}] gt={meta['weight_class']} pred={predicted} ({mark}, {elapsed:.1f}s)")

    summary = summarize(all_records)
    print("\nAccuracy by representation:")
    for representation, stats in summary.items():
        if stats["accuracy"] is None:
            print(f"  {representation}: no scored results")
        else:
            print(f"  {representation}: {stats['correct']}/{stats['n']} = {stats['accuracy']:.2f}")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps({"instances": instances, "accuracy_by_representation": summary}, indent=2))


if __name__ == "__main__":
    main()
