import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from harness import classify_clip
from labels import get_instance_label, load_labels, select_dev_set
from representations import build_all_representations

load_dotenv()

CSV_PATH = "All Joints/Absolute_Data.csv"
MODEL = "gemini-3.5-flash"
N_PER_CLASS = 2
SEED = 42
RENDER_DIR = "renders"
RESULTS_DIR = Path("results/raw_responses")
SUMMARY_PATH = Path("results/dev_set_summary.json")
# Free tier for gemini-3.5-flash allows 5 requests/minute; space calls out to
# stay under that instead of burning quota on retries.
SECONDS_BETWEEN_CALLS = 13

PROMPT = (
    'You are shown a video of a person walking. Classify their gait into exactly '
    'one of the following three categories: 0 = normal walking, 1 = limping '
    '(one leg affected), 2 = shuffling (both legs affected). Respond only with '
    'valid JSON in this exact format: {"weight_class": <0, 1, or 2>, '
    '"reasoning": "<one sentence explaining what you observed>"}'
)


def log_result(instance: int, representation: str, file_path: str, result: dict, ground_truth: dict) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    predicted = result["parsed"]["weight_class"] if result["success"] else None
    record = {
        "instance": instance,
        "representation": representation,
        "file_path": file_path,
        "model": MODEL,
        "prompt": PROMPT,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_response": result["raw_response"],
        "parsed": result["parsed"],
        "success": result["success"],
        "error": result["error"],
        "ground_truth": ground_truth,
        "correct": predicted == ground_truth["weight_class"] if predicted is not None else None,
    }
    out_path = RESULTS_DIR / f"instance{instance}_{representation}.json"
    out_path.write_text(json.dumps(record, indent=2))
    return record


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
        paths = build_all_representations(instance, f"{RENDER_DIR}/{instance}")
        representations = list(paths.keys())

        for representation, file_path in paths.items():
            result = classify_clip(client, file_path, PROMPT, model=MODEL)
            record = log_result(instance, representation, file_path, result, ground_truth)
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
