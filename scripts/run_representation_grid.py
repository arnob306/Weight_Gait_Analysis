import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from harness import classify_clip
from labels import get_instance_label, load_labels

load_dotenv()

INSTANCE = 12
MODEL = "gemini-3.5-flash"
CSV_PATH = "All Joints/Absolute_Data.csv"
RESULTS_DIR = Path("results/raw_responses")

REPRESENTATIONS = {
    "raw_video": f"All RGB Videos/Instance_{INSTANCE}.0.mp4",
    "skeleton": f"skeleton_Instance_{INSTANCE}.0.mp4",
    "overlay": f"overlay_Instance_{INSTANCE}.0.mp4",
    "montage": f"montage_Instance_{INSTANCE}.0.png",
}

PROMPT = (
    'You are shown a video of a person walking. Classify their gait into exactly '
    'one of the following three categories: 0 = normal walking, 1 = limping '
    '(one leg affected), 2 = shuffling (both legs affected). Respond only with '
    'valid JSON in this exact format: {"weight_class": <0, 1, or 2>, '
    '"reasoning": "<one sentence explaining what you observed>"}'
)


def log_result(representation: str, file_path: str, result: dict, ground_truth: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    predicted = result["parsed"]["weight_class"] if result["success"] else None
    record = {
        "instance": INSTANCE,
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
    out_path = RESULTS_DIR / f"instance{INSTANCE}_{representation}.json"
    out_path.write_text(json.dumps(record, indent=2))


def main() -> None:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    labels = load_labels(CSV_PATH)
    ground_truth = get_instance_label(labels, INSTANCE)
    print(f"Instance {INSTANCE} ground truth: {ground_truth}")

    for representation, file_path in REPRESENTATIONS.items():
        if not Path(file_path).exists():
            print(f"[{representation}] SKIPPED -- file not found: {file_path}")
            continue

        result = classify_clip(client, file_path, PROMPT, model=MODEL)
        log_result(representation, file_path, result, ground_truth)

        if result["success"]:
            predicted = result["parsed"]["weight_class"]
            mark = "correct" if predicted == ground_truth["weight_class"] else "WRONG"
            print(f"[{representation}] predicted={predicted} ({mark}) -- {result['parsed']['reasoning']}")
        else:
            print(f"[{representation}] FAILED -- {result['error']}")
            print(f"    raw response: {result['raw_response']!r}")


if __name__ == "__main__":
    main()
