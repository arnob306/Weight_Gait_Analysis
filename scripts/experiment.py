import json
from datetime import datetime, timezone
from pathlib import Path

NUM_SAMPLED_FRAMES = 8  # fixed N so Gemini and Qwen see the identical frame budget

PROMPT = (
    'You are an expert in clinical gait analysis. You are shown a sequence of frames '
    'sampled evenly across a video of a person walking. Carefully examine their '
    'walking movement across the frames, paying attention to: step length and '
    'symmetry between the left and right leg, foot clearance from the ground during '
    'each step, arm swing pattern, stride timing, and any dragging, hesitation, or '
    'uneven weight distribution between the legs. '
    'Base your judgment on the walking movement pattern itself, not on clothing, '
    'footwear, or other visual details unrelated to how the person moves. '
    'Classify their gait into exactly one of the following three categories: '
    '0 = normal walking, 1 = limping (one leg affected), 2 = shuffling (both legs '
    'affected). Respond only with valid JSON in this exact format: '
    '{"weight_class": <0, 1, or 2>, "reasoning": "<2-3 sentences describing the '
    'specific gait characteristics you observed that led to this classification>"}'
)


def log_result(
    results_dir: Path,
    model: str,
    instance: int,
    representation: str,
    file_path: str,
    result: dict,
    ground_truth: dict,
) -> dict:
    """Scores one classification result against ground truth and logs it to disk."""
    results_dir.mkdir(parents=True, exist_ok=True)
    predicted = result["parsed"]["weight_class"] if result["success"] else None
    record = {
        "instance": instance,
        "representation": representation,
        "file_path": file_path,
        "model": model,
        "prompt": PROMPT,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_response": result["raw_response"],
        "parsed": result["parsed"],
        "success": result["success"],
        "error": result["error"],
        "ground_truth": ground_truth,
        "correct": predicted == ground_truth["weight_class"] if predicted is not None else None,
    }
    out_path = results_dir / f"instance{instance}_{representation}.json"
    out_path.write_text(json.dumps(record, indent=2))
    return record
