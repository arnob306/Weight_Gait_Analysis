"""
Branch D, Phase 1 -- attribute elicitation. See docs/branch_d_guideline.md.

The model is shown the same N sampled frames as Branch C but is never asked
for a class. It rates 27 named gait attributes on a 3-point ordinal scale,
split over three grouped calls per clip (legs / foot+temporal / upper body)
so each JSON response stays small. Output feeds Phase 2 (logistic regression,
run locally) and Phase 3 (agreement with joint-computed values).

Three scale types, chosen so every rating is monotone in its computed
counterpart (a requirement for Spearman agreement in Phase 3):
  MAGNITUDE -- 0 = smaller than typical, 1 = typical, 2 = larger than typical
  TIMING    -- 0 = earlier/shorter, 1 = typical, 2 = later/longer
  SEVERITY  -- 0 = none, 1 = mild, 2 = marked  (asymmetries, variability)
Every attribute may also be answered null ("cannot judge from these frames");
nulls are recorded, not imputed -- which attributes are elicitable is itself
a finding (guideline section 6).

Kaggle usage (after the notebook's setup cells have defined generate_greedy):
    import branch_d_elicit
    branch_d_elicit.run(generate_greedy, input_dir=REPR_DIR)
"""
import json
import time
from pathlib import Path

import cv2

from experiment import NUM_SAMPLED_FRAMES
from frame_sampling import extract_frames, sample_frame_indices
from run_kaggle_qwen import INSTANCE_META

OUTPUT_PATH = Path("/kaggle/working/results/branch_d_elicitation.json")
REPRESENTATION = "skeleton"
VALID_RATINGS = (0, 1, 2)

MAGNITUDE = "0 = smaller than typical, 1 = typical, 2 = larger than typical"
TIMING = "0 = earlier/shorter than typical, 1 = typical, 2 = later/longer than typical"
SEVERITY = "0 = none, 1 = mild, 2 = marked"

# group -> [(key, description, scale)]
ATTRIBUTE_GROUPS = {
    "legs": [
        ("knee_flexion_rom_left", "range of knee bending over the stride, left leg", MAGNITUDE),
        ("knee_flexion_rom_right", "range of knee bending over the stride, right leg", MAGNITUDE),
        ("hip_flexion_rom_left", "range of forward/backward thigh swing, left leg", MAGNITUDE),
        ("hip_flexion_rom_right", "range of forward/backward thigh swing, right leg", MAGNITUDE),
        ("knee_angle_asymmetry", "difference between left and right knee bending", SEVERITY),
        ("hip_angle_asymmetry", "difference between left and right thigh swing", SEVERITY),
        ("step_length", "maximum front-to-back distance between the two feet during a step", MAGNITUDE),
        ("step_width", "side-to-side distance between the two knees", MAGNITUDE),
    ],
    "foot_temporal": [
        ("foot_vertical_rom_left", "toe clearance: how high the left foot lifts off the ground during swing", MAGNITUDE),
        ("foot_vertical_rom_right", "toe clearance: how high the right foot lifts off the ground during swing", MAGNITUDE),
        ("foot_clearance_asymmetry", "difference between left and right toe clearance (one foot dragging more)", SEVERITY),
        ("stance_fraction_left", "share of the stride the left foot spends on the ground", TIMING),
        ("stance_fraction_right", "share of the stride the right foot spends on the ground", TIMING),
        ("stance_swing_asymmetry", "difference between legs in time spent on the ground versus in the air", SEVERITY),
        ("peak_knee_flexion_timing_left", "when in the stride cycle the left knee bends the most", TIMING),
        ("peak_knee_flexion_timing_right", "when in the stride cycle the right knee bends the most", TIMING),
        ("stride_time", "duration of one full stride", TIMING),
        ("step_length_variability", "how much step length changes from one step to the next", SEVERITY),
        ("knee_angle_variability", "how jerky or unsteady knee motion is between consecutive frames", SEVERITY),
    ],
    "upper_body": [
        ("trunk_lean_rom", "how much the midpoint of the shoulders moves relative to the hips (trunk lean or sway)", MAGNITUDE),
        ("arm_swing_amplitude_left", "how far the left wrist travels relative to the left shoulder", MAGNITUDE),
        ("arm_swing_amplitude_right", "how far the right wrist travels relative to the right shoulder", MAGNITUDE),
        ("arm_swing_asymmetry", "difference between left and right wrist travel", SEVERITY),
        ("elbow_flexion_rom", "range of the shoulder-elbow-wrist angle (how much the arms bend while swinging)", MAGNITUDE),
        ("elbow_height_difference", "difference in height between the two elbows at peak arm swing", SEVERITY),
        ("shoulder_height_asymmetry", "one shoulder dropped or hiked relative to the other", SEVERITY),
        ("shoulder_rotation_rom", "how much the line between the shoulders rotates relative to the walking direction", MAGNITUDE),
    ],
}

PROMPT_HEADER = (
    "You are shown a sequence of frames sampled evenly across a video of a "
    "person walking, rendered as a skeleton. Do not diagnose or classify the "
    "gait. Only rate each movement attribute listed below, based on what is "
    "visible in the frames. If an attribute cannot be judged from these "
    "frames, answer null for it rather than guessing.\n\nAttributes:\n"
)


def build_prompt(group: str) -> str:
    lines = [f'- "{key}": {desc}. Scale: {scale}.' for key, desc, scale in ATTRIBUTE_GROUPS[group]]
    template = ", ".join(f'"{key}": <0, 1, 2, or null>' for key, _, _ in ATTRIBUTE_GROUPS[group])
    return (
        PROMPT_HEADER
        + "\n".join(lines)
        + "\n\nRespond only with valid JSON in this exact format: {" + template + "}"
    )


def parse_ratings(raw_text: str | None, group: str) -> tuple[dict, dict, str | None]:
    """Returns (ratings, status, error). Every key in the group is present in
    both dicts. status separates a deliberate null ('null' -- the model said it
    cannot judge) from a missing or out-of-range value ('invalid'), since only
    the former is evidence about elicitability."""
    keys = [key for key, _, _ in ATTRIBUTE_GROUPS[group]]
    failed = ({key: None for key in keys}, {key: "invalid" for key in keys})

    if not raw_text or "{" not in raw_text:
        return *failed, "No JSON object found in response"
    try:
        parsed, _ = json.JSONDecoder().raw_decode(raw_text, raw_text.find("{"))
    except json.JSONDecodeError as e:
        return *failed, f"JSON decode error: {e}"
    if not isinstance(parsed, dict):
        return *failed, "Response JSON is not an object"

    ratings, status = {}, {}
    for key in keys:
        value = parsed.get(key)
        # bool is an int subclass; reject true/false explicitly
        if value in VALID_RATINGS and not isinstance(value, bool):
            ratings[key], status[key] = value, "rated"
        elif key in parsed and value is None:
            ratings[key], status[key] = None, "null"
        else:
            ratings[key], status[key] = None, "invalid"
    return ratings, status, None


def load_frames(input_dir: Path, instance: int, start: int, end: int) -> list:
    video_path = input_dir / str(instance) / f"{REPRESENTATION}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    indices = sample_frame_indices(total_frames, NUM_SAMPLED_FRAMES, start=start, end=end)
    return extract_frames(str(video_path), indices)


def elicit_instance(generate, input_dir: Path, instance: int) -> dict:
    meta = INSTANCE_META[instance]
    frames = load_frames(input_dir, instance, meta["start"], meta["end"])
    record = {"instance": instance, "weight_class": meta["weight_class"],
              "ratings": {}, "status": {}, "raw": {}, "errors": {}}
    for group in ATTRIBUTE_GROUPS:
        raw_text = generate(frames, build_prompt(group))
        ratings, status, error = parse_ratings(raw_text, group)
        record["ratings"].update(ratings)
        record["status"].update(status)
        record["raw"][group] = raw_text
        record["errors"][group] = error
    return record


def run(generate, input_dir: Path | str, output_path: Path = OUTPUT_PATH) -> list[dict]:
    """generate(frames, prompt) -> raw text. Pass the notebook's generate_greedy
    so decoding matches Branch C. Writes after every clip so a Kaggle session
    drop loses at most one clip; re-running skips clips already on disk."""
    input_dir = Path(input_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(output_path.read_text()) if output_path.exists() else []
    done = {r["instance"] for r in records}
    todo = [i for i in sorted(INSTANCE_META) if i not in done]
    print(f"Branch D elicitation: {len(done)} done, {len(todo)} to go "
          f"({len(ATTRIBUTE_GROUPS)} calls each)")

    for n, instance in enumerate(todo, start=1):
        t0 = time.time()
        record = elicit_instance(generate, input_dir, instance)
        records = [*records, record]
        output_path.write_text(json.dumps(records, indent=2))
        n_rated = sum(s == "rated" for s in record["status"].values())
        print(f"[{n}/{len(todo)}][{instance}] rated {n_rated}/{len(record['status'])} "
              f"({time.time() - t0:.1f}s)")

    return records
