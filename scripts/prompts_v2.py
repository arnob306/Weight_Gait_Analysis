"""
Gemma prompt set v2: three frozen prompts built from blocks, so any two differ by
exactly one block. The review text is docs/gemma_prompts_v2_draft.md.

  Z_zero_shot   A + B + C + F
  R_rubric      A + B + C + D + F
  S_few_shot    A + B + C + D + E + F   (examples not chosen yet; build_prompt raises)

A task and class definition, B what the images show (the only block that changes
between the four inputs), C timing and viewpoint (only numbers change), D rubric,
E worked examples, F output format. P0 to P3 in prompts.py stay as development
history.

Assumption behind the timing sentence: the stored clips keep every recorded frame,
so a gap of k stored frames is k / SOURCE_FPS seconds. SOURCE_FPS is the dataset's
typical recording rate (README: 27 to 32 fps); the stored .mp4 header says 7 fps.
Confirm before the numbers go into the report.
"""
import numpy as np

from gait_parse import LEG_ENUMS, LEG_OTHER, TOP_ENUMS

_NUMBER_WORDS = {4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
SOURCE_FPS = 28.5
MONTAGE_COLS = 4

VARIANTS = {
    "Z_zero_shot": ("A", "B", "C", "F"),
    "R_rubric": ("A", "B", "C", "D", "F"),
    "S_few_shot": ("A", "B", "C", "D", "E", "F"),
}
FRAME_REPS = ("raw_video", "skeleton", "overlay", "raw_ankle_cut")
BLOCK_A = (
    "You are an expert in clinical gait analysis. You will see {n} frames, in time order, from one "
    "short video of a person walking. Describe how each of the walker's legs moves, using only what "
    "you can see in the frames, and then classify the walk.\n\n"
    "You will report:\n"
    "- gait_class: how many of the walker's legs move in an impaired way. 0 = neither leg (normal "
    "walking), 1 = exactly one leg, 2 = both legs. Give the integer only.\n"
    "- freeze: whether the walker stops, or nearly stops, at some point in the video.\n"
    "- obstacle: whether the walker steps over an object on the floor.\n\n"
    "Judge the movement only. Clothing, footwear, anything attached to the legs, the background and "
    "who the person is are not evidence."
)

BLOCK_B = {
    "raw_video": "Each image is one frame of the video.",
    "raw_ankle_cut": (
        "Each image is one frame of the video. The lower part of the image, from the "
        "walker's ankles down, is blacked out."
    ),
    "skeleton": (
        "Each image is a stick figure drawn from body-joint positions estimated from one "
        "frame of the video, on a plain background. The walker's left arm and leg are drawn in blue, "
        "the right arm and leg in red, and the head and trunk in grey. The joint positions are "
        "estimates and can be slightly wrong in a single frame."
    ),
    "overlay": (
        "Each image is one frame of the video with a stick figure drawn on top of the "
        "person, from estimated body-joint positions. The walker's left arm and leg are drawn in blue, "
        "the right arm and leg in red. The stick figure is a guide and can be slightly wrong in a "
        "single frame; describe how the person moves."
    ),
    "montage": (
        "You are shown a single image containing all {n} frames, arranged in a {rows} x {cols} grid. "
        "Read the grid left to right, top row first. Each tile has its frame number in its top-left "
        "corner."
    ),
}

BLOCK_C = (
    "Each frame is labelled with its number, from 1 to {n}. "
    "The frames are about {dt} seconds apart and together cover about {span} seconds. You will see "
    "part of the stride, not every step, so rely on differences that persist across several frames "
    "and not on a single pose.{direction}"
)
_DIRECTION = (
    " The walker moves toward the {direction} of the image, so the walker's {near} leg is the one "
    "nearer the camera."
)

BLOCK_D = (
    "HOW TO JUDGE EACH LEG\n"
    "Compare each leg with the other leg and with typical unimpaired walking. Count a leg as impaired "
    "if it shows one or more of these signs on most of the ordinary steps you can see, not just in a "
    "single frame:\n"
    "- Step length: the leg's forward step is shorter than typical, so the feet stay closer "
    "together.\n"
    "- Foot lift: the foot barely clears the floor during its forward swing, or scuffs or drags.\n"
    "- Knee bend: the leg looks stiff, with less knee bend than typical while it swings forward.\n"
    "These signs can be small and are often hard to see. If you cannot tell, mark the field "
    "'unclear' instead of guessing.\n\n"
    "Legs that look alike are not automatically normal. If both legs take short, low or stiff steps "
    "compared with typical walking, both are impaired. If only one leg shows the signs, or one leg "
    "shows them clearly more than the other, only that leg is impaired.\n"
    "Describe arm swing, rhythm and overall speed, but do not decide the class from them alone.\n\n"
    "EVENTS THAT ARE NOT IMPAIRMENT\n"
    "About half of the videos contain one of two events. Report them in 'freeze' and 'obstacle', and "
    "judge the legs from the ordinary steps, not from these moments.\n"
    "- Freeze: the walker stops or nearly stops for a moment. Standing still is not a short step.\n"
    "- Obstacle: the walker steps over a small object on the floor. That step lifts the foot and "
    "bends the knee much more than usual; it is not evidence about the leg.\n"
    "A video can contain one of these events and impaired legs at the same time.\n\n"
    "BASE RATES\n"
    "About one third of the videos have neither leg impaired, one third one leg, and one third both "
    "legs. About one video in four contains a freeze and about one in four an obstacle step. No answer "
    "is more likely before you look.\n\n"
    "HOW TO WORK\n"
    "1. Follow the left leg through the frames and fill in its fields, citing frame numbers.\n"
    "2. Do the same for the right leg.\n"
    "3. Compare the two legs in 'comparison'.\n"
    "4. Check for a freeze or an obstacle step and note the frames.\n"
    "5. Decide whether each leg is impaired. gait_class must equal the number of legs you marked as "
    "impaired.\n"
    "6. If a leg is hard to see, say so in its 'visibility' field and lower 'confidence'. An honest "
    "'unclear' is better than a guess."
)


def _enum(values) -> str:
    return " | ".join(f'"{v}"' for v in values)


def _leg_fields() -> str:
    lines = [f'    "{name}": {_enum(vals)},' for name, vals in LEG_ENUMS.items()]
    lines.append('    "evidence_frames": [numbers of the frames that show the above],')
    lines.append('    "description": "one sentence on what this leg does, only what is visible"')
    return "\n".join(lines)


def _block_f() -> str:
    return (
        "Return ONE JSON object with exactly these fields, in this order. Use only the values listed "
        "for each field.\n"
        "{\n"
        '  "left_leg": {\n' + _leg_fields() + "\n  },\n"
        f'  "right_leg": {{ the same {_NUMBER_WORDS[len(LEG_ENUMS) + len(LEG_OTHER)]} fields }},\n'
        f'  "arm_swing": {_enum(TOP_ENUMS["arm_swing"])},\n'
        f'  "rhythm": {_enum(TOP_ENUMS["rhythm"])},\n'
        '  "comparison": "1-2 sentences comparing the two legs",\n'
        '  "freeze": true | false,\n'
        '  "freeze_frames": [frame numbers, or empty],\n'
        '  "obstacle": true | false,\n'
        '  "obstacle_frames": [frame numbers, or empty],\n'
        '  "left_leg_impaired": true | false,\n'
        '  "right_leg_impaired": true | false,\n'
        '  "gait_class": 0 | 1 | 2,\n'
        f'  "confidence": {_enum(TOP_ENUMS["confidence"])}\n'
        "}\n"
        "gait_class is the integer 0, 1 or 2, never a word.\n"
        "Output the JSON object only: no other text and no markdown fences."
    )


BLOCK_F = _block_f()

PROBE_WEIGHTS = (
    "Look only at the person's legs. Is anything strapped or attached to either leg, such as a band, "
    "brace or weight?\n"
    'Return JSON only: {"left_leg_attachment": "yes" | "no" | "unclear", '
    '"right_leg_attachment": "yes" | "no" | "unclear", "frames": [frame numbers where it is visible]}'
)
PROBE_DIRECTION = (
    "Which way is the walker moving across the image: toward the left side or toward the right side?\n"
    'Return JSON only: {"direction": "left" | "right" | "unclear"}'
)
PROBE_PAIRS = {
    "freeze": (
        "Does the walker stop or nearly stop at any point in these frames?",
        "Does the walker keep moving forward without stopping in all of these frames?",
    ),
    "obstacle": (
        "Does the walker step over an object on the floor in these frames?",
        "Does the walker take only ordinary steps, without stepping over anything, in these frames?",
    ),
    "left_leg": (
        "Does the walker's left leg move in an impaired way?",
        "Does the walker's left leg move normally?",
    ),
    "right_leg": (
        "Does the walker's right leg move in an impaired way?",
        "Does the walker's right leg move normally?",
    ),
}
PROBE_YES_NO = "Answer with one word: yes or no."


def clip_timing(indices: list[int], source_fps: float = SOURCE_FPS) -> tuple[float, float]:
    """(mean seconds between sampled frames, seconds from first to last), from stored-frame indices."""
    if len(indices) < 2:
        return 0.0, 0.0
    gaps = np.diff(indices)
    return float(np.mean(gaps)) / source_fps, float(indices[-1] - indices[0]) / source_fps


def frame_labels(rep: str, n: int) -> list[str] | None:
    """Text line sent before each image; None for the montage, whose tiles carry the numbers."""
    return None if rep == "montage" else [f"Frame {i} of {n}" for i in range(1, n + 1)]


def _fill(rep: str, indices: list[int], direction: str | None, near_leg: str | None) -> dict:
    if rep != "montage" and rep not in FRAME_REPS:
        raise ValueError(f"unknown input {rep!r}")
    n = len(indices)
    dt, span = clip_timing(indices)
    dir_text = _DIRECTION.format(direction=direction, near=near_leg) if direction and near_leg else ""
    return {"n": n, "rows": -(-n // MONTAGE_COLS), "cols": MONTAGE_COLS,
            "dt": f"{dt:.2f}", "span": f"{span:.1f}", "direction": dir_text}


def _view(rep: str, indices, direction, near_leg) -> str:
    """Blocks B and C for one clip: what the images show, then timing and viewpoint."""
    fill = _fill(rep, indices, direction, near_leg)
    return "\n\n".join([BLOCK_B[rep].format(**fill), BLOCK_C.format(**fill)])


def build_prompt(variant: str, rep: str, indices: list[int], direction: str | None,
                 near_leg: str | None) -> str:
    """The full text for one clip under one prompt variant and input."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r}")
    blocks = VARIANTS[variant]
    if "E" in blocks:
        raise NotImplementedError("few-shot examples have not been chosen yet")
    fill = _fill(rep, indices, direction, near_leg)
    parts = [BLOCK_A.format(**fill), _view(rep, indices, direction, near_leg)]
    if "D" in blocks:
        parts.append(BLOCK_D)
    parts.append(BLOCK_F)
    return "\n\n".join(parts)


def build_probe(kind: str, rep: str, indices: list[int], direction: str | None, near_leg: str | None,
                question: str | None = None) -> str:
    """Probe prompts, run separately from classification. kind is 'weights', or a PROBE_PAIRS
    topic with question set to one of its two questions. Blocks B and C come first."""
    n = len(indices)
    if kind == "direction":  # the viewpoint sentence in block C would give the answer away
        return f"You will see {n} frames from a video of a person walking.\n\n" \
               f"{_view(rep, indices, None, None)}\n\n{PROBE_DIRECTION}"
    view = _view(rep, indices, direction, near_leg)
    if kind == "weights":
        return f"You will see {n} frames from a video of a person walking.\n\n{view}\n\n{PROBE_WEIGHTS}"
    if kind not in PROBE_PAIRS or question not in PROBE_PAIRS[kind]:
        raise ValueError(f"unknown probe {kind!r} / question {question!r}")
    return f"{view}\n\n{question} {PROBE_YES_NO}"
