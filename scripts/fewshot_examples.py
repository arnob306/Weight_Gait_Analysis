"""
The worked examples for prompt S (block E): three clips of participants who are in neither
the dev nor the held-out set (samples.FEWSHOT_INSTANCES), the same three for every clip.

  975   class 0, obstacle step    (participant 16)
  1712  class 1, freeze           (participant 28)
  343   class 2                   (participant 5)

Each example is shown as ONE grid image of its numbered frames (the memory fallback: sending
every frame of three examples next to the clip's own frames does not fit on the T4), then its
answer in the format of block F. The observation fields were written by looking at the grids
in kaggle_upload/fewshot/, not copied from the labels. They describe movement only.

Known weakness, to state in the report: in example 1712 the frames do not clearly show an
impaired left leg, so its answer marks the leg fields 'unclear', says so in the comparison and
gives confidence 'low', while the class (1) follows the label.
"""
import json
from typing import Callable

from samples import FEWSHOT_INSTANCES

INTRO = (
    "WORKED EXAMPLES\n"
    "Below are {k} worked examples from other people, each with its correct answer. Each example "
    "is one image holding all of its frames in a grid, read left to right, top row first, with "
    "each tile's frame number in its top-left corner, drawn the same way as the frames you will "
    "be shown. The person in the video you must classify is someone else."
)
OUTRO = "End of the worked examples. The new video follows."


def _leg(visibility, step, lift, knee, frames, description):
    return {"visibility": visibility, "step_length": step, "foot_lift": lift,
            "knee_bend_in_swing": knee, "evidence_frames": frames, "description": description}


ANSWERS = {
    975: {
        "left_leg": _leg("clear", "typical", "typical", "typical", [2, 3, 4],
                         "The left leg takes long, free steps and the foot clears the floor."),
        "right_leg": _leg("clear", "typical", "typical", "typical", [2, 3, 4],
                          "The right leg swings forward with a normal step and normal knee bend."),
        "arm_swing": "typical",
        "rhythm": "steady",
        "comparison": "The two legs look alike and both move freely.",
        "freeze": False, "freeze_frames": [],
        "obstacle": True, "obstacle_frames": [3],
        "left_leg_impaired": False, "right_leg_impaired": False,
        "gait_class": 0, "confidence": "high",
    },
    1712: {
        "left_leg": _leg("clear", "unclear", "unclear", "unclear", [2, 4, 5],
                         "Only a few frames show the left leg stepping, and nothing in them is clearly abnormal."),
        "right_leg": _leg("partly hidden", "unclear", "unclear", "unclear", [2, 5],
                          "The right leg is the far leg and is partly hidden behind the left."),
        "arm_swing": "unclear",
        "rhythm": "irregular",
        "comparison": "No clear difference between the legs; the stop in frame 3 leaves few steps to compare.",
        "freeze": True, "freeze_frames": [3],
        "obstacle": False, "obstacle_frames": [],
        "left_leg_impaired": True, "right_leg_impaired": False,
        "gait_class": 1, "confidence": "low",
    },
    343: {
        "left_leg": _leg("clear", "short", "unclear", "unclear", [2, 3, 4],
                         "The left foot lands close to the right foot, so the forward steps are short."),
        "right_leg": _leg("clear", "short", "unclear", "unclear", [2, 3, 4],
                          "The right foot also stays close to the left foot with short forward steps."),
        "arm_swing": "typical",
        "rhythm": "steady",
        "comparison": "Both legs take the same short steps, with the feet staying close together.",
        "freeze": False, "freeze_frames": [],
        "obstacle": False, "obstacle_frames": [],
        "left_leg_impaired": True, "right_leg_impaired": True,
        "gait_class": 2, "confidence": "medium",
    },
}
GRID_FILES = {"raw_video": "raw_video", "montage": "raw_video", "skeleton": "skeleton",
              "overlay": "overlay", "raw_ankle_cut": "raw_video", "blind": "raw_video"}


def answer_text(instance: int) -> str:
    return json.dumps(ANSWERS[instance], separators=(",", ":"))  # compact: saves tokens, and memory on the T4


def example_items(view: str, load_grid: Callable[[int, str], object]) -> list[tuple[str, object]]:
    """The worked examples as (text, image or None) items, in the order they are sent.
    load_grid(instance, grid_name) returns the example's grid image for that input."""
    grid = GRID_FILES[view]
    items = [(INTRO.format(k=len(FEWSHOT_INSTANCES)), None)]
    for n, instance in enumerate(FEWSHOT_INSTANCES, start=1):
        items.append((f"Example {n}, frames:", load_grid(instance, grid)))
        items.append((f"Example {n}, correct answer:\n{answer_text(instance)}", None))
    items.append((OUTRO, None))
    return items
