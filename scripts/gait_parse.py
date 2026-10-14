"""
Parser and checks for the Gemma v2 answer format (see prompts_v2.py, Block F).

The allowed values live here once; prompts_v2 builds the output-format text from
them, so the prompt and the checker cannot drift apart.

parse_gait_response(raw) -> (parsed | None, ok, error)
  ok is True when gait_class is the integer 0, 1 or 2; everything else is
  recorded in parsed["_checks"] and never makes the answer unusable:
    missing_fields  required fields the model left out
    invalid_values  fields whose value is outside the allowed list
    consistent      gait_class == number of legs flagged impaired
                    (None if either flag is missing). This is the internal
                    consistency measure for the transparency analysis.
"""
import json

LEG_ENUMS = {
    "visibility": ("clear", "partly hidden", "mostly hidden"),
    "step_length": ("short", "typical", "unclear"),
    "foot_lift": ("low", "typical", "unclear"),
    "knee_bend_in_swing": ("reduced", "typical", "unclear"),
}
LEG_OTHER = ("evidence_frames", "description")
TOP_ENUMS = {
    "arm_swing": ("reduced", "typical", "uneven between sides", "unclear"),
    "rhythm": ("steady", "irregular", "unclear"),
    "confidence": ("high", "medium", "low"),
}
BOOL_FIELDS = ("freeze", "obstacle", "left_leg_impaired", "right_leg_impaired")
LEGS = ("left_leg", "right_leg")


def _is_int_class(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value in (0, 1, 2)


def _check(parsed: dict) -> dict:
    missing, invalid = [], []
    for leg in LEGS:
        block = parsed.get(leg)
        if not isinstance(block, dict):
            missing.append(leg)
            continue
        for name, allowed in LEG_ENUMS.items():
            if name not in block:
                missing.append(f"{leg}.{name}")
            elif block[name] not in allowed:
                invalid.append(f"{leg}.{name}={block[name]!r}")
        for name in LEG_OTHER:
            if name not in block:
                missing.append(f"{leg}.{name}")
    for name, allowed in TOP_ENUMS.items():
        if name not in parsed:
            missing.append(name)
        elif parsed[name] not in allowed:
            invalid.append(f"{name}={parsed[name]!r}")
    for name in BOOL_FIELDS:
        if name not in parsed:
            missing.append(name)
        elif not isinstance(parsed[name], bool):
            invalid.append(f"{name}={parsed[name]!r}")

    flags = [parsed.get("left_leg_impaired"), parsed.get("right_leg_impaired")]
    consistent = None
    if all(isinstance(f, bool) for f in flags):
        consistent = parsed["gait_class"] == sum(flags)
    return {"missing_fields": missing, "invalid_values": invalid, "consistent": consistent}


def parse_gait_response(raw_text):
    """Extract the first JSON object from a response (fenced or bare) and check it."""
    if raw_text is None:
        return None, False, "Empty response (no text returned by the model)"
    start = raw_text.find("{")
    if start == -1:
        return None, False, "No JSON object found in response"
    try:
        parsed, _ = json.JSONDecoder().raw_decode(raw_text, start)
    except json.JSONDecodeError as e:
        return None, False, f"JSON decode error: {e}"
    if not isinstance(parsed, dict):
        return None, False, "Response JSON is not an object"
    if "gait_class" not in parsed:
        return None, False, "Missing 'gait_class' field"
    if not _is_int_class(parsed["gait_class"]):
        return None, False, f"Invalid gait_class value: {parsed['gait_class']!r}"
    return {**parsed, "_checks": _check(parsed)}, True, None


PROBE_VALUES = ("yes", "no", "unclear")
PROBE_KEYS = ("left_leg_attachment", "right_leg_attachment")


def parse_probe_response(raw_text):
    """Weights probe answer: both attachment fields must be 'yes', 'no' or 'unclear'.
    Returns (parsed | None, ok, error); 'frames' is kept if present."""
    if raw_text is None:
        return None, False, "Empty response (no text returned by the model)"
    start = raw_text.find("{")
    if start == -1:
        return None, False, "No JSON object found in response"
    try:
        parsed, _ = json.JSONDecoder().raw_decode(raw_text, start)
    except json.JSONDecodeError as e:
        return None, False, f"JSON decode error: {e}"
    if not isinstance(parsed, dict):
        return None, False, "Response JSON is not an object"
    for key in PROBE_KEYS:
        if parsed.get(key) not in PROBE_VALUES:
            return None, False, f"Missing or invalid {key}: {parsed.get(key)!r}"
    return parsed, True, None


DIRECTION_VALUES = ("left", "right", "unclear")


def parse_direction_response(raw_text):
    """Direction control answer: {"direction": "left" | "right" | "unclear"}.
    Returns (parsed | None, ok, error)."""
    if raw_text is None:
        return None, False, "Empty response (no text returned by the model)"
    start = raw_text.find("{")
    if start == -1:
        return None, False, "No JSON object found in response"
    try:
        parsed, _ = json.JSONDecoder().raw_decode(raw_text, start)
    except json.JSONDecodeError as e:
        return None, False, f"JSON decode error: {e}"
    if not isinstance(parsed, dict) or parsed.get("direction") not in DIRECTION_VALUES:
        return None, False, f"Missing or invalid direction: {parsed!r}"
    return parsed, True, None


def parse_yesno_response(raw_text):
    """One-word yes/no probe answer. Returns ("yes" | "no" | None, ok, error); case, quotes and
    trailing punctuation are ignored, anything else is not an answer."""
    if raw_text is None or not raw_text.strip():
        return None, False, "Empty response (no text returned by the model)"
    first = raw_text.strip().split()[0].lower().strip("`*\"'.,;:!")
    if first in ("yes", "no"):
        return first, True, None
    return None, False, f"Not a yes/no answer: {raw_text.strip()[:40]!r}"


def nine_class(gait_class: int, freeze: bool, obstacle: bool) -> int:
    """3 x gait_class + event, event = 0 none, 1 freeze, 2 obstacle. The dataset never
    has both events in one clip; if the model reports both, freeze wins."""
    event = 1 if freeze else 2 if obstacle else 0
    return 3 * gait_class + event


def leg_truth(weight_class: int) -> tuple[bool, bool]:
    """(left impaired, right impaired): the weight is on the left leg in class 1
    and on both legs in class 2 (WeightGait README)."""
    return weight_class >= 1, weight_class == 2
