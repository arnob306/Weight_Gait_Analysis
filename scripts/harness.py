import re
import json

def parse_model_response(raw_text):
    """
    Extract and validate a JSON object from a model's raw text response.
    Returns (parsed_dict_or_None, success_bool, error_message_or_None).
    """
    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if not match:
        return None, False, "No JSON object found in response"

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return None, False, f"JSON decode error: {e}"

    if "weight_class" not in parsed:
        return None, False, "Missing 'weight_class' field"
    if parsed["weight_class"] not in (0, 1, 2):
        return None, False, f"Invalid weight_class value: {parsed['weight_class']!r}"
    if "reasoning" not in parsed:
        return None, False, "Missing 'reasoning' field"

    return parsed, True, None


def classify_clip(client, video_path, prompt, model="gemini-3.5-flash"):
    """
    Uploads one clip, sends it with the given prompt, and returns a structured
    result. Never raises on a malformed response -- logs the failure instead.
    """
    uploaded_file = client.files.upload(file=video_path)
    response = client.models.generate_content(
        model=model,
        contents=[uploaded_file, prompt],
    )
    parsed, success, error = parse_model_response(response.text)
    return {
        "video_path": video_path,
        "raw_response": response.text,
        "parsed": parsed,
        "success": success,
        "error": error,
    }


def classify_frames_local(generate_fn, frames, prompt):
    """
    Runs any local vision-language model over a list of sampled frames and
    returns the same structured result shape as classify_clip.

    generate_fn(frames, prompt) -> raw response text. The caller supplies it,
    so this module has no dependency on any particular model library -- an
    open-weights model such as Qwen3-VL-4B running in a Kaggle notebook and
    the Gemini API both produce results comparable by the same harness.
    """
    raw_text = generate_fn(frames, prompt)
    parsed, success, error = parse_model_response(raw_text)
    return {
        "video_path": None,
        "raw_response": raw_text,
        "parsed": parsed,
        "success": success,
        "error": error,
    }