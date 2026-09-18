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