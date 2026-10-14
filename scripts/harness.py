import json
import time


def parse_model_response(raw_text):
    """
    Extract and validate a JSON object from a model's raw text response.
    Returns (parsed_dict_or_None, success_bool, error_message_or_None).
    """
    if raw_text is None:
        return None, False, "Empty response (no text returned by the model)"

    start = raw_text.find("{")
    if start == -1:
        return None, False, "No JSON object found in response"

    try:
        parsed, _ = json.JSONDecoder().raw_decode(raw_text, start)
    except json.JSONDecodeError as e:
        return None, False, f"JSON decode error: {e}"

    if "weight_class" not in parsed:
        return None, False, "Missing 'weight_class' field"
    if parsed["weight_class"] not in (0, 1, 2):
        return None, False, f"Invalid weight_class value: {parsed['weight_class']!r}"
    if "reasoning" not in parsed:
        return None, False, "Missing 'reasoning' field"

    return parsed, True, None


def _generate_with_retry(client, model, build_contents, max_retries, retry_delay, cleanup=None):
    """
    Calls client.models.generate_content with contents from build_contents(),
    retrying on transient failures (rate limit, server overload). build_contents
    is called fresh on every attempt since an upload-based contents list can't
    be reused after a failed attempt. Returns (response_or_None, error_or_None).
    """
    from google.genai import errors as genai_errors

    RETRYABLE_CLIENT_CODES = {429}  # rate limit; other 4xx (bad key, 404, ...) won't succeed on retry

    last_error = None
    for attempt in range(max_retries):
        contents = build_contents()
        try:
            response = client.models.generate_content(model=model, contents=contents)
        except genai_errors.ServerError as e:
            last_error = e
        except genai_errors.ClientError as e:
            if e.code not in RETRYABLE_CLIENT_CODES:
                return None, f"Non-retryable API error: {e}"
            last_error = e
        else:
            return response, None
        finally:
            if cleanup is not None:
                cleanup(contents)

        if attempt < max_retries - 1:
            time.sleep(retry_delay * (attempt + 1))

    return None, f"API call failed after {max_retries} attempts: {last_error}"


def classify_clip(client, video_path, prompt, model="gemini-3.5-flash", max_retries=3, retry_delay=15):
    """
    Uploads one clip, sends it with the given prompt, and returns a structured
    result. Never raises -- a malformed response or a transient API failure
    (rate limit, server overload) is logged as a failed result instead of
    crashing whatever loop is calling this.
    """
    from google.genai import errors as genai_errors

    def build_contents():
        return [client.files.upload(file=video_path), prompt]

    def cleanup(contents):
        try:
            client.files.delete(name=contents[0].name)
        except genai_errors.APIError:
            pass  # best-effort cleanup; Gemini garbage-collects uploads anyway

    response, error = _generate_with_retry(client, model, build_contents, max_retries, retry_delay, cleanup=cleanup)
    if error is not None:
        return {"video_path": video_path, "raw_response": None, "parsed": None, "success": False, "error": error}

    parsed, success, error = parse_model_response(response.text)
    return {
        "video_path": video_path,
        "raw_response": response.text,
        "parsed": parsed,
        "success": success,
        "error": error,
    }


def classify_frames_gemini(client, frames, prompt, model="gemini-3.5-flash", max_retries=3, retry_delay=15):
    """
    Sends a fixed list of pre-sampled frames (BGR numpy arrays) to Gemini as
    inline images, rather than uploading a whole video file. Used so Gemini
    and a local model see exactly the same N frames for a fair comparison --
    classify_clip's whole-video upload has no equivalent frame budget.
    """
    import cv2
    from PIL import Image

    images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]

    def build_contents():
        return [*images, prompt]

    response, error = _generate_with_retry(client, model, build_contents, max_retries, retry_delay)
    if error is not None:
        return {"video_path": None, "raw_response": None, "parsed": None, "success": False, "error": error}

    parsed, success, error = parse_model_response(response.text)
    return {
        "video_path": None,
        "raw_response": response.text,
        "parsed": parsed,
        "success": success,
        "error": error,
    }


def _generate_with_retry_claude(client, build_kwargs, max_retries, retry_delay):
    """
    Calls client.messages.create with kwargs from build_kwargs(), retrying on
    transient failures (rate limit, server overload, connection errors) --
    the Claude-specific counterpart to _generate_with_retry, since Anthropic's
    SDK raises its own exception types rather than google.genai's. Returns
    (response_or_None, error_or_None).
    """
    import anthropic

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.messages.create(**build_kwargs())
        except (anthropic.RateLimitError, anthropic.APIConnectionError) as e:
            last_error = e
        except anthropic.APIStatusError as e:
            if e.status_code < 500:
                return None, f"Non-retryable API error: {e}"
            last_error = e
        else:
            return response, None

        if attempt < max_retries - 1:
            time.sleep(retry_delay * (attempt + 1))

    return None, f"API call failed after {max_retries} attempts: {last_error}"


def classify_frames_claude(client, frames, prompt, model="claude-haiku-4-5", temperature=0,
                            max_tokens=512, max_retries=3, retry_delay=15):
    """
    Sends a fixed list of pre-sampled frames (BGR numpy arrays) to a Claude
    model as inline base64 images -- the Claude counterpart to
    classify_frames_gemini, so Claude, Gemini and the local model all see
    exactly the same N frames and the same prompt for a fair cross-model
    comparison. Frames are encoded as PNG (lossless), not JPEG, so this
    pipeline doesn't introduce a compression difference the other two
    backends' in-process image handling never has.

    temperature=0 is the closest match to Qwen's forced greedy decoding
    (do_sample=False) that a current Claude model's API allows -- claude-
    opus-5-5 and claude-sonnet-5-5 reject an explicit temperature entirely,
    which is why this harness defaults to claude-haiku-4-5: this cross-model
    check is one fixed configuration mirroring the main Qwen grid, not a
    fresh hyperparameter sweep, so matching Qwen's determinism took priority
    over raw model capability.
    """
    import base64

    import cv2

    def encode_frame(frame):
        # cv2.imencode expects BGR input (OpenCV's native frame format, same
        # as every frame this harness already handles) and writes a correctly
        # colored PNG -- no cvtColor needed first, unlike the PIL path
        # classify_frames_gemini uses.
        ok, buf = cv2.imencode(".png", frame)
        return base64.standard_b64encode(buf).decode("utf-8")

    images = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": encode_frame(f)}}
        for f in frames
    ]

    def build_kwargs():
        return {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": [*images, {"type": "text", "text": prompt}]}],
        }

    response, error = _generate_with_retry_claude(client, build_kwargs, max_retries, retry_delay)
    if error is not None:
        return {"video_path": None, "raw_response": None, "parsed": None, "success": False, "error": error}

    raw_text = next((block.text for block in response.content if block.type == "text"), None)
    parsed, success, error = parse_model_response(raw_text)
    return {
        "video_path": None,
        "raw_response": raw_text,
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