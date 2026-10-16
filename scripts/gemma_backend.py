"""
Local VLM backend for Gemma-4-E4B-it, for a Kaggle notebook with a GPU
(Settings > Accelerator > GPU T4 x2). Same contract as kaggle_qwen_backend:
generate_fn(frames, prompt) -> raw response text, so harness.classify_frames_local
and the existing parser and scoring work unchanged.

Kaggle notebook usage (install in its own first cell, before any import):
    !pip install -q -U transformers accelerate

    import sys
    sys.path.append('/kaggle/input/datasets/arnobsaha1/weightgait-pilot')
    import gemma_backend as gb
    raw = gb.generate_fn(frames, prompt)

Decoding is greedy (do_sample=False), the lowest temperature available, and
the seed is fixed at 27. Google's model card recommends temperature 1.0 with
top_p 0.95 and top_k 64; greedy is a deliberate departure from that, for
reproducibility, and should be stated in the methods.

Frames are passed as a list of images in time order. The model card's video
path decides its own frame rate; passing frames lets the fps sampler (1, 2 or
3 fps, equally spaced) control exactly which frames the model sees.

Not runnable locally: it needs the model weights and a GPU.
"""
import cv2
import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

MODEL_ID = "google/gemma-4-E4B-it"
SEED = 27

# Settings a notebook may change before the first call (module attributes are read
# when the model loads / when a call is made):
#   MAX_SOFT_TOKENS  visual tokens per image; one of 70, 140, 280, 560, 1120. The
#                    processor's default is 280, which rescales our 424x240 frames
#                    to about 645K pixels; a smaller budget costs less memory and time.
#   MAX_MEMORY       per-GPU cap on weights, so GPU 0 keeps room for the vision
#                    encoder's activations (it holds the image tower).
#   DTYPE            "auto" keeps the checkpoint's bfloat16; T4 GPUs have no native
#                    bfloat16, so "float16" is the thing to try if calls are slow.
MAX_SOFT_TOKENS = 280
MAX_MEMORY = {0: "7GiB", 1: "14GiB"}
DTYPE = "auto"

_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is None:
        torch.manual_seed(SEED)
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID, dtype=DTYPE, device_map="auto", max_memory=MAX_MEMORY,
        )
    return _model, _processor


def _build_inputs(frames, prompt: str, labels=None, examples=None):
    """labels: optional text line sent just before each image (e.g. "Frame 3 of 5").
    examples: optional (text, image or None) pairs sent first, in order: the worked examples."""
    model, processor = _load()
    processor.image_processor.max_soft_tokens = MAX_SOFT_TOKENS
    images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    if labels is not None and len(labels) != len(images):
        raise ValueError(f"{len(labels)} labels for {len(images)} images")

    content = []
    for text, example_image in examples or []:
        content.append({"type": "text", "text": text})
        if example_image is not None:
            content.append({"type": "image", "image": Image.fromarray(cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB))})
    for k, image in enumerate(images):
        if labels is not None:
            content.append({"type": "text", "text": labels[k]})
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    return processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True,
    ).to(model.device)


def count_input_tokens(frames, prompt: str, labels=None, examples=None) -> int:
    """Length of the model input for these frames, to check what MAX_SOFT_TOKENS does."""
    return int(_build_inputs(frames, prompt, labels, examples)["input_ids"].shape[-1])


def generate_fn(frames, prompt: str, max_new_tokens: int = 512, labels=None, examples=None,
                temperature: float | None = None, seed: int | None = None) -> str:
    """frames: list of BGR numpy arrays, as returned by frame_sampling.extract_frames.
    An empty list gives the text-only blind control. labels: optional text line before
    each image. examples: worked examples, see _build_inputs. temperature: None keeps the
    greedy default; a number samples at that temperature (top_p 0.95, top_k 64, the model card's
    values) after seeding with seed, for the confidence-agreement runs."""
    model, processor = _load()
    inputs = _build_inputs(frames, prompt, labels, examples)
    input_len = inputs["input_ids"].shape[-1]

    sampling = ({"do_sample": True, "temperature": temperature, "top_p": 0.95, "top_k": 64}
                if temperature is not None else
                {"do_sample": False, "temperature": None,  # clear the sampling defaults in the model's generation config
                 "top_p": None, "top_k": None})
    if seed is not None:
        torch.manual_seed(seed)
    try:
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, **sampling)
        return processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
    finally:
        torch.cuda.empty_cache()  # return the vision encoder's temporary memory between calls
