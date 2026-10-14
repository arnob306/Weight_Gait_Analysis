"""
Local VLM backend for Qwen3-VL-4B-Instruct, meant to run inside a Kaggle
notebook with a GPU accelerator enabled (Settings > Accelerator > GPU T4 x2
or P100). Not runnable in this repo's environment -- it needs the model
weights and a GPU, neither of which are available here.

Kaggle notebook usage:
    !pip install -q transformers accelerate qwen-vl-utils

    from kaggle_qwen_backend import generate_fn
    from harness import classify_frames_local
    from frame_sampling import sample_frame_indices, extract_frames

    indices = sample_frame_indices(total_frames, n=16)
    frames = extract_frames(video_path, indices)
    result = classify_frames_local(generate_fn, frames, prompt)
"""
import cv2
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is None:
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        )
    return _model, _processor


def generate_fn(frames, prompt: str, temperature: float | None = None, max_new_tokens: int = 512) -> str:
    """
    frames: list of BGR numpy arrays, as returned by frame_sampling.extract_frames.
    Builds a multi-image chat prompt and returns the model's raw text response,
    matching the contract harness.classify_frames_local expects.

    temperature: if given, forces sampling (do_sample=True) at that temperature
    instead of the model's default generation config -- needed for a genuine
    self-consistency test (repeated calls on the same input should only vary
    if sampling is actually enabled).
    """
    model, processor = _load()
    images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]

    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    generate_kwargs = {"max_new_tokens": max_new_tokens}
    if temperature is not None:
        generate_kwargs.update(do_sample=True, temperature=temperature)

    output_ids = model.generate(**inputs, **generate_kwargs)
    response = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    )[0]
    return response
