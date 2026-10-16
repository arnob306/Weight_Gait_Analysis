"""
Branch D with Gemma: the same 27 attributes and three grouped calls per clip as
branch_d_elicit.py (the attribute list, scales and parser are imported from there), run on the
36 dev clips with the same frames the classification runs use, for two inputs: skeleton and
raw video. The model is never asked for a class.

Only the opening sentence of the prompt changes with the input, so the two inputs differ in
what the images show and nothing else.

Notebook usage, after the setup cells:
    import branch_d_gemma
    branch_d_gemma.run(DEV_INSTANCES, ["skeleton", "raw_video"])
216 calls (36 clips x 2 inputs x 3 groups). A rerun skips finished calls.
"""
import json
import time
from pathlib import Path

import prompts_v2
import run_gemma
from branch_d_elicit import ATTRIBUTE_GROUPS, build_prompt as attribute_list, parse_ratings

OUT_PATH = Path("/kaggle/working/results/branch_d_gemma.jsonl")
REPS = ("skeleton", "raw_video")
MAX_NEW_TOKENS = 400  # the answer is one short JSON object of nine or eleven numbers

OPENING = {
    "skeleton": (
        "You are shown {n} frames, in time order, from a video of a person walking. Each image is "
        "a stick figure drawn from body-joint positions estimated from one frame, on a plain "
        "background; the joint positions can be slightly wrong in a single frame."),
    "raw_video": "You are shown {n} frames, in time order, from a video of a person walking.",
}
HEADER_SPLIT = "Do not diagnose or classify"  # where branch_d_elicit's shared instructions start


def build_prompt(rep: str, group: str, n_images: int) -> str:
    """Opening sentence for this input, then branch_d_elicit's instructions and attribute list."""
    shared = attribute_list(group)
    shared = shared[shared.index(HEADER_SPLIT):]
    return OPENING[rep].format(n=n_images) + " " + shared


def _key(rep: str, group: str, instance: int) -> str:
    return f"{rep}|{group}|{instance}"


def _done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    return {_key(r["rep"], r["group"], r["instance"])
            for r in map(json.loads, out_path.read_text().splitlines())}


def run(instances: list[int], reps: list[str] = list(REPS), out_path: Path = OUT_PATH) -> None:
    from gemma_backend import generate_fn  # needs torch and a GPU

    windows = json.loads((run_gemma.DATA_DIR / "windows.json").read_text())
    cuts = json.loads((run_gemma.DATA_DIR / "ankle_cuts.json").read_text())
    clip_meta = json.loads((run_gemma.DATA_DIR / "clip_meta.json").read_text())
    done = _done(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    todo = [(rep, group, i) for rep in reps for i in instances for group in ATTRIBUTE_GROUPS
            if _key(rep, group, i) not in done]
    print(f"{len(done)} calls on disk, {len(todo)} to run")

    clips = {}
    for n, (rep, group, instance) in enumerate(todo, start=1):
        t0 = time.time()
        if (rep, instance) not in clips:  # the three groups of a clip share one set of frames
            clips = {(rep, instance): run_gemma._v2_clip(instance, rep, 1, "window", clip_meta, windows, cuts)}
        _, images, indices, _, _ = clips[(rep, instance)]
        labels = prompts_v2.frame_labels(rep, len(images))
        raw = generate_fn(images, build_prompt(rep, group, len(images)),
                          max_new_tokens=MAX_NEW_TOKENS, labels=labels)
        ratings, status, error = parse_ratings(raw, group)
        record = {"rep": rep, "group": group, "instance": instance,
                  "weight_class": run_gemma.CLASS_OF[instance], "n_images": len(images),
                  "indices": indices, "ratings": ratings, "status": status, "error": error, "raw": raw}
        with out_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        n_rated = sum(s == "rated" for s in status.values())
        print(f"[{n}/{len(todo)}] {rep} {group} {instance}: rated {n_rated}/{len(status)} "
              f"({len(images)} img, {time.time() - t0:.0f}s)")
