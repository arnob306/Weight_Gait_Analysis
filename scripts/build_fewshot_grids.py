"""
Render the worked-example grids for the few-shot prompt: one image per example and input,
all of the example's frames tiled (4 columns, numbered), made by the same sampling, crop
and tiling code the runner uses for a clip.

  fewshot/<instance>_raw_video.png   also used for the montage input
  fewshot/<instance>_skeleton.png
  fewshot/<instance>_overlay.png

Run after prepare_kaggle_upload.py, from the repo root:  python scripts/build_fewshot_grids.py
"""
import json
from pathlib import Path

import cv2

import run_gemma
from samples import FEWSHOT_INSTANCES

ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / "kaggle_upload"
GRID_REPS = ("raw_video", "skeleton", "overlay")
FPS = 1


def build_grids(stage: Path = STAGE) -> list[Path]:
    run_gemma.DATA_DIR = stage
    cuts = json.loads((stage / "ankle_cuts.json").read_text())
    clip_meta = json.loads((stage / "clip_meta.json").read_text())
    out_dir = stage / "fewshot"
    out_dir.mkdir(exist_ok=True)
    written = []
    for instance in FEWSHOT_INSTANCES:
        meta = clip_meta[str(instance)]
        for rep in GRID_REPS:
            frames, indices = run_gemma.load_clip(
                instance, rep, FPS, "window", {str(instance): meta["window"]}, cuts,
                centres=meta["cx"], max_images=run_gemma.MAX_IMAGES)
            grid = run_gemma._tile(frames, run_gemma.MONTAGE_COLS, numbered=True)
            path = out_dir / f"{instance}_{rep}.png"
            cv2.imwrite(str(path), grid)
            written.append(path)
            print(f"{path.name}: {len(frames)} frames, stored indices {indices}")
    return written


if __name__ == "__main__":
    build_grids()
