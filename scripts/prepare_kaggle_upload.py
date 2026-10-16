"""
Stage everything the Gemma notebook needs, in the Kaggle dataset's own folder
layout, under kaggle_upload/. Upload the CONTENTS of that folder to the
weightgait-pilot dataset as a new version.

What it builds:
  windows.json       active frame window [start, end] for every staged clip (the same
                     function that produced the windows baked into the Qwen runs,
                     which it was checked to reproduce exactly on all 30)
  ankle_cuts.json    ankle cut row for every staged clip (see ankle_cut.py)
  clip_meta.json     walking window, direction, near leg, walker centre, class, events
  Instance_<id>.0.mp4                       raw video for every staged clip
  kaggle_representations/<id>/skeleton.mp4  skeleton (rendered here)
  kaggle_representations/<id>/overlay.mp4   overlay (rendered here)
  *.py                                      the scripts the notebook imports

Every clip in samples.STAGED_INSTANCES is staged: the pilot draw (30 development, 30
held-out, seed 27) and the stratified dev and held-out sets built from it, so finished
pilot runs and new runs use one dataset. The montage is built at run time, so no montage
files are staged.

Run from the repo root:  python scripts/prepare_kaggle_upload.py
"""
import json
import os
import shutil
from pathlib import Path

import skeleton_utils as su
from ankle_cut import CSV_PATH, ankle_cut_row
from clip_meta import build_clip_meta
from labels import load_labels
from representations import render_overlay_video, render_skeleton_video
from samples import STAGED_INSTANCES

ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / "kaggle_upload"
SCRIPTS = ["frame_sampling.py", "prompts.py", "gemma_backend.py", "run_gemma.py", "samples.py",
           "harness.py", "experiment.py",  # the last two are imported by run_gemma and prompts
           "prompts_v2.py", "gait_parse.py",
           "fewshot_examples.py",
           "branch_d_gemma.py", "branch_d_elicit.py", "run_kaggle_qwen.py"]  # Branch D (the last two are imported by the first)  # prompt S: the worked-example answers; their grids are fewshot/*.png  # the v2 prompt set and its parser (imported by run_gemma)


def stage_clip(instance: int, joints) -> None:
    shutil.copy(ROOT / "All RGB Videos" / f"Instance_{instance}.0.mp4", STAGE / f"Instance_{instance}.0.mp4")
    rep_dir = STAGE / "kaggle_representations" / str(instance)
    rep_dir.mkdir(parents=True, exist_ok=True)
    # left limbs blue, right limbs red, head and trunk grey: the colours prompts_v2 describes
    render_skeleton_video(instance, str(rep_dir / "skeleton.mp4"), instance_joints=joints, by_side=True)
    render_overlay_video(instance, str(rep_dir / "overlay.mp4"), instance_joints=joints, by_side=True)


def main() -> None:
    os.chdir(ROOT)  # representations.py reads raw videos by relative path
    shutil.rmtree(STAGE, ignore_errors=True)  # drop clips staged for an earlier draw
    STAGE.mkdir()

    labels = load_labels(str(CSV_PATH))
    windows, cuts, meta = {}, {}, {}
    for instance in STAGED_INSTANCES:
        joints = su.load_instance_joints(str(CSV_PATH), instance)
        windows[str(instance)] = list(su.active_frame_range(joints))
        cuts[str(instance)] = ankle_cut_row(CSV_PATH, instance)
        meta[str(instance)] = build_clip_meta(instance, str(CSV_PATH), labels)
        stage_clip(instance, joints)
    (STAGE / "windows.json").write_text(json.dumps(windows, indent=1))
    (STAGE / "ankle_cuts.json").write_text(json.dumps(cuts, indent=1))
    (STAGE / "clip_meta.json").write_text(json.dumps(meta))

    for name in SCRIPTS:
        shutil.copy(ROOT / "scripts" / name, STAGE / name)

    files = [p for p in STAGE.rglob("*") if p.is_file()]
    total_mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"staged {len(files)} files, {total_mb:.1f} MB in {STAGE}")


if __name__ == "__main__":
    main()
