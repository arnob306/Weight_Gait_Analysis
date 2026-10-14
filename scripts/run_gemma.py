"""
Gemma gait experiments: one runner for every stage, for a Kaggle notebook.

Inputs ("reps"):
  raw_video, skeleton, overlay   frames sampled from that video
  montage                        the raw-video frames tiled into one image
                                 (4 columns, at most 16 tiles)
  raw_ankle_cut                  raw-video frames with everything below the
                                 clip's ankle cut row blacked out
  blind                          no image at all (one call per prompt)

Frames are equally spaced at fps_values on the stored 7 fps timeline, either
inside the walker's active window (mode="window") or across the whole clip
(mode="full"). Decoding is greedy and seeded (see gemma_backend).

Two kinds of prompt:
  P0_rubric ... P3_guided_balanced   (prompts.py) the original short answer, run as before.
  Z_zero_shot, R_rubric              (prompts_v2.py) the v2 set. For these the runner also
                                     crops every frame around the walker (CROP_WIDTH px,
                                     centred on the walker, same crop for every input),
                                     caps the images per clip at MAX_IMAGES, numbers the
                                     frames, fills the timing and direction into the
                                     prompt per clip, and records the per-leg answers.

Every call is appended to the output file as one JSON line, and a rerun skips
calls already on disk, so a dropped Kaggle session loses at most one call.

Notebook usage, after the setup cells:
    import run_gemma
    from samples import DEV_INSTANCES
    run_gemma.run(DEV_INSTANCES, ["raw_video", "skeleton", "overlay", "montage"],
                  ["Z_zero_shot", "R_rubric"], [1])
    run_gemma.summarize(); run_gemma.summarize_v2()
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import prompts_v2
from frame_sampling import extract_frames, sample_frame_indices_fps
from gait_parse import (leg_truth, nine_class, parse_direction_response, parse_gait_response,
                        parse_probe_response, parse_yesno_response)
from harness import parse_model_response
from prompts import PROMPTS
from samples import CLASS_OF

DATA_DIR = Path("/kaggle/input/datasets/arnobsaha1/weightgait-pilot")
OUT_PATH = Path("/kaggle/working/results/gemma_runs.jsonl")
MONTAGE_COLS = 4
MAX_MONTAGE_TILES = 16
MAX_IMAGES = 16        # v2 only: images per clip, thinned evenly (keeps memory in check)
CROP_WIDTH = 160       # v2 only: pixels kept around the walker's centre (frames are 424 wide)
V2_MAX_NEW_TOKENS = 1000
PROBE_PATH = Path("/kaggle/working/results/gemma_probes.jsonl")
PROBE_NAME = "probe_weights"
PROBE_MAX_NEW_TOKENS = 120
PROBE_PAIR_MAX_NEW_TOKENS = 8      # one-word yes/no answers
DIRECTION_NAME = "probe_direction"
PAIR_TOPICS = ("freeze", "obstacle", "left_leg", "right_leg")
BLIND_INSTANCE = 0  # placeholder: a blind call sees no clip, so one call per prompt suffices
BLIND_INDICES = list(range(0, 35, 7))  # v2 blind call: the prompt text for a typical 5-image clip


def _video_path(instance: int, rep: str) -> Path:
    if rep in ("raw_video", "montage", "raw_ankle_cut"):
        return DATA_DIR / f"Instance_{instance}.0.mp4"
    return DATA_DIR / "kaggle_representations" / str(instance) / f"{rep}.mp4"


def _thin(indices: list[int], cap: int) -> list[int]:
    """At most cap indices, evenly spread, keeping the first and last."""
    if len(indices) <= cap:
        return indices
    return [indices[int(round(k))] for k in np.linspace(0, len(indices) - 1, cap)]


def _tile(frames: list[np.ndarray], cols: int, numbered: bool = False) -> np.ndarray:
    rows = (len(frames) + cols - 1) // cols
    h, w = frames[0].shape[:2]
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = frame
        if numbered:
            cv2.putText(canvas, str(i + 1), (c * w + 6, r * h + 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _mask_below(frame: np.ndarray, cut_row: int) -> np.ndarray:
    out = frame.copy()
    out[int(cut_row):] = 0
    return out


def _crop(frame: np.ndarray, centre_x: int, width: int = CROP_WIDTH) -> np.ndarray:
    """Full-height strip of the frame, width px wide, centred on the walker."""
    x0 = int(min(max(centre_x - width // 2, 0), max(frame.shape[1] - width, 0)))
    return frame[:, x0:x0 + width]


def sample_indices(instance: int, rep: str, fps: float, mode: str, windows: dict,
                   max_images: int | None = None) -> list[int]:
    """Stored-frame indices the model is shown for one clip under one setting."""
    cap = cv2.VideoCapture(str(_video_path(instance, rep)))
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    start, end = windows[str(instance)] if mode == "window" else (0, total - 1)
    indices = sample_frame_indices_fps(native_fps, fps, start, end)
    limit = MAX_MONTAGE_TILES if rep == "montage" else max_images
    return _thin(indices, limit) if limit else indices


def load_clip(instance: int, rep: str, fps: float, mode: str, windows: dict, cuts: dict,
              centres: list[int] | None = None, max_images: int | None = None):
    """(images, indices) for one clip. With centres (the walker's x per stored frame) every
    frame is cropped around the walker and montage tiles are numbered."""
    if rep == "blind":
        return [], []
    indices = sample_indices(instance, rep, fps, mode, windows, max_images)
    frames = extract_frames(str(_video_path(instance, rep)), indices)
    if rep == "raw_ankle_cut":
        frames = [_mask_below(f, cuts[str(instance)]) for f in frames]
    if centres is not None:
        frames = [_crop(f, centres[i]) for f, i in zip(frames, indices)]
    if rep == "montage":
        return [_tile(frames, MONTAGE_COLS, numbered=centres is not None)], indices
    return frames, indices


def load_inputs(instance: int, rep: str, fps: float, mode: str, windows: dict, cuts: dict) -> list[np.ndarray]:
    """The list of images the model is shown for one clip under one setting (original behaviour)."""
    return load_clip(instance, rep, fps, mode, windows, cuts)[0]


def _key(split, prompt, fps, mode, rep, instance) -> str:
    return f"{split}|{prompt}|{fps}|{mode}|{rep}|{instance}"


def _done_keys(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    keys = set()
    for line in out_path.read_text().splitlines():
        r = json.loads(line)
        keys.add(_key(r["split"], r["prompt"], r["fps"], r["mode"], r["rep"], r["instance"]))
    return keys


def v2_fields(parsed: dict | None, meta: dict | None, direction: str | None) -> dict:
    """What a v2 call records beyond the common fields: the model's per-leg and event answers
    next to the ground truth, so every claim can be scored later."""
    p = parsed or {}
    checks = p.get("_checks", {})
    freeze, obstacle = p.get("freeze"), p.get("obstacle")
    out = {"direction": direction, "consistent": checks.get("consistent"),
           "freeze_pred": freeze, "obstacle_pred": obstacle,
           "left_pred": p.get("left_leg_impaired"), "right_pred": p.get("right_leg_impaired"),
           "confidence": p.get("confidence"), "nine_pred": None,
           "freeze_true": None, "obstacle_true": None, "nine_true": None}
    if parsed is not None and isinstance(freeze, bool) and isinstance(obstacle, bool):
        out["nine_pred"] = nine_class(p["gait_class"], freeze, obstacle)
    if meta is not None:
        out.update(freeze_true=meta["freeze"], obstacle_true=meta["obstacle"],
                   nine_true=nine_class(meta["weight_class"], meta["freeze"], meta["obstacle"]))
    return out


def _v2_clip(instance: int, rep: str, fps: float, mode: str, clip_meta: dict, windows: dict, cuts: dict):
    """(meta, images, indices, direction, near_leg) for one clip, cropped around the walker.
    The walking window is the clip's one-pass window, so no frame is empty."""
    meta = clip_meta.get(str(instance))
    clip_windows = {str(instance): meta["window"]} if meta else windows
    images, indices = load_clip(instance, rep, fps, mode, clip_windows, cuts,
                                centres=meta["cx"] if meta else None, max_images=MAX_IMAGES)
    direction, near = (meta["direction"], meta["near_leg"]) if meta else (None, None)
    return meta, images, indices, direction, near


def run(instances: list[int], reps: list[str], prompt_names: list[str], fps_values: list[float],
        mode: str = "window", split: str = "dev", out_path: Path = OUT_PATH) -> None:
    from gemma_backend import generate_fn  # needs torch and a GPU; imported here so this module loads without them

    windows = json.loads((DATA_DIR / "windows.json").read_text())
    cuts = json.loads((DATA_DIR / "ankle_cuts.json").read_text())
    clip_meta = json.loads((DATA_DIR / "clip_meta.json").read_text()) if any(
        p in prompts_v2.VARIANTS for p in prompt_names) else {}
    done = _done_keys(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    jobs = []
    for prompt in prompt_names:
        for fps in fps_values:
            for rep in reps:
                if rep == "blind":
                    jobs.append((prompt, fps_values[0], rep, BLIND_INSTANCE))
                    continue
                jobs.extend((prompt, fps, rep, i) for i in instances)
    jobs = list(dict.fromkeys(jobs))
    todo = [j for j in jobs if _key(split, j[0], j[1], mode, j[2], j[3]) not in done]
    print(f"{len(done)} calls on disk, {len(todo)} to run")

    for n, (prompt, fps, rep, instance) in enumerate(todo, start=1):
        t0 = time.time()
        extra = {}
        if prompt in prompts_v2.VARIANTS:
            meta, images, indices, direction, near = _v2_clip(instance, rep, fps, mode, clip_meta,
                                                              windows, cuts)
            view = "raw_video" if rep == "blind" else rep
            text = prompts_v2.build_prompt(prompt, view, indices or BLIND_INDICES, direction, near)
            labels = prompts_v2.frame_labels(rep, len(images)) if images else None
            raw = generate_fn(images, text, max_new_tokens=V2_MAX_NEW_TOKENS, labels=labels)
            parsed, ok, error = parse_gait_response(raw)
            pred = parsed["gait_class"] if ok else None
            extra = {**v2_fields(parsed, meta, direction), "indices": indices}
        else:
            images = load_inputs(instance, rep, fps, mode, windows, cuts)
            raw = generate_fn(images, PROMPTS[prompt])
            parsed, ok, error = parse_model_response(raw)
            pred = parsed["weight_class"] if ok else None
        true_class = CLASS_OF.get(instance)
        record = {
            "split": split, "prompt": prompt, "fps": fps, "mode": mode, "rep": rep,
            "instance": instance, "true_class": true_class, "n_images": len(images),
            "pred": pred, "success": ok, "error": error,
            "parsed": parsed, "raw": raw, **extra,
        }
        with out_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        mark = "-" if true_class is None else ("OK" if record["pred"] == true_class else "WRONG")
        print(f"[{n}/{len(todo)}] {prompt} {fps}fps {rep} {instance}: gt={true_class} pred={record['pred']} "
              f"({mark}, {len(images)} img, {time.time() - t0:.0f}s)")


def _groups(out_path: Path) -> dict:
    groups = defaultdict(list)
    for line in out_path.read_text().splitlines():
        r = json.loads(line)
        groups[(r["split"], r["prompt"], r["fps"], r["mode"], r["rep"])].append(r)
    return groups


def summarize(out_path: Path = OUT_PATH) -> list[dict]:
    """Accuracy per setting against the 1/3 chance baseline, plus how often each
    class was predicted (a setting that predicts one class every time has
    collapsed, whatever its accuracy)."""
    from scipy.stats import binomtest

    rows = []
    for key, records in sorted(_groups(out_path).items()):
        scored = [r for r in records if r["true_class"] is not None]
        if not scored:
            continue
        correct = sum(r["pred"] == r["true_class"] for r in scored)
        preds = {c: sum(r["pred"] == c for r in scored) for c in (0, 1, 2)}
        preds["fail"] = sum(r["pred"] is None for r in scored)
        p = binomtest(correct, len(scored), 1 / 3, alternative="greater").pvalue
        rows.append({"setting": key, "n": len(scored), "correct": correct,
                     "accuracy": correct / len(scored), "p_vs_chance": p, "predicted": preds})
    print(f"{'split|prompt|fps|mode|rep':58s} {'acc':>9s} {'p':>6s}  predicted 0/1/2/fail")
    for r in rows:
        pr = r["predicted"]
        print(f"{'|'.join(map(str, r['setting'])):58s} {r['correct']:>3d}/{r['n']:<3d}{r['accuracy']:>5.0%} "
              f"{r['p_vs_chance']:>6.3f}  {pr[0]}/{pr[1]}/{pr[2]}/{pr['fail']}")
    return rows


def _rate(hits: int, total: int) -> str:
    return f"{hits}/{total}" if total else "n/a"


def v2_row(records: list[dict]) -> dict:
    """Counts behind one v2 setting: class, nine-class, leg flags, events, internal consistency."""
    ok = [r for r in records if r["success"]]
    legs = [r for r in ok if r["left_pred"] is not None and r["right_pred"] is not None
            and r["true_class"] is not None]
    legs_right = sum((r["left_pred"], r["right_pred"]) == leg_truth(r["true_class"]) for r in legs)
    nine = [r for r in ok if r["nine_pred"] is not None and r["nine_true"] is not None]
    cons = [r for r in ok if r["consistent"] is not None]
    fz = [r for r in ok if r["freeze_pred"] is not None and r["freeze_true"] is not None]
    ob = [r for r in ok if r["obstacle_pred"] is not None and r["obstacle_true"] is not None]
    return {
        "class": (sum(r["pred"] == r["true_class"] for r in ok), len(ok)),
        "nine": (sum(r["nine_pred"] == r["nine_true"] for r in nine), len(nine)),
        "legs": (legs_right, len(legs)),
        "freeze": (sum(r["freeze_pred"] == r["freeze_true"] for r in fz), len(fz)),
        "obstacle": (sum(r["obstacle_pred"] == r["obstacle_true"] for r in ob), len(ob)),
        "consistent": (sum(bool(r["consistent"]) for r in cons), len(cons)),
        "failed": len(records) - len(ok),
    }


def summarize_v2(out_path: Path = OUT_PATH) -> list[dict]:
    """One line per v2 setting. 'legs' counts clips where both leg flags are right (left impaired
    in classes 1 and 2, right in class 2); 'consistent' counts answers whose class equals the
    number of legs the model flagged; nine = 3 x class + event."""
    rows = []
    for key, records in sorted(_groups(out_path).items()):
        if key[1] not in prompts_v2.VARIANTS:
            continue
        scored = [r for r in records if r["true_class"] is not None]
        if scored:
            rows.append({"setting": key, **v2_row(scored)})
    print(f"{'split|prompt|fps|mode|rep':52s} class   nine    legs    freeze  obst    consist fail")
    for r in rows:
        cells = "  ".join(f"{_rate(*r[k]):7s}" for k in ("class", "nine", "legs", "freeze", "obstacle", "consistent"))
        print(f"{'|'.join(map(str, r['setting'])):52s} {cells} {r['failed']}")
    return rows


def _pair_present(topic: str, meta: dict) -> bool:
    """Is the topic true of this clip? Freeze and obstacle are flags in the data; the left leg is
    weighted in classes 1 and 2, the right leg in class 2 only."""
    return {"freeze": bool(meta["freeze"]), "obstacle": bool(meta["obstacle"]),
            "left_leg": meta["weight_class"] >= 1, "right_leg": meta["weight_class"] == 2}[topic]


def _probe_specs(kind: str, topics) -> list[dict]:
    """One spec per distinct call asked of each clip. A yes/no pair is two specs (question A is
    'yes' when the topic is present, question B is 'yes' when it is absent), asked in separate calls."""
    if kind == "weights":
        return [{"name": PROBE_NAME, "kind": "weights", "question": None, "tokens": PROBE_MAX_NEW_TOKENS}]
    if kind == "direction":
        return [{"name": DIRECTION_NAME, "kind": "direction", "question": None, "tokens": PROBE_MAX_NEW_TOKENS}]
    if kind == "pairs":
        return [{"name": f"probe_{t}_{side}", "kind": t, "question": q, "topic": t, "side": side,
                 "tokens": PROBE_PAIR_MAX_NEW_TOKENS}
                for t in topics for side, q in zip("AB", prompts_v2.PROBE_PAIRS[t])]
    raise ValueError(f"unknown probe kind {kind!r}; use 'weights', 'direction' or 'pairs'")


def _probe_answer(spec: dict, raw: str, meta: dict | None) -> tuple[bool, str | None, dict]:
    """(parsed ok, error, the fields this probe adds to its row)."""
    if spec["kind"] == "weights":
        parsed, ok, error = parse_probe_response(raw)
        return ok, error, {"left_attach": parsed["left_leg_attachment"] if ok else None,
                           "right_attach": parsed["right_leg_attachment"] if ok else None,
                           "frames": parsed.get("frames") if ok else None}
    if spec["kind"] == "direction":
        parsed, ok, error = parse_direction_response(raw)
        return ok, error, {"pred_direction": parsed["direction"] if ok else None,
                           "true_direction": meta["direction"] if meta else None}
    answer, ok, error = parse_yesno_response(raw)
    present = _pair_present(spec["topic"], meta)
    expected_yes = present if spec["side"] == "A" else not present
    return ok, error, {"topic": spec["topic"], "side": spec["side"], "answer": answer, "present": present,
                       "correct": bool(ok and (answer == "yes") == expected_yes)}


def run_probe(instances: list[int], reps: list[str], fps_values: list[float] = (1,),
              mode: str = "window", split: str = "dev", out_path: Path = PROBE_PATH,
              kind: str = "weights", topics=PAIR_TOPICS) -> None:
    """Probes, each asked alone and never in the same call as the classification.
      kind="weights"    is anything strapped or attached to either leg? Run it on raw_video and
                        overlay (the weights are visible) and on raw_ankle_cut (blacked out, so
                        'yes' there is a hallucination).
      kind="direction"  which way is the walker moving? A control: the answer is known for every
                        clip, so it shows the images reach the model.
      kind="pairs"      for each topic (freeze, obstacle, left_leg, right_leg) two opposite yes/no
                        questions in separate calls; see summarize_pairs."""
    from gemma_backend import generate_fn  # needs torch and a GPU

    windows = json.loads((DATA_DIR / "windows.json").read_text())
    cuts = json.loads((DATA_DIR / "ankle_cuts.json").read_text())
    clip_meta = json.loads((DATA_DIR / "clip_meta.json").read_text())
    done = _done_keys(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    todo = [(spec, fps, rep, i) for spec in _probe_specs(kind, topics) for fps in fps_values
            for rep in reps for i in instances
            if _key(split, spec["name"], fps, mode, rep, i) not in done]
    print(f"{len(done)} probe calls on disk, {len(todo)} to run")

    for n, (spec, fps, rep, instance) in enumerate(todo, start=1):
        t0 = time.time()
        meta, images, indices, direction, near = _v2_clip(instance, rep, fps, mode, clip_meta, windows, cuts)
        text = prompts_v2.build_probe(spec["kind"], rep, indices, direction, near, spec["question"])
        raw = generate_fn(images, text, max_new_tokens=spec["tokens"],
                          labels=prompts_v2.frame_labels(rep, len(images)))
        ok, error, fields = _probe_answer(spec, raw, meta)
        record = {"split": split, "prompt": spec["name"], "fps": fps, "mode": mode, "rep": rep,
                  "instance": instance, "true_class": CLASS_OF.get(instance), "n_images": len(images),
                  "success": ok, "error": error, **fields, "indices": indices, "raw": raw}
        with out_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        shown = {k: v for k, v in fields.items() if k != "frames"}
        print(f"[{n}/{len(todo)}] {spec['name']} {fps}fps {rep} {instance}: class={record['true_class']} "
              f"{shown} ({len(images)} img, {time.time() - t0:.0f}s)")


def summarize_probe(out_path: Path = PROBE_PATH) -> list[dict]:
    """Per input: how often the model says 'yes' to an attachment on a leg that carries a weight
    (left in classes 1 and 2, right in class 2) against a leg that does not. A model that sees the
    band says yes mostly on the first and rarely on the second."""
    rows = []
    for key, records in sorted(_groups(out_path).items()):
        if key[1] != PROBE_NAME:
            continue
        ok = [r for r in records if r["success"] and r["true_class"] is not None]
        sides = {}
        for side, weighted in (("left", lambda c: c >= 1), ("right", lambda c: c == 2)):
            field = f"{side}_attach"
            on = [r for r in ok if weighted(r["true_class"])]
            off = [r for r in ok if not weighted(r["true_class"])]
            sides[side] = {"yes_on_weighted": (sum(r[field] == "yes" for r in on), len(on)),
                           "yes_on_unweighted": (sum(r[field] == "yes" for r in off), len(off))}
        rows.append({"setting": key, "rep": key[4], "failed": len(records) - len(ok), **sides})
    print(f"{'rep':14s} {'left: yes on weighted':>22s} {'left: yes on unweighted':>24s} "
          f"{'right: yes on weighted':>23s} {'right: yes on unweighted':>25s} fail")
    for r in rows:
        cells = [_rate(*r[s][k]) for s in ("left", "right") for k in ("yes_on_weighted", "yes_on_unweighted")]
        print(f"{r['rep']:14s} {cells[0]:>22s} {cells[1]:>24s} {cells[2]:>23s} {cells[3]:>25s} {r['failed']}")
    return rows


def summarize_direction(out_path: Path = PROBE_PATH) -> list[dict]:
    """Direction control per input: answers that match the walker's real direction, out of the
    answers that parsed. 'unclear' counts as not correct. Near all-correct means the images reach
    the model and it can read where the walker is going."""
    rows = []
    for key, records in sorted(_groups(out_path).items()):
        if key[1] != DIRECTION_NAME:
            continue
        ok = [r for r in records if r["success"] and r["true_direction"] is not None]
        rows.append({"setting": key, "rep": key[4], "n": len(ok), "failed": len(records) - len(ok),
                     "correct": sum(r["pred_direction"] == r["true_direction"] for r in ok),
                     "unclear": sum(r["pred_direction"] == "unclear" for r in ok)})
    print(f"{'rep':14s} {'correct':>9s} {'unclear':>8s} fail")
    for r in rows:
        print(f"{r['rep']:14s} {_rate(r['correct'], r['n']):>9s} {r['unclear']:>8d} {r['failed']}")
    return rows


def summarize_pairs(out_path: Path = PROBE_PATH) -> list[dict]:
    """Hallucination protocol, per input and topic. For each clip both questions of a pair are
    asked; the clip counts as non-hallucinated on that topic only if BOTH answers are right.
    Also shows each question's own accuracy, so a model that always says 'no' (A right on absent
    clips, B wrong) is visible."""
    by_clip = defaultdict(dict)  # (rep, topic, instance) -> {"A": row, "B": row}
    for key, records in _groups(out_path).items():
        for r in records:
            if r.get("topic") and r["success"]:
                by_clip[(key[4], r["topic"], r["instance"])][r["side"]] = r
    cells = defaultdict(lambda: {"n": 0, "a": 0, "b": 0, "both": 0})
    for (rep, topic, _), sides in by_clip.items():
        if set(sides) != {"A", "B"}:
            continue
        c = cells[(rep, topic)]
        c["n"] += 1
        c["a"] += sides["A"]["correct"]
        c["b"] += sides["B"]["correct"]
        c["both"] += sides["A"]["correct"] and sides["B"]["correct"]
    rows = [{"rep": rep, "topic": topic, **c} for (rep, topic), c in sorted(cells.items())]
    print(f"{'rep':14s} {'topic':10s} {'A right':>8s} {'B right':>8s} {'both right':>11s}")
    for r in rows:
        print(f"{r['rep']:14s} {r['topic']:10s} {_rate(r['a'], r['n']):>8s} {_rate(r['b'], r['n']):>8s} "
              f"{_rate(r['both'], r['n']):>11s}")
    return rows
