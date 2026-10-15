"""Local checks for the Gemma v2 pieces: parser, prompt blocks, clip facts, side colours.
Run from scripts/:  python -m pytest test_gemma_v2.py -q"""
import json

import numpy as np
import pytest

import clip_meta
import gait_parse
import prompts_v2 as pv
import skeleton_utils as su


def _answer(**over) -> dict:
    leg = {"visibility": "clear", "step_length": "short", "foot_lift": "low",
           "knee_bend_in_swing": "reduced",
           "evidence_frames": [1, 2], "description": "short low steps"}
    base = {"left_leg": leg, "right_leg": dict(leg, step_length="typical"), "arm_swing": "typical",
            "rhythm": "steady", "comparison": "left differs", "freeze": False, "freeze_frames": [],
            "obstacle": False, "obstacle_frames": [], "left_leg_impaired": True,
            "right_leg_impaired": False, "gait_class": 1, "confidence": "medium"}
    return {**base, **over}


# ---- parser ----
def test_parse_valid_answer_is_consistent():
    parsed, ok, err = gait_parse.parse_gait_response(json.dumps(_answer()))
    assert ok and err is None and parsed["gait_class"] == 1
    assert parsed["_checks"] == {"missing_fields": [], "invalid_values": [], "consistent": True}


def test_parse_accepts_fenced_json():
    raw = "```json\n" + json.dumps(_answer()) + "\n```"
    assert gait_parse.parse_gait_response(raw)[1]


def test_parse_flags_class_that_disagrees_with_leg_flags():
    parsed, ok, _ = gait_parse.parse_gait_response(json.dumps(_answer(gait_class=2)))
    assert ok and parsed["_checks"]["consistent"] is False


@pytest.mark.parametrize("bad", ["one", True, 3, None, "1"])
def test_parse_rejects_non_integer_class(bad):
    assert not gait_parse.parse_gait_response(json.dumps(_answer(gait_class=bad)))[1]


def test_parse_missing_flags_gives_unknown_consistency_not_failure():
    answer = _answer()
    del answer["left_leg_impaired"]
    parsed, ok, _ = gait_parse.parse_gait_response(json.dumps(answer))
    assert ok and parsed["_checks"]["consistent"] is None
    assert "left_leg_impaired" in parsed["_checks"]["missing_fields"]


def test_parse_records_value_outside_allowed_list():
    answer = _answer()
    answer["left_leg"]["step_length"] = "very short"
    parsed, ok, _ = gait_parse.parse_gait_response(json.dumps(answer))
    assert ok and parsed["_checks"]["invalid_values"] == ["left_leg.step_length='very short'"]


@pytest.mark.parametrize("raw", [None, "no json here", '{"weight_class": 1}', "{bad json"])
def test_parse_failures_return_reason(raw):
    parsed, ok, err = gait_parse.parse_gait_response(raw)
    assert parsed is None and not ok and err


def test_nine_class_and_leg_truth():
    assert gait_parse.nine_class(0, False, False) == 0
    assert gait_parse.nine_class(1, True, False) == 4
    assert gait_parse.nine_class(2, False, True) == 8
    assert [gait_parse.leg_truth(c) for c in (0, 1, 2)] == [(False, False), (True, False), (True, True)]


# ---- prompts ----
IDX = [5, 12, 19, 26]
REPS = ["raw_video", "skeleton", "overlay", "montage", "raw_ankle_cut"]


def test_rubric_prompt_is_zero_shot_plus_exactly_block_d():
    z = pv.build_prompt("Z_zero_shot", "raw_video", IDX, "left", "left")
    r = pv.build_prompt("R_rubric", "raw_video", IDX, "left", "left")
    assert r.replace(pv.BLOCK_D + "\n\n", "") == z


def test_inputs_differ_only_in_block_b():
    base = pv.build_prompt("R_rubric", "raw_video", IDX, "left", "left")
    for rep in REPS[1:]:
        other = pv.build_prompt("R_rubric", rep, IDX, "left", "left")
        fill = pv._fill(rep, IDX, "left", "left")
        swapped = other.replace(pv.BLOCK_B[rep].format(**fill), pv.BLOCK_B["raw_video"].format(**fill))
        assert swapped == base, rep


def test_timing_and_direction_are_filled_per_clip():
    text = pv.build_prompt("Z_zero_shot", "raw_video", IDX, "right", "right")
    assert "about 0.25 seconds apart" in text and "about 0.7 seconds" in text
    assert "moves toward the right of the image, so the walker's right leg is the one nearer" in text
    assert "moves toward" not in pv.build_prompt("Z_zero_shot", "raw_video", IDX, None, None)


def test_old_vocabulary_is_gone():
    for variant in ("Z_zero_shot", "R_rubric"):
        for rep in REPS:
            text = pv.build_prompt(variant, rep, IDX, "left", "left").lower()
            for word in ("weight_class", "limp", "shuffl", "hesitation", "antalgic", "parkinson",
                         "source", "do not default"):
                assert word not in text, (variant, rep, word)


def test_output_format_lists_every_checked_value():
    for values in list(gait_parse.LEG_ENUMS.values()) + list(gait_parse.TOP_ENUMS.values()):
        for v in values:
            assert f'"{v}"' in pv.BLOCK_F


def test_every_rubric_sign_has_a_field_and_no_field_lacks_a_sign():
    signs = {"Step length": "step_length", "Foot lift": "foot_lift", "Knee bend": "knee_bend_in_swing"}
    for label, field in signs.items():
        assert f"- {label}:" in pv.BLOCK_D and field in gait_parse.LEG_ENUMS
    assert set(gait_parse.LEG_ENUMS) - set(signs.values()) == {"visibility"}
    assert "bearing" not in pv.BLOCK_D and "bearing" not in pv.BLOCK_F
    assert "the same six fields" in pv.BLOCK_F


def test_frame_labels_and_montage_numbers():
    assert pv.frame_labels("raw_video", 3) == ["Frame 1 of 3", "Frame 2 of 3", "Frame 3 of 3"]
    assert pv.frame_labels("montage", 3) is None
    assert "2 x 4 grid" in pv.build_prompt("Z_zero_shot", "montage", list(range(0, 56, 7)), "left", "left")


def test_few_shot_prompt_differs_from_rubric_by_block_e_only():
    s = pv.build_prompt("S_few_shot", "raw_video", IDX, "left", "left")
    r = pv.build_prompt("R_rubric", "raw_video", IDX, "left", "left")
    assert pv.BLOCK_E in s and pv.BLOCK_E not in r
    assert s.replace(pv.BLOCK_E + "\n\n", "") == r
    assert s.index(pv.BLOCK_D) < s.index(pv.BLOCK_E) < s.index(pv.BLOCK_F)


# ---- worked examples (prompt S) ----
def test_example_answers_parse_and_are_consistent():
    import fewshot_examples as fe
    for instance in fe.ANSWERS:
        parsed, ok, err = gait_parse.parse_gait_response(fe.answer_text(instance))
        assert ok and err is None and parsed["_checks"]["consistent"], instance


def test_examples_cover_every_class_and_both_events_from_unused_participants():
    import fewshot_examples as fe
    import samples
    assert list(fe.ANSWERS) == samples.FEWSHOT_INSTANCES
    assert {samples.CLASS_OF[i] for i in fe.ANSWERS} == {0, 1, 2}
    assert any(a["freeze"] for a in fe.ANSWERS.values()) and any(a["obstacle"] for a in fe.ANSWERS.values())
    for instance, answer in fe.ANSWERS.items():
        assert answer["gait_class"] == samples.CLASS_OF[instance]
        assert instance // samples.INSTANCES_PER_PARTICIPANT not in set(samples.DEV_PERSONS) | set(samples.HELD_OUT_PERSONS)
        assert instance in samples.STAGED_INSTANCES
        assert instance not in samples.DEV_INSTANCES and instance not in samples.HELD_OUT_INSTANCES


def test_example_items_order_and_inputs():
    import fewshot_examples as fe
    asked = []
    items = fe.example_items("montage", lambda i, g: asked.append((i, g)) or f"img{i}")
    assert [t for t, _ in items][0].startswith("WORKED EXAMPLES") and "someone else" in items[0][0]
    assert [img for _, img in items if img is not None] == ["img975", "img1712", "img343"]
    assert asked == [(975, "raw_video"), (1712, "raw_video"), (343, "raw_video")]
    assert items[2][0].startswith("Example 1, correct answer:") and items[-1][1] is None
    fe.example_items("skeleton", lambda i, g: asked.append((i, g)))
    assert asked[-1] == (343, "skeleton")


def test_example_grids_rendered_for_each_input(tmp_path):
    import build_fewshot_grids as bg
    import samples
    stage = bg.STAGE
    if not (stage / "fewshot").exists():
        pytest.skip("kaggle_upload not staged")
    for instance in samples.FEWSHOT_INSTANCES:
        for rep in bg.GRID_REPS:
            assert (stage / "fewshot" / f"{instance}_{rep}.png").stat().st_size > 1000


def test_unknown_names_raise():
    with pytest.raises(ValueError):
        pv.build_prompt("Q", "raw_video", IDX, None, None)
    with pytest.raises(ValueError):
        pv.build_prompt("Z_zero_shot", "depth", IDX, None, None)


def test_probes():
    assert "strapped or attached" in pv.build_probe("weights", "raw_video", IDX, "left", "left")
    q = pv.PROBE_PAIRS["freeze"][0]
    text = pv.build_probe("freeze", "skeleton", IDX, "left", "left", q)
    assert text.endswith(f"{q} {pv.PROBE_YES_NO}") and "stick figure" in text
    with pytest.raises(ValueError):
        pv.build_probe("freeze", "skeleton", IDX, "left", "left", "unlisted question")


# ---- weights probe ----
def test_probe_parser_accepts_fenced_json_and_records_values():
    raw = '```json\n{"left_leg_attachment": "yes", "right_leg_attachment": "no", "frames": [1, 2]}\n```'
    parsed, ok, err = gait_parse.parse_probe_response(raw)
    assert ok and err is None and parsed["left_leg_attachment"] == "yes" and parsed["frames"] == [1, 2]


@pytest.mark.parametrize("raw", [None, "no json", "{bad", '{"left_leg_attachment": "yes"}',
                                 '{"left_leg_attachment": "maybe", "right_leg_attachment": "no"}',
                                 '{"left_leg_attachment": true, "right_leg_attachment": "no"}'])
def test_probe_parser_rejects_bad_answers(raw):
    parsed, ok, err = gait_parse.parse_probe_response(raw)
    assert parsed is None and not ok and err


def test_probe_summary_counts_yes_by_truth(tmp_path):
    import run_gemma
    rows = [
        {"rep": "raw_video", "true_class": 1, "success": True, "left_attach": "yes", "right_attach": "no"},
        {"rep": "raw_video", "true_class": 0, "success": True, "left_attach": "no", "right_attach": "yes"},
        {"rep": "raw_video", "true_class": 2, "success": False, "left_attach": None, "right_attach": None},
    ]
    path = tmp_path / "p.jsonl"
    path.write_text("\n".join(json.dumps({"split": "dev", "prompt": "probe_weights", "fps": 1,
                                          "mode": "window", **r}) for r in rows))
    (out,) = run_gemma.summarize_probe(path)
    assert out["left"] == {"yes_on_weighted": (1, 1), "yes_on_unweighted": (0, 1)}
    assert out["right"] == {"yes_on_weighted": (0, 0), "yes_on_unweighted": (1, 2)}
    assert out["failed"] == 1


# ---- direction control ----
def test_direction_probe_never_leaks_the_answer():
    for rep in ("raw_video", "overlay", "skeleton"):
        text = pv.build_probe("direction", rep, IDX, None, None)
        assert "moves toward" not in text and "nearer the camera" not in text
        assert "toward the left side or toward the right side" in text and '"direction"' in text


@pytest.mark.parametrize("raw,expected", [('{"direction": "left"}', "left"),
                                          ('```json\n{"direction": "right"}\n```', "right"),
                                          ('{"direction": "unclear"}', "unclear")])
def test_direction_parser_accepts_allowed_values(raw, expected):
    parsed, ok, err = gait_parse.parse_direction_response(raw)
    assert ok and err is None and parsed["direction"] == expected


@pytest.mark.parametrize("raw", [None, "left", "{bad", '{"direction": "up"}', '{"heading": "left"}'])
def test_direction_parser_rejects_bad_answers(raw):
    parsed, ok, err = gait_parse.parse_direction_response(raw)
    assert parsed is None and not ok and err


def test_direction_summary_scores_against_clip_direction(tmp_path):
    import run_gemma
    base = {"split": "dev", "prompt": "probe_direction", "fps": 1, "mode": "window", "rep": "raw_video"}
    rows = [dict(base, success=True, true_class=0, pred_direction="left", true_direction="left"),
            dict(base, success=True, true_class=1, pred_direction="right", true_direction="left"),
            dict(base, success=True, true_class=2, pred_direction="unclear", true_direction="right"),
            dict(base, success=False, true_class=0, pred_direction=None, true_direction="left")]
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    (out,) = run_gemma.summarize_direction(path)
    assert out["correct"] == 1 and out["n"] == 3 and out["unclear"] == 1 and out["failed"] == 1


# ---- stratified split ----
import os  # noqa: E402

import samples  # noqa: E402

_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "All Joints", "Absolute_Data.csv")
needs_csv = pytest.mark.skipif(not os.path.exists(_CSV), reason="joint CSV is only on the local machine")


def test_pilot_draw_is_unchanged():
    assert len(samples.PILOT_DEV_INSTANCES) == 30 and len(samples.PILOT_HELD_OUT_INSTANCES) == 30
    assert samples.PILOT_DEV_INSTANCES[:3] == [427, 548, 675]
    assert samples.ANKLE_CUT_SAMPLE == [442, 548, 582, 675, 745, 919, 920, 957, 1217, 1220, 1369, 1584,
                                        1607, 1678, 1752]


def test_stratified_sets_are_sized_disjoint_and_keep_finished_work():
    dev, held = samples.DEV_INSTANCES, samples.HELD_OUT_INSTANCES
    assert len(dev) == len(held) == 36 and len(set(dev)) == 36 and len(set(held)) == 36
    for ids in (dev, held):
        assert [sum(samples.class_of(i) == c for i in ids) for c in (0, 1, 2)] == [12, 12, 12]
    persons = lambda ids: {i // samples.INSTANCES_PER_PARTICIPANT for i in ids}
    assert persons(dev) == set(samples.DEV_PERSONS) and persons(held) == set(samples.HELD_OUT_PERSONS)
    assert not persons(dev) & persons(held)
    assert set(samples.ANKLE_CUT_SAMPLE) <= set(dev)               # the 2 fps and ankle-cut runs stay valid
    assert len(set(dev) & set(samples.PILOT_DEV_INSTANCES)) == 26   # 26 of 36 dev clips already run
    assert set(samples.STAGED_INSTANCES) >= set(samples.PILOT_DEV_INSTANCES) | set(dev) | set(held)


def test_class_of_matches_the_pilot_draw():
    for c, ids in samples.DEV_BY_CLASS.items():
        assert all(samples.class_of(i) == c for i in ids)


@needs_csv
def test_stratified_sets_are_balanced_on_class_and_event_and_match_the_script():
    import stratified_split as ss
    from labels import load_labels
    labels = load_labels(os.path.abspath(_CSV))
    want = {(c, e): q for c in (0, 1, 2) for e, q in ss.EVENT_QUOTA.items()}
    for ids in (samples.DEV_INSTANCES, samples.HELD_OUT_INSTANCES):
        assert ss.composition(labels, ids) == want
    built = ss.build_sets()
    assert built["dev"] == sorted(samples.DEV_INSTANCES) and built["held_out"] == sorted(samples.HELD_OUT_INSTANCES)


# ---- yes/no pairs ----
@pytest.mark.parametrize("raw,expected", [("Yes", "yes"), ("no.", "no"), ('"Yes"\n', "yes"),
                                          ("YES, it does", "yes"), ("**No**", "no")])
def test_yesno_parser_accepts_one_word_answers(raw, expected):
    assert gait_parse.parse_yesno_response(raw) == (expected, True, None)


@pytest.mark.parametrize("raw", [None, "", "   ", "maybe", "I think yes", "unclear"])
def test_yesno_parser_rejects_everything_else(raw):
    value, ok, err = gait_parse.parse_yesno_response(raw)
    assert value is None and not ok and err


def test_pair_answers_are_scored_against_the_clip():
    import run_gemma
    meta = {"freeze": True, "obstacle": False, "weight_class": 1}
    spec = {"kind": "freeze", "topic": "freeze"}
    # question A is 'yes' when the topic is present; question B is 'yes' when it is absent
    assert run_gemma._probe_answer({**spec, "side": "A"}, "Yes", meta)[2]["correct"] is True
    assert run_gemma._probe_answer({**spec, "side": "A"}, "No", meta)[2]["correct"] is False
    assert run_gemma._probe_answer({**spec, "side": "B"}, "No", meta)[2]["correct"] is True
    assert run_gemma._probe_answer({**spec, "side": "B"}, "Yes", meta)[2]["correct"] is False
    legs = {"kind": "left_leg", "topic": "left_leg", "side": "A"}
    assert run_gemma._probe_answer(legs, "yes", meta)[2]["present"] is True       # class 1: left weighted
    assert run_gemma._probe_answer({**legs, "topic": "right_leg"}, "yes", meta)[2]["present"] is False


def test_pair_summary_needs_both_answers_for_a_clip_to_count(tmp_path):
    import run_gemma
    base = {"split": "dev", "fps": 1, "mode": "window", "rep": "raw_video", "success": True,
            "topic": "freeze", "true_class": 0}
    rows = [dict(base, prompt="probe_freeze_A", instance=1, side="A", correct=True),
            dict(base, prompt="probe_freeze_B", instance=1, side="B", correct=True),
            dict(base, prompt="probe_freeze_A", instance=2, side="A", correct=True),
            dict(base, prompt="probe_freeze_B", instance=2, side="B", correct=False),
            dict(base, prompt="probe_freeze_A", instance=3, side="A", correct=True)]  # B never asked
    path = tmp_path / "pairs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    (out,) = run_gemma.summarize_pairs(path)
    assert (out["n"], out["a"], out["b"], out["both"]) == (2, 2, 1, 1)


def test_probe_specs():
    import run_gemma
    names = [s["name"] for s in run_gemma._probe_specs("pairs", ("freeze", "left_leg"))]
    assert names == ["probe_freeze_A", "probe_freeze_B", "probe_left_leg_A", "probe_left_leg_B"]
    assert [s["name"] for s in run_gemma._probe_specs("direction", ())] == ["probe_direction"]
    with pytest.raises(ValueError):
        run_gemma._probe_specs("nonsense", ())


# ---- clip facts ----
def test_walking_direction():
    assert clip_meta.walking_direction([100, 140, 180, 220]) == "right"
    assert clip_meta.walking_direction([300, 250, None, 150]) == "left"
    assert clip_meta.walking_direction([200, 205, 198, 203]) is None
    assert clip_meta.walking_direction([None, None]) is None


def test_longest_run_picks_one_pass_and_bridges_short_gaps():
    seen = [False] * 2 + [True] * 5 + [False] * 8 + [True] * 12 + [False] * 3
    assert clip_meta.longest_run(seen) == (15, 26)
    bridged = [True] * 4 + [False] * 2 + [True] * 4
    assert clip_meta.longest_run(bridged) == (0, 9)
    assert clip_meta.longest_run([False] * 5) == (0, 4)


def _frame(x):
    return ([(x, 5)] * 17, [True] * 17)


def test_walking_window_keeps_active_window_when_walker_stays_in_view():
    frames = [_frame(100 + 5 * k) for k in range(20)]
    assert clip_meta.walking_window(frames, (2, 17)) == (2, 17)


def test_walking_window_uses_one_pass_when_walker_leaves_and_returns():
    empty = ([(0, 0)] * 17, [False] * 17)
    frames = [_frame(50)] * 3 + [empty] * 10 + [_frame(80)] * 12
    assert clip_meta.walking_window(frames, (0, 24)) == (13, 24)


def test_centre_series_carries_forward_over_missed_frames():
    joints = [(100, 5), (120, 5)]
    seen = (joints, [True, True])
    missed = (joints, [False, False])
    assert clip_meta.centre_series([missed, seen, missed]) == [212, 110, 110]


# ---- side colours ----
def _joints():
    pts = [(0, 0)] * 17
    pts[13], pts[15] = (50, 100), (50, 140)   # left knee, left foot
    pts[14], pts[16] = (80, 100), (80, 140)   # right knee, right foot
    return pts, [True] * 17


def test_by_side_colours_left_blue_right_red():
    joints, valid = _joints()
    canvas = np.zeros((200, 200, 3), np.uint8)
    out = su.draw_skeleton(canvas, joints, valid, by_side=True)
    assert tuple(out[120, 50]) == su.BLUE and tuple(out[120, 80]) == su.RED


def test_default_drawing_is_unchanged():
    joints, valid = _joints()
    out = su.draw_skeleton(np.zeros((200, 200, 3), np.uint8), joints, valid)
    assert tuple(out[120, 50]) == (255, 255, 255) and tuple(out[120, 80]) == (255, 255, 255)
