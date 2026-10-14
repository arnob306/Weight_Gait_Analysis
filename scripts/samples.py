"""
Which clips each Gemma experiment uses, and how they were chosen. One place, so
every runner and notebook cell draws the same clips.

Everything comes from one seeded stream, RandomState(27):
  1. permute the 32 participants; the first 10 are the development participants
     and the next 10 are the held-out participants, so the two sets cannot share
     a person;
  2. for each of those participants, in sorted order, and for each class, draw one
     of that participant's 20 clips of that class (instance id =
     60 * participant + 20 * class + k, k in 0..19), development first.
Each set is therefore 30 clips, 10 per class, one clip per class per participant.

That draw is the PILOT draw (30 + 30 clips): it is balanced on class but not on events
(the dev clips hold 2 freeze and 10 obstacle clips). The sets the experiments score are
the STRATIFIED sets built from it by stratified_split.py: the same participants, 36 clips
each, 12 per class and within every class 6 with no event, 3 with a freeze, 3 with an
obstacle step. Pilot clips are kept wherever they fit, so no finished run is repeated;
the few pilot clips that no longer fit stay in the pilot results only.

Development set (DEV_INSTANCES): prompt and sampling choices are made on these clips.
Held-out set (HELD_OUT_INSTANCES): reported once, after the configuration is fixed.
Ankle-cut sample: 5 clips per class from the PILOT development clips, drawn with a fresh
RandomState(27), for the small ankle-cut test; all 15 are inside the stratified dev set.

These are not the 30 clips behind the Qwen results; the Qwen runs are reported
separately as the documented failure case.
"""
import numpy as np

SEED = 27
N_PARTICIPANTS = 32
N_SET_PARTICIPANTS = 10
CLASSES = (0, 1, 2)
CLIPS_PER_PARTICIPANT_CLASS = 20
INSTANCES_PER_PARTICIPANT = 60


def instance_id(participant: int, weight_class: int, k: int) -> int:
    return INSTANCES_PER_PARTICIPANT * participant + CLIPS_PER_PARTICIPANT_CLASS * weight_class + k


def draw_sets(seed: int = SEED) -> dict:
    """The development and held-out sets, drawn from one seeded stream."""
    rng = np.random.RandomState(seed)
    order = rng.permutation(N_PARTICIPANTS).tolist()
    sets = {}
    for name, persons in (("dev", order[:N_SET_PARTICIPANTS]),
                          ("held_out", order[N_SET_PARTICIPANTS:2 * N_SET_PARTICIPANTS])):
        persons = sorted(persons)
        by_class = {c: [] for c in CLASSES}
        for p in persons:
            for c in CLASSES:
                by_class[c].append(instance_id(p, c, int(rng.randint(CLIPS_PER_PARTICIPANT_CLASS))))
        sets[name] = {"persons": persons, "by_class": by_class}
    return sets


def flatten(by_class: dict[int, list[int]]) -> list[int]:
    return [i for c in sorted(by_class) for i in by_class[c]]


def ankle_cut_sample(dev_by_class: dict[int, list[int]], per_class: int = 5, seed: int = SEED) -> list[int]:
    """Small stratified draw from the development clips: per_class clips from each
    class, drawn with one seeded stream in class order."""
    rng = np.random.RandomState(seed)
    return sorted(
        i for c in sorted(dev_by_class)
        for i in rng.choice(dev_by_class[c], per_class, replace=False).tolist()
    )


_SETS = draw_sets()
DEV_PERSONS = tuple(_SETS["dev"]["persons"])
HELD_OUT_PERSONS = tuple(_SETS["held_out"]["persons"])
DEV_BY_CLASS = _SETS["dev"]["by_class"]
HELD_OUT_BY_CLASS = _SETS["held_out"]["by_class"]

PILOT_DEV_INSTANCES = flatten(DEV_BY_CLASS)
PILOT_HELD_OUT_INSTANCES = flatten(HELD_OUT_BY_CLASS)
ANKLE_CUT_SAMPLE = ankle_cut_sample(DEV_BY_CLASS)

# Stratified sets: output of stratified_split.py (checked against it in test_gemma_v2.py).
DEV_INSTANCES = [427, 442, 467, 548, 561, 582, 670, 675, 696, 703, 745, 753, 770, 901, 919, 920, 957,
                 1210, 1217, 1220, 1247, 1328, 1351, 1369, 1563, 1584, 1597, 1607, 1620, 1654, 1674,
                 1678, 1752, 1775, 1794, 1799]
HELD_OUT_INSTANCES = [16, 20, 31, 41, 189, 199, 201, 234, 243, 274, 294, 295, 366, 392, 407, 843, 876,
                      877, 893, 1094, 1104, 1124, 1278, 1296, 1302, 1317, 1384, 1409, 1436, 1500, 1513,
                      1527, 1545, 1873, 1884, 1902]
# Every clip any experiment has used or will use, so one staging covers pilot results and new runs.
STAGED_INSTANCES = sorted(set(PILOT_DEV_INSTANCES) | set(PILOT_HELD_OUT_INSTANCES)
                          | set(DEV_INSTANCES) | set(HELD_OUT_INSTANCES))


def class_of(instance: int) -> int:
    """Weight class from the instance id (id = 60 * participant + 20 * class + k)."""
    return (instance % INSTANCES_PER_PARTICIPANT) // CLIPS_PER_PARTICIPANT_CLASS


CLASS_OF = {i: class_of(i) for i in range(N_PARTICIPANTS * INSTANCES_PER_PARTICIPANT)}
