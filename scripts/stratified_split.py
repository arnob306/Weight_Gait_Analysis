"""
Stratified dev and held-out sets, balanced on class and event.

Each set has 12 clips per class, and within every class 6 with no event, 3 with a freeze
and 3 with an obstacle step (36 clips; 18 / 9 / 9 by event, the 50 / 25 / 25 rates the
prompt states). The participants are the ones of the original seed-27 draw (samples.py),
so the two sets still share nobody. To keep finished work, clips of the original draw are
kept wherever they fit a cell (clips in `protected` first); the remaining places are filled
from the same participants' other clips, taking the least-used participant first, with a
seeded shuffle so the result is reproducible.

Run from the repo root:  python scripts/stratified_split.py
It prints the composition and writes results/stratified_split.json; samples.py holds the
same lists as literals and test_gemma_v2.py checks that the two agree.
"""
import collections
import json
import random
from pathlib import Path

from labels import load_labels

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "All Joints" / "Absolute_Data.csv"
SEED = 27
CLASSES = (0, 1, 2)
EVENT_QUOTA = {"none": 6, "freeze": 3, "obstacle": 3}  # per class


def event_of(labels, instance: int) -> str:
    """'freeze', 'obstacle' or 'none'. The dataset never has both in one clip."""
    if labels.loc[instance, "gait_freeze"]:
        return "freeze"
    return "obstacle" if labels.loc[instance, "obstacle"] else "none"


def stratify(labels, original: list[int], persons, protected=(), seed: int = SEED) -> list[int]:
    """The stratified set for these participants (see the module docstring)."""
    rng = random.Random(seed)
    person = lambda i: int(labels.loc[i, "person_id"])
    cls = lambda i: int(labels.loc[i, "weight_class"])
    pool = [i for i in labels.index if person(i) in persons]
    chosen: list[int] = []
    for c in CLASSES:
        for event, quota in EVENT_QUOTA.items():
            fits = sorted((i for i in original if cls(i) == c and event_of(labels, i) == event),
                          key=lambda i: (i not in protected, i))
            take = fits[:quota]
            uses = collections.Counter(person(i) for i in chosen + take)
            rest = [i for i in pool if cls(i) == c and event_of(labels, i) == event and i not in original]
            rng.shuffle(rest)
            while len(take) < quota:
                rest.sort(key=lambda i: uses[person(i)])  # stable: keeps the shuffled order on ties
                pick = rest.pop(0)
                take.append(pick)
                uses[person(pick)] += 1
            chosen += take
    return sorted(chosen)


def composition(labels, instances: list[int]) -> dict:
    return dict(sorted(collections.Counter(
        (int(labels.loc[i, "weight_class"]), event_of(labels, i)) for i in instances).items()))


def build_sets() -> dict:
    """{"dev": [...], "held_out": [...]} from the original draw's participants and clips."""
    import samples
    labels = load_labels(str(CSV_PATH))
    return {
        "dev": stratify(labels, list(samples.PILOT_DEV_INSTANCES), set(samples.DEV_PERSONS),
                        protected=set(samples.ANKLE_CUT_SAMPLE)),
        "held_out": stratify(labels, list(samples.PILOT_HELD_OUT_INSTANCES), set(samples.HELD_OUT_PERSONS)),
    }


def main() -> None:
    import samples
    labels = load_labels(str(CSV_PATH))
    sets = build_sets()
    for name, pilot in (("dev", samples.PILOT_DEV_INSTANCES), ("held_out", samples.PILOT_HELD_OUT_INSTANCES)):
        new = sets[name]
        print(f"{name}: {len(new)} clips, {len(set(new) & set(pilot))} from the original draw, "
              f"{len(set(new) - set(pilot))} new, original clips dropped {sorted(set(pilot) - set(new))}")
        print("  (class, event) counts:", composition(labels, new))
    out = ROOT / "results" / "stratified_split.json"
    out.write_text(json.dumps(sets))
    print("wrote", out)


if __name__ == "__main__":
    main()
