import pandas as pd

WEIGHT_CLASS_NAMES = {0: "normal", 1: "limp", 2: "shuffle"}


def load_labels(csv_path: str) -> pd.DataFrame:
    """One row per instance: weight_class, gait_freeze, obstacle, person_id.

    These live in Absolute_Data.csv columns 2-5 (0-indexed), confirmed against
    the WeightGait dataset Readme and checked directly against the data: each
    column is constant within an instance and takes exactly the value range
    the Readme describes (weight_class in {0,1,2}, two boolean flags, and 32
    distinct person IDs).
    """
    df = pd.read_csv(csv_path, header=None, usecols=[0, 2, 3, 4, 5])
    df.columns = ["instance", "weight_class", "gait_freeze", "obstacle", "person_id"]
    return df.drop_duplicates(subset="instance").set_index("instance")


def get_instance_label(labels: pd.DataFrame, instance: int) -> dict:
    row = labels.loc[instance]
    return {
        "weight_class": int(row["weight_class"]),
        "gait_freeze": bool(row["gait_freeze"]),
        "obstacle": bool(row["obstacle"]),
        "person_id": int(row["person_id"]),
    }


def select_dev_set(labels: pd.DataFrame, n_per_class: int, seed: int) -> list[int]:
    """Fixed-seed sample of instance ids, balanced across weight_class.

    Per the brief: prompt iteration happens on a held-out dev subset, never
    on the test set -- this is that subset, not a final evaluation sample.
    """
    counts = labels["weight_class"].value_counts()
    insufficient = counts[counts < n_per_class]
    if not insufficient.empty:
        raise ValueError(
            f"n_per_class={n_per_class} exceeds available clips for weight_class(es) "
            f"{insufficient.index.tolist()} (have {insufficient.tolist()})"
        )

    chosen = (
        labels.reset_index()
        .groupby("weight_class", group_keys=False)
        .apply(lambda g: g.sample(n=n_per_class, random_state=seed))
    )
    return sorted(chosen["instance"].tolist())
