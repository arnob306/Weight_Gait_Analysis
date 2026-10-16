"""
The 27 measured gait attributes for the 36 dev clips, computed from the joint data over the
same walking window the Gemma runs use. These are the values the Branch D ratings are checked
against (branch_d_gemma_analysis.py).

Run from scripts/:  python branch_d_measured_dev.py
Writes results/measured_dev36.csv.
"""
from pathlib import Path

import pandas as pd

import measured_by_class as mbc
from samples import CLASS_OF, DEV_INSTANCES

OUT_PATH = Path(__file__).resolve().parent.parent / "results" / "measured_dev36.csv"


def main() -> None:
    mbc.cache_joint_table()
    rows = [{"instance": i, "weight_class": CLASS_OF[i], **mbc.clip_features(i, CLASS_OF[i])}
            for i in DEV_INSTANCES]
    table = pd.DataFrame(rows)
    table.to_csv(OUT_PATH, index=False)
    print(f"{len(table)} clips, {table.shape[1] - 2} attributes -> {OUT_PATH}")
    missing = table.isna().sum()
    print(missing[missing > 0].to_string() if (missing > 0).any() else "no missing values")


if __name__ == "__main__":
    main()
