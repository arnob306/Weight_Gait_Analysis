"""
Branch D, Phase 2 -- classification. See docs/branch_d_guideline.md.

Fits a logistic regression on Phase 1's elicited attribute ratings (not the
raw frames) and evaluates it under leave-one-subject-out cross-validation,
grouping folds by person_id so no participant appears on both sides of any
fold -- with only 9 participants across 30 clips, a random split would leak
the same person's gait "signature" into both train and test and inflate
the reported accuracy (guideline section 5).

Does not run until Phase 1's output (branch_d_elicit.py, run on Kaggle) has
been produced.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from branch_d_elicit import ATTRIBUTE_GROUPS
from labels import get_instance_label, load_labels

ELICITATION_PATH = Path(__file__).resolve().parent.parent / "results" / "branch_d_elicitation.json"
LABELS_CSV = Path(__file__).resolve().parent.parent / "All Joints" / "Absolute_Data.csv"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "results" / "branch_d_classification.json"

ATTRIBUTE_KEYS = [key for group in ATTRIBUTE_GROUPS.values() for key, _, _ in group]


def build_dataset(elicitation_path: Path, labels_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Returns (X, y, groups, instances). X has one row per clip and one
    column per attribute in ATTRIBUTE_KEYS order; a rating the model
    declined (null) or failed to produce (invalid) is left as NaN here and
    imputed later, inside the CV pipeline -- imputing before the split
    would leak test-fold information into the training-fold statistics."""
    records = json.loads(elicitation_path.read_text())
    labels = load_labels(str(labels_csv))

    X, y, groups, instances = [], [], [], []
    for record in records:
        X.append([
            record["ratings"].get(key) if record["ratings"].get(key) is not None else np.nan
            for key in ATTRIBUTE_KEYS
        ])
        y.append(record["weight_class"])
        groups.append(get_instance_label(labels, record["instance"])["person_id"])
        instances.append(record["instance"])

    return np.array(X, dtype=float), np.array(y, dtype=int), np.array(groups, dtype=int), instances


def build_pipeline() -> Pipeline:
    """Impute -> scale -> classify, as one pipeline so cross_val_predict
    refits the imputer and scaler inside each fold rather than once on the
    whole dataset."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("classify", LogisticRegression(max_iter=1000)),
    ])


def run(elicitation_path: Path = ELICITATION_PATH, labels_csv: Path = LABELS_CSV,
        output_path: Path = OUTPUT_PATH) -> dict:
    X, y, groups, instances = build_dataset(elicitation_path, labels_csv)
    n_subjects = len(set(groups))
    if n_subjects < 2:
        raise ValueError(f"Need at least 2 distinct subjects for leave-one-subject-out CV, found {n_subjects}")

    predictions = cross_val_predict(build_pipeline(), X, y, cv=LeaveOneGroupOut(), groups=groups)

    accuracy = float(np.mean(predictions == y))
    classes = sorted(set(y.tolist()))
    per_class = {}
    for cls in classes:
        mask = y == cls
        per_class[int(cls)] = float(np.mean(predictions[mask] == y[mask])) if mask.any() else None
    confusion = confusion_matrix(y, predictions, labels=classes).tolist()

    # Fit on all data for interpretability only -- these coefficients
    # describe what the full attribute set contributes, not a held-out
    # accuracy claim (that claim is the LOSO numbers above).
    full_pipeline = build_pipeline()
    full_pipeline.fit(X, y)
    coefficients = full_pipeline.named_steps["classify"].coef_.tolist()

    result = {
        "n_clips": len(instances),
        "n_subjects": n_subjects,
        "loso_accuracy": accuracy,
        "loso_accuracy_per_class": per_class,
        "confusion_matrix": {"labels": classes, "matrix": confusion},
        "predictions": [
            {"instance": inst, "true": int(t), "predicted": int(p)}
            for inst, t, p in zip(instances, y.tolist(), predictions.tolist())
        ],
        "coefficients": {"attributes": ATTRIBUTE_KEYS, "classes": classes, "per_class": coefficients},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(f"LOSO accuracy: {accuracy:.2f} ({int(np.sum(predictions == y))}/{len(y)})")
    print(f"Per-class accuracy: {per_class}")
    print(f"Wrote {output_path}")
    return result


if __name__ == "__main__":
    run()
