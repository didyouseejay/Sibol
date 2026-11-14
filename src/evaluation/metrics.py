"""Classification metrics and result files, shared by all SIBOL models.

scikit-learn is used here for the *metrics only* (never as a classifier).
Macro averages weight every class equally, which matters because the
Flowers-102 test split is imbalanced (20-238 images per class).
"""

from __future__ import annotations

import csv
import json
import platform
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> dict:
    """Top-1 accuracy and macro precision / recall / F1 over ``n_classes`` labels.

    Classes that are never predicted get precision 0 (``zero_division=0``).
    """
    labels = list(range(n_classes))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        "n_samples": int(len(y_true)),
        "top1_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
    }


def per_class_metrics(y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]) -> list[dict]:
    """Precision, recall, F1 and support for every class, one dict per class."""
    labels = list(range(len(class_names)))
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return [
        {"class_id": i, "class_name": class_names[i], "precision": float(p[i]), "recall": float(r[i]),
         "f1": float(f1[i]), "support": int(support[i])}
        for i in labels
    ]


def confusion(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> np.ndarray:
    """``(n_classes, n_classes)`` counts; rows = true class, columns = predicted class."""
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


# ---------------------------------------------------------------------------
# Result files
# ---------------------------------------------------------------------------
def _to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj: Mapping, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_builtin(obj), indent=2), encoding="utf-8")
    return path


def save_rows_csv(rows: Sequence[Mapping], path: Path, fieldnames: Optional[Sequence[str]] = None) -> Path:
    """Write a list of dicts as CSV (columns from the first row unless given)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fieldnames or rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_to_builtin(dict(row)))
    return path


def save_matrix_csv(matrix: np.ndarray, path: Path, labels: Sequence[str]) -> Path:
    """Write a square matrix with row/column labels as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true \\ predicted", *labels])
        for name, row in zip(labels, matrix):
            writer.writerow([name, *[int(v) for v in row]])
    return path


def environment_info() -> dict:
    """Versions of the main libraries, for the results file."""
    import sklearn

    return {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__,
            "platform": platform.platform(), "processor": platform.processor() or "unknown"}
