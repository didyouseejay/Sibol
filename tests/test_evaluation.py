"""Tests for src/evaluation (metrics, result files, figures)."""

import csv
import json

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from src.evaluation import metrics as M
from src.evaluation import plots as P


def test_metrics_on_known_example():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    m = M.classification_metrics(y_true, y_pred, n_classes=3)
    assert m["top1_accuracy"] == pytest.approx(4 / 6)
    # per class precision: 0 -> 1/2, 1 -> 2/3, 2 -> 1; recall: 0 -> 1/2, 1 -> 1, 2 -> 1/2
    assert m["macro_precision"] == pytest.approx((1 / 2 + 2 / 3 + 1) / 3)
    assert m["macro_recall"] == pytest.approx((1 / 2 + 1 + 1 / 2) / 3)


def test_unpredicted_and_absent_classes_count_in_macro_average():
    m = M.classification_metrics([0, 0], [0, 0], n_classes=4)
    assert m["top1_accuracy"] == 1.0
    assert m["macro_f1"] == pytest.approx(0.25)    # classes 1-3 get F1 = 0


def test_confusion_and_per_class():
    cm = M.confusion([0, 1, 1], [0, 0, 1], n_classes=3)
    np.testing.assert_array_equal(cm, [[1, 0, 0], [1, 1, 0], [0, 0, 0]])
    rows = M.per_class_metrics([0, 1, 1], [0, 0, 1], ["a", "b", "c"])
    assert [r["support"] for r in rows] == [1, 2, 0]


def test_result_files(tmp_path):
    M.save_json({"a": np.float32(1.5), "b": np.arange(3), "c": {"d": np.int64(2)}}, tmp_path / "r.json")
    assert json.loads((tmp_path / "r.json").read_text()) == {"a": 1.5, "b": [0, 1, 2], "c": {"d": 2}}
    M.save_rows_csv([{"x": 1, "y": 0.5}], tmp_path / "r.csv")
    assert list(csv.DictReader(open(tmp_path / "r.csv"))) == [{"x": "1", "y": "0.5"}]
    M.save_matrix_csv(np.eye(2, dtype=int), tmp_path / "cm.csv", ["a", "b"])
    assert (tmp_path / "cm.csv").read_text().splitlines()[1] == "a,1,0"


def test_figures_are_written(tmp_path):
    rng = np.random.default_rng(0)
    P.plot_training_curve([3, 2, 1.5], path=tmp_path / "curve.png")
    rows = [{"learning_rate": lr, "reg_strength": r, "positive_weight": w, "val_macro_f1": rng.random()}
            for lr in (0.001, 0.01) for r in (1e-4, 1e-3) for w in ("None", "balanced")]
    P.plot_validation_grid(rows, path=tmp_path / "grid.png")
    P.plot_confusion_matrix(rng.integers(0, 5, size=(12, 12)), path=tmp_path / "cm.png")
    per_class = [{"class_id": i, "class_name": f"c{i}", "f1": rng.random(), "support": 20} for i in range(40)]
    P.plot_per_class_f1(per_class, path=tmp_path / "f1.png")
    P.plot_hog_demo(rng.integers(0, 255, (40, 60, 3), dtype=np.uint8), rng.random((16, 16)), rng.random((16, 16)),
                    rng.random((2, 2, 9)), cell_size=8, path=tmp_path / "hog.png")
    P.plot_ablation([{"name": "HOG", "val_macro_f1": 0.12, "kept": True},
                     {"name": "+ sqrt", "val_macro_f1": 0.11, "kept": False}], path=tmp_path / "abl.png")
    P.plot_color_demo(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8), rng.random(320), 16, 4,
                      path=tmp_path / "col.png")
    for name in ("curve", "grid", "cm", "f1", "hog", "abl", "col"):
        assert (tmp_path / f"{name}.png").stat().st_size > 1000
