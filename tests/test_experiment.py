"""Tests for Milestone 2b: colour caching, recipes, design matrices and grid search
(src/classical/features.py, src/classical/experiment.py)."""

import numpy as np
import pytest
from PIL import Image

from src.classical import experiment as E
from src.classical import features as F
from src.classical.color import ColorConfig
from src.classical.hog import HOGConfig
from src.classical.linear_svm import SVMConfig
from src.classical.preprocessing import PreprocessConfig
from src.evaluation.metrics import classification_metrics

HOG, COL = HOGConfig(), ColorConfig()


def _images(n, seed=0):
    rng = np.random.default_rng(seed)
    imgs = [Image.fromarray(rng.integers(0, 256, size=(60, 90, 3), dtype=np.uint8)) for _ in range(n)]
    labels = [i % 3 for i in range(n)]
    return [f"image_{i:05d}" for i in range(1, n + 1)], labels, (lambda i: (imgs[i], labels[i]))


# ------------------------------------------------------------ generic cache
def test_color_features_are_cached_separately_from_hog(tmp_path):
    ids, labels, load = _images(4)
    spec = F.color_spec(PreprocessConfig(), COL)
    fs, status = F.get_or_extract(spec, "train", ids, labels, load, tmp_path)
    assert not fs.from_cache and fs.features.shape == (4, 320)
    np.testing.assert_allclose(fs.features.reshape(4, 5, 64).sum(axis=2), 1.0, atol=1e-5)
    fs2, status2 = F.get_or_extract(spec, "train", ids, labels, lambda i: 1 / 0, tmp_path)
    assert fs2.from_cache and "color_train_" in status2
    F.get_or_extract_features("train", ids, labels, load, tmp_path)
    assert {p.name.split("_")[0] for p in tmp_path.glob("*.npz")} == {"hog", "color"}


def test_resize_mode_changes_the_cache_key(tmp_path):
    a = F.spec_cache_path(tmp_path, "val", F.hog_spec(PreprocessConfig(), HOG))
    b = F.spec_cache_path(tmp_path, "val", F.hog_spec(PreprocessConfig(resize_mode="center_crop"), HOG))
    assert a != b and a.name == "hog_val_b5525eb6a0c2.npz"


def test_spec_rejects_mismatched_config():
    with pytest.raises(ValueError):
        F.FeatureSpec("hog", PreprocessConfig(), COL)


# ------------------------------------------------------------ design matrices
def _fake_features(n_train=6, n_val=4, seed=0):
    rng = np.random.default_rng(seed)
    hog = {"train": rng.random((n_train, 8100)).astype(np.float32), "val": rng.random((n_val, 8100)).astype(np.float32)}
    col = {"train": rng.random((n_train, 320)).astype(np.float32), "val": rng.random((n_val, 320)).astype(np.float32)}
    labels = {"train": np.arange(n_train) % 3, "val": np.arange(n_val) % 3}
    return hog, col, labels


def test_baseline_design_is_plain_standardised_hog():
    hog, col, labels = _fake_features()
    d = E.build_design(E.Recipe("base"), hog, labels, col)
    assert d.X["train"].shape == (6, 8100) and d.X["val"].shape == (4, 8100)
    np.testing.assert_allclose(d.scaler.mean_, hog["train"].mean(axis=0), rtol=1e-6)


def test_sqrt_and_color_and_weight():
    hog, col, labels = _fake_features()
    r = E.Recipe("all", sqrt=True, color=True, color_weight=3.0)
    d = E.build_design(r, hog, labels, col)
    assert d.n_features == 8420
    raw_train = np.sqrt(np.concatenate([hog["train"], col["train"]], axis=1))
    np.testing.assert_allclose(d.scaler.mean_, raw_train.mean(axis=0), rtol=1e-5)
    # colour columns carry 3x the standardised spread of HOG columns
    np.testing.assert_allclose(d.X["train"][:, 8100:].std(axis=0), 3.0, rtol=1e-6)
    np.testing.assert_allclose(d.X["train"][:, :8100].std(axis=0), 1.0, rtol=1e-6)


def test_flip_augments_train_only_and_scaler_uses_train_only():
    hog, col, labels = _fake_features()
    r = E.Recipe("flip", color=True, flip=True)
    d = E.build_design(r, hog, labels, col)
    assert d.X["train"].shape[0] == 12 and d.X["val"].shape[0] == 4
    np.testing.assert_array_equal(d.y["train"], np.concatenate([labels["train"], labels["train"]]))
    assert d.scaler.n_fit_samples_ == 12
    raw = np.concatenate([hog["train"], col["train"]], axis=1)
    perm = E.flip_index(r, HOG, COL, 128)
    np.testing.assert_allclose(d.scaler.mean_, np.concatenate([raw, raw[:, perm]]).mean(axis=0), rtol=1e-5)


def test_transform_split_matches_design_for_val():
    hog, col, labels = _fake_features()
    r = E.Recipe("x", sqrt=True, color=True, color_weight=2.0, flip=True)
    d = E.build_design(r, hog, labels, col)
    np.testing.assert_allclose(E.transform_split(d, r, hog["val"], col["val"]), d.X["val"], rtol=1e-6)


def test_color_recipe_without_color_features_fails():
    hog, _, labels = _fake_features()
    with pytest.raises(ValueError):
        E.build_design(E.Recipe("c", color=True), hog, labels, None)


def test_flip_on_real_images_equals_features_of_mirrored_images():
    """End to end: permuting cached features == extracting features from mirrored preprocessed images."""
    from src.classical.color import color_histogram
    from src.classical.hog import hog_features
    from src.classical.preprocessing import preprocess_rgb, rgb_to_gray

    ids, labels, load = _images(2, seed=3)
    r = E.Recipe("f", color=True, flip=True)
    perm = E.flip_index(r, HOG, COL, 128)
    for i in range(2):
        rgb = preprocess_rgb(load(i)[0])
        f = np.concatenate([hog_features(rgb_to_gray(rgb)), color_histogram(rgb)])
        mirrored = rgb[:, ::-1]
        f_m = np.concatenate([hog_features(rgb_to_gray(mirrored)), color_histogram(mirrored)])
        np.testing.assert_allclose(f_m, f[perm], atol=1e-9)


# ------------------------------------------------------------ grid search
def test_grid_search_selects_by_validation_macro_f1():
    rng = np.random.default_rng(0)
    centres = rng.normal(0, 3, size=(3, 10))
    Xtr = np.concatenate([c + 0.5 * rng.normal(size=(20, 10)) for c in centres])
    Xva = np.concatenate([c + 0.5 * rng.normal(size=(10, 10)) for c in centres])
    ytr, yva = np.repeat(np.arange(3), 20), np.repeat(np.arange(3), 10)
    configs = [SVMConfig(learning_rate=1e-6, n_epochs=1), SVMConfig(learning_rate=0.05, n_epochs=20),
               SVMConfig(learning_rate=0.05, n_epochs=20, positive_weight="balanced")]
    rows, best, model = E.svm_grid_search(Xtr, ytr, Xva, yva, configs, 3, classification_metrics)
    assert len(rows) == 3
    keys = [(r["val_macro_f1"], r["val_top1_accuracy"]) for r in rows]
    assert best is rows[keys.index(max(keys))]          # best key; ties go to the earlier configuration
    # the returned model is the selected one: it reproduces the selected row's validation score
    assert classification_metrics(yva, model.predict(Xva), 3)["macro_f1"] == pytest.approx(best["val_macro_f1"])
