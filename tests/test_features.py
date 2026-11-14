"""Tests for split-level feature extraction, caching and split separation
(src/classical/features.py), plus an end-to-end run of the classical pipeline
on synthetic images. No dataset or network access is needed.
"""

import numpy as np
import pytest
from PIL import Image

from src.classical import features as F
from src.classical.hog import HOGConfig
from src.classical.linear_svm import LinearSVMOvR, SVMConfig
from src.classical.preprocessing import FeatureStandardizer, PreprocessConfig
from src.data import flowers102 as f102
from src.evaluation import metrics as M

PRE, HOG = PreprocessConfig(), HOGConfig()


def stripes(angle_deg, seed, size=(90, 120)):
    """RGB image of stripes at a given angle, plus noise (one 'class' per angle)."""
    rng = np.random.default_rng(seed)
    h, w = size
    y, x = np.mgrid[0:h, 0:w]
    t = np.radians(angle_deg)
    phase = rng.uniform(0, 2 * np.pi)
    img = 0.5 + 0.4 * np.sin((x * np.cos(t) + y * np.sin(t)) / 4.0 + phase) + 0.05 * rng.normal(size=(h, w))
    rgb = (np.clip(img, 0, 1)[..., None] * [255, 200, 150]).astype(np.uint8)
    return Image.fromarray(rgb)


def fake_split(name, start, n, n_classes=3):
    """A split-like object: image IDs, labels and a loader returning (PIL image, label)."""
    ids = [f"image_{i:05d}" for i in range(start, start + n)]
    labels = [i % n_classes for i in range(n)]
    angles = [0, 60, 120]

    def load(i):
        return stripes(angles[labels[i]], seed=start + i), labels[i]

    return ids, labels, load


# ------------------------------------------------------------------ caching
def test_extract_then_reload_from_cache(tmp_path, monkeypatch):
    ids, labels, load = fake_split("train", 1, 6)
    fs, status = F.get_or_extract_features("train", ids, labels, load, tmp_path, PRE, HOG)
    assert not fs.from_cache and "extracted" in status
    assert fs.features.shape == (6, 8100) and fs.features.dtype == np.float32
    assert np.all(np.isfinite(fs.features))
    assert fs.meta["extraction_seconds"] > 0

    # second call must not touch the images at all
    def boom(i):
        raise AssertionError("loader called although a valid cache exists")

    fs2, status2 = F.get_or_extract_features("train", ids, labels, boom, tmp_path, PRE, HOG)
    assert fs2.from_cache and "loaded from cache" in status2
    np.testing.assert_array_equal(fs2.features, fs.features)
    assert fs2.image_ids.tolist() == ids and fs2.labels.tolist() == labels
    assert fs2.meta["extraction_seconds"] == pytest.approx(fs.meta["extraction_seconds"])


def test_cache_is_rebuilt_when_config_changes(tmp_path):
    ids, labels, load = fake_split("train", 1, 4)
    F.get_or_extract_features("train", ids, labels, load, tmp_path, PRE, HOG)
    other = HOGConfig(n_bins=12)
    fs, status = F.get_or_extract_features("train", ids, labels, load, tmp_path, PRE, other)
    assert not fs.from_cache
    assert fs.features.shape == (4, other.feature_length((128, 128)))
    assert len(list(tmp_path.glob("hog_train_*.npz"))) == 2   # different configs -> different files


def test_cache_rejected_when_split_contents_change(tmp_path):
    ids, labels, load = fake_split("val", 1, 4)
    path = F.cache_path(tmp_path, "val", PRE, HOG)
    F.get_or_extract_features("val", ids, labels, load, tmp_path, PRE, HOG)
    changed = labels.copy()
    changed[0] = 2
    fs, reason = F.load_feature_cache(path, F.expected_meta("val", ids, changed, PRE, HOG))
    assert fs is None and "samples_sha256" in reason
    fs, reason = F.load_feature_cache(path, F.expected_meta("val", ids[:3], labels[:3], PRE, HOG))
    assert fs is None


def test_cache_of_one_split_never_serves_another(tmp_path):
    ids, labels, load = fake_split("train", 1, 4)
    F.get_or_extract_features("train", ids, labels, load, tmp_path, PRE, HOG)
    train_file = F.cache_path(tmp_path, "train", PRE, HOG)
    # even a copied/renamed file is rejected because the split name is in the metadata
    val_file = F.cache_path(tmp_path, "val", PRE, HOG)
    val_file.write_bytes(train_file.read_bytes())
    fs, reason = F.load_feature_cache(val_file, F.expected_meta("val", ids, labels, PRE, HOG))
    assert fs is None and "split" in reason


def test_corrupted_cache_is_reextracted(tmp_path):
    ids, labels, load = fake_split("test", 1, 3)
    path = F.cache_path(tmp_path, "test", PRE, HOG)
    path.write_bytes(b"not a real npz file")
    fs, status = F.get_or_extract_features("test", ids, labels, load, tmp_path, PRE, HOG)
    assert not fs.from_cache and "unreadable" in status
    assert F.load_feature_cache(path, F.expected_meta("test", ids, labels, PRE, HOG))[0] is not None


def test_force_reextracts(tmp_path):
    ids, labels, load = fake_split("train", 1, 3)
    F.get_or_extract_features("train", ids, labels, load, tmp_path, PRE, HOG)
    fs, status = F.get_or_extract_features("train", ids, labels, load, tmp_path, PRE, HOG, force=True)
    assert not fs.from_cache and "force" in status


def test_loader_label_mismatch_is_detected(tmp_path):
    ids, labels, load = fake_split("train", 1, 3)
    with pytest.raises(RuntimeError):
        F.get_or_extract_features("train", ids, [l + 1 for l in labels], load, tmp_path, PRE, HOG)


def test_image_ids_from_files():
    assert F.image_ids_from_files(["image_00001.jpg", "/a/b/image_08189.jpg"]) == ["image_00001", "image_08189"]


# --------------------------------------------------------- split separation
def _fs(split, ids):
    return F.FeatureSet(split, np.zeros((len(ids), 2), np.float32), np.array(ids), np.zeros(len(ids), int), {}, False)


def test_disjoint_splits_pass():
    F.assert_disjoint_splits({"train": _fs("train", ["a", "b"]), "val": _fs("val", ["c"]), "test": _fs("test", ["d"])},
                             expected_sizes={"train": 2, "val": 1, "test": 1})


def test_overlapping_splits_fail():
    with pytest.raises(AssertionError, match="both train and test"):
        F.assert_disjoint_splits({"train": _fs("train", ["a", "b"]), "test": _fs("test", ["b"])})


def test_duplicates_and_wrong_sizes_fail():
    with pytest.raises(AssertionError, match="duplicate"):
        F.assert_disjoint_splits({"train": _fs("train", ["a", "a"])})
    with pytest.raises(AssertionError, match="expected"):
        F.assert_disjoint_splits({"train": _fs("train", ["a"])}, expected_sizes={"train": 1020})


def test_official_split_sizes_are_used_unchanged():
    assert f102.EXPECTED_SPLIT_SIZES == {"train": 1020, "val": 1020, "test": 6149}


# ------------------------------------------------------ end-to-end pipeline
def test_pipeline_end_to_end_on_synthetic_images(tmp_path):
    """HOG -> standardise on train only -> SVM -> metrics, with disjoint splits."""
    splits = {"train": fake_split("train", 1, 30), "val": fake_split("val", 1001, 15),
              "test": fake_split("test", 2001, 15)}
    sets = {s: F.get_or_extract_features(s, *splits[s], tmp_path, PRE, HOG)[0] for s in splits}
    F.assert_disjoint_splits(sets)

    scaler = FeatureStandardizer().fit(sets["train"].features)
    X = {s: scaler.transform(fs.features) for s, fs in sets.items()}
    model = LinearSVMOvR(SVMConfig(learning_rate=0.001, reg_strength=1e-3, n_epochs=20, batch_size=8))
    model.fit(X["train"], sets["train"].labels)

    for s in ("val", "test"):
        scores = model.decision_function(X[s])
        assert scores.shape == (15, 3)
        m = M.classification_metrics(sets[s].labels, model.predict(X[s]), n_classes=3)
        assert m["top1_accuracy"] >= 0.9, (s, m)   # stripe orientation is what HOG measures
