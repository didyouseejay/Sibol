"""Tests for src/data/flowers102.py that do NOT require the real dataset.

A tiny fake archive is built in a temporary folder to test extraction.
Network access is blocked in the tests that check DOWNLOAD=False behaviour.

Run from the project root with:
    python -m pytest tests/
"""

import io
import socket
import tarfile
from pathlib import Path

import pytest

from src.data import flowers102 as f102


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def no_network(monkeypatch):
    """Make any attempt to open a network connection fail loudly."""

    def blocked(*args, **kwargs):
        raise AssertionError("Network access attempted during a DOWNLOAD=False test")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(f102, "download_source_files", blocked)


def _add_file(tar, name, data=b"fake-jpeg"):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


@pytest.fixture
def fake_source(tmp_path):
    """Source folder with a 3-image fake archive plus two fake .mat files.

    The archive also contains entries that must NOT be extracted:
    a path-traversal attempt, a nested file and a non-jpg file.
    """
    src = tmp_path / "source"
    src.mkdir()
    with tarfile.open(src / f102.ARCHIVE_FILE, "w:gz") as tar:
        for i in range(1, 4):
            _add_file(tar, f"jpg/image_{i:05d}.jpg")
        _add_file(tar, "../evil.jpg")
        _add_file(tar, "jpg/sub/nested.jpg")
        _add_file(tar, "jpg/readme.txt")
    (src / f102.LABELS_FILE).write_bytes(b"labels")
    (src / f102.SETID_FILE).write_bytes(b"setid")
    return src


# ---------------------------------------------------------------------------
# Constants and small helpers
# ---------------------------------------------------------------------------
def test_expected_split_sizes_add_up():
    assert sum(f102.EXPECTED_SPLIT_SIZES.values()) == f102.EXPECTED_TOTAL_IMAGES


def test_train_and_val_have_ten_per_class():
    for split, per_class in f102.EXPECTED_IMAGES_PER_CLASS.items():
        assert f102.EXPECTED_SPLIT_SIZES[split] == per_class * f102.NUM_CLASSES


def test_three_source_files_required():
    assert set(f102.SOURCE_FILES) == {"102flowers.tgz", "imagelabels.mat", "setid.mat"}


def test_class_counts_includes_empty_classes():
    assert f102.class_counts([0, 0, 2], num_classes=4) == [2, 0, 1, 0]


def test_get_labels_uses_private_attribute():
    class FakeDataset:
        _labels = [3, 1, 4]

    assert f102.get_labels(FakeDataset()) == [3, 1, 4]


def test_get_labels_falls_back_to_iteration():
    assert f102.get_labels([(None, 5), (None, 7)]) == [5, 7]


def test_get_image_files_returns_basenames():
    class FakeDataset:
        _image_files = [Path("/x/jpg/image_00001.jpg"), Path("/x/jpg/image_00002.jpg")]

    assert f102.get_image_files(FakeDataset()) == ["image_00001.jpg", "image_00002.jpg"]


def test_default_data_root_colab_vs_local(tmp_path):
    assert f102.default_data_root(True) == Path("/content/drive/MyDrive/sibol/data")
    assert f102.default_data_root(False, tmp_path) == tmp_path / "data"


# ---------------------------------------------------------------------------
# DOWNLOAD=False must never download and must explain what is missing
# ---------------------------------------------------------------------------
def test_missing_files_raise_clear_error_without_network(tmp_path, no_network):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError) as exc:
        f102.prepare_dataset(empty, tmp_path / "work", download=False)
    message = str(exc.value)
    for name in f102.SOURCE_FILES:
        assert name in message
    assert str(empty) in message
    assert "DOWNLOAD = False" in message
    assert not (tmp_path / "work").exists()  # nothing was created


def test_partially_missing_files_are_listed(fake_source, tmp_path, no_network):
    (fake_source / f102.SETID_FILE).unlink()
    with pytest.raises(FileNotFoundError) as exc:
        f102.prepare_dataset(fake_source, tmp_path / "work", check_md5=False)
    assert "setid.mat" in str(exc.value)
    assert "102flowers.tgz" not in str(exc.value).split("Nothing was downloaded")[0]


def test_download_true_is_the_only_path_to_downloading(fake_source, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(f102, "download_source_files", lambda d: calls.append(d) or [])
    f102.prepare_dataset(fake_source, tmp_path / "w1", download=False, check_md5=False, expected_images=3)
    assert calls == []
    f102.prepare_dataset(fake_source, tmp_path / "w2", download=True, check_md5=False, expected_images=3)
    assert calls == [fake_source]


def test_load_split_never_downloads_and_explains(tmp_path, no_network):
    with pytest.raises(FileNotFoundError) as exc:
        f102.load_split(tmp_path, "train")
    assert "prepare_dataset" in str(exc.value)


def test_load_split_rejects_bad_split(tmp_path):
    with pytest.raises(ValueError):
        f102.load_split(tmp_path, "validation")


# ---------------------------------------------------------------------------
# Checksums and extraction
# ---------------------------------------------------------------------------
def test_md5_mismatch_is_reported(fake_source, no_network):
    with pytest.raises(ValueError, match="MD5"):
        f102.verify_source_files(fake_source, check_md5=True)


def test_prepare_extracts_only_safe_jpgs_and_copies_mat_files(fake_source, tmp_path, no_network):
    work = tmp_path / "work"
    f102.prepare_dataset(fake_source, work, check_md5=False, expected_images=3)

    base = work / "flowers-102"
    assert sorted(p.name for p in (base / "jpg").iterdir()) == [
        "image_00001.jpg",
        "image_00002.jpg",
        "image_00003.jpg",
    ]
    assert not (tmp_path / "evil.jpg").exists()
    assert not (work / "evil.jpg").exists()
    assert not (base / "jpg.partial").exists()
    assert (base / "imagelabels.mat").read_bytes() == b"labels"
    assert (base / "setid.mat").read_bytes() == b"setid"
    assert f102.is_prepared(work, expected_images=3)


def test_prepare_reuses_complete_extraction(fake_source, tmp_path, monkeypatch, no_network):
    work = tmp_path / "work"
    f102.prepare_dataset(fake_source, work, check_md5=False, expected_images=3)

    def fail(*args, **kwargs):
        raise AssertionError("should not re-extract a complete copy")

    monkeypatch.setattr(f102, "extract_images", fail)
    f102.prepare_dataset(fake_source, work, check_md5=False, expected_images=3)


def test_prepare_reextracts_incomplete_copy(fake_source, tmp_path, no_network):
    work = tmp_path / "work"
    f102.prepare_dataset(fake_source, work, check_md5=False, expected_images=3)
    (work / "flowers-102" / "jpg" / "image_00002.jpg").unlink()
    assert not f102.is_prepared(work, expected_images=3)
    f102.prepare_dataset(fake_source, work, check_md5=False, expected_images=3)
    assert f102.is_prepared(work, expected_images=3)


def test_wrong_image_count_raises(fake_source, tmp_path, no_network):
    with pytest.raises(RuntimeError, match="Extracted 3 images"):
        f102.prepare_dataset(fake_source, tmp_path / "work", check_md5=False, expected_images=5)


# ---------------------------------------------------------------------------
# The 10 integrity checks
# ---------------------------------------------------------------------------
def _official_like_splits():
    """Synthetic labels/files with the same shape as the official splits."""
    labels = {
        "train": [c for c in range(102) for _ in range(10)],
        "val": [c for c in range(102) for _ in range(10)],
        "test": [c for c in range(102) for _ in range(60)] + [0] * 29,  # 6149 total
    }
    files, k = {}, 0
    for s in f102.SPLITS:
        files[s] = [f"image_{k + i:05d}.jpg" for i in range(len(labels[s]))]
        k += len(labels[s])
    return labels, files


def test_integrity_checks_all_pass_on_official_shape():
    labels, files = _official_like_splits()
    results = f102.integrity_checks(labels, files)
    assert len(results) == 10
    assert all(passed for _, passed, _ in results), results


def test_integrity_checks_detect_overlap():
    labels, files = _official_like_splits()
    files["test"][0] = files["train"][0]
    results = dict((name, ok) for name, ok, _ in f102.integrity_checks(labels, files))
    assert results["splits are disjoint"] is False


def test_integrity_checks_fail_when_file_names_unavailable():
    labels, _ = _official_like_splits()
    results = f102.integrity_checks(labels, {"train": [], "val": [], "test": []})
    assert len(results) == 10
    assert dict((n, ok) for n, ok, _ in results)["splits are disjoint"] is False


def test_integrity_checks_detect_unbalanced_train():
    labels, files = _official_like_splits()
    labels["train"][0] = 1  # class 0 now has 9, class 1 has 11
    results = dict((name, ok) for name, ok, _ in f102.integrity_checks(labels, files))
    assert results["train images per class"] is False
    assert results["train size"] is True
