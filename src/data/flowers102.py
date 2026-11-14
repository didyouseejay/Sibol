"""Helpers for working with the Oxford Flowers-102 dataset.

Every notebook and script in SIBOL prepares and loads the dataset through
this module, so that all three approaches use the same files and the same
*official* train / validation / test splits.

Data flow
---------
1. **Source files** (persistent, e.g. on Google Drive)::

       <source_dir>/102flowers.tgz     images (~345 MB archive)
       <source_dir>/imagelabels.mat    class label of every image
       <source_dir>/setid.mat          official train / val / test split

2. :func:`prepare_dataset` verifies the source files, extracts the images to a
   fast working directory (e.g. Colab's temporary ``/content`` disk) and copies
   the two ``.mat`` files next to them::

       <work_root>/flowers-102/jpg/image_00001.jpg ... image_08189.jpg
       <work_root>/flowers-102/imagelabels.mat
       <work_root>/flowers-102/setid.mat

3. :func:`load_split` opens one split from ``<work_root>`` with torchvision.

Network policy
--------------
Nothing in this module accesses the internet unless the caller passes
``download=True`` to :func:`prepare_dataset`. :func:`load_split` never
downloads. Missing files raise :class:`FileNotFoundError` explaining which
files are missing and where to put them.

Reference:
    M.-E. Nilsback and A. Zisserman. "Automated Flower Classification over a
    Large Number of Classes." ICVGIP 2008.
    https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Optional, Sequence

# ---------------------------------------------------------------------------
# Facts about the official dataset (from the dataset authors' documentation).
# These are *expected* values used to verify the data, not results.
# ---------------------------------------------------------------------------
NUM_CLASSES = 102
SPLITS = ("train", "val", "test")

# Official split sizes: 10 images per class for train and val, rest for test.
EXPECTED_SPLIT_SIZES = {"train": 1020, "val": 1020, "test": 6149}
EXPECTED_TOTAL_IMAGES = 8189
EXPECTED_IMAGES_PER_CLASS = {"train": 10, "val": 10}

# Folder name torchvision expects inside the data root.
TORCHVISION_SUBDIR = "flowers-102"

# The three source files and their official MD5 checksums
# (same values torchvision uses in torchvision.datasets.Flowers102).
ARCHIVE_FILE = "102flowers.tgz"
LABELS_FILE = "imagelabels.mat"
SETID_FILE = "setid.mat"
SOURCE_FILES = {
    ARCHIVE_FILE: "52808999861908f626f3c1f4e79d11fa",
    LABELS_FILE: "e0620be6f572b9609742df49c70aed4d",
    SETID_FILE: "a5357ecc9cb78c4bef273ce3793fc85c",
}
OFFICIAL_URL_PREFIX = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def default_data_root(in_colab: bool, project_root: Optional[Path] = None) -> Path:
    """Return the default folder holding the three source files.

    - Colab: ``/content/drive/MyDrive/sibol/data`` (persistent Google Drive).
    - Local: ``<project_root>/data`` (git-ignored).
    """
    if in_colab:
        return Path("/content/drive/MyDrive/sibol/data")
    if project_root is None:
        project_root = Path.cwd()
    return Path(project_root) / "data"


def dataset_dir(work_root: Path) -> Path:
    """Folder torchvision reads from: ``<work_root>/flowers-102``."""
    return Path(work_root) / TORCHVISION_SUBDIR


# ---------------------------------------------------------------------------
# Source-file checks
# ---------------------------------------------------------------------------
def md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the MD5 checksum of a file, reading it in 1 MB chunks."""
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def missing_source_files(source_dir: Path) -> list[str]:
    """Return the names of required source files that are not in ``source_dir``."""
    source_dir = Path(source_dir)
    return [name for name in SOURCE_FILES if not (source_dir / name).is_file()]


def missing_files_message(source_dir: Path, missing: Sequence[str]) -> str:
    """Build a clear, actionable error message for missing source files."""
    lines = [
        f"Missing Flowers-102 source file(s) in {Path(source_dir)}:",
        *[f"  - {name}" for name in missing],
        "",
        "Nothing was downloaded because DOWNLOAD = False.",
        "To fix this, do ONE of the following:",
        f"  1. Place the missing file(s) in: {Path(source_dir)}",
        f"     (official downloads: {OFFICIAL_URL_PREFIX})",
        "  2. Point the source-folder setting to where the files already are.",
        "  3. Set DOWNLOAD = True to download them from the official site.",
    ]
    return "\n".join(lines)


def verify_source_files(source_dir: Path, check_md5: bool = True) -> dict[str, dict]:
    """Check that all three source files exist (and optionally match MD5).

    Never accesses the network.

    Returns:
        ``{file_name: {"path": Path, "size": int, "md5_ok": bool | None}}``
        (``md5_ok`` is None when ``check_md5`` is False).

    Raises:
        FileNotFoundError: if any source file is missing.
        ValueError: if a file's MD5 does not match the official checksum.
    """
    source_dir = Path(source_dir)
    missing = missing_source_files(source_dir)
    if missing:
        raise FileNotFoundError(missing_files_message(source_dir, missing))

    report = {}
    for name, expected_md5 in SOURCE_FILES.items():
        path = source_dir / name
        md5_ok = None
        if check_md5:
            actual = md5sum(path)
            md5_ok = actual == expected_md5
            if not md5_ok:
                raise ValueError(
                    f"{path} is corrupted or not the official file: "
                    f"MD5 {actual}, expected {expected_md5}. Replace it with a fresh copy."
                )
        report[name] = {"path": path, "size": path.stat().st_size, "md5_ok": md5_ok}
    return report


def download_source_files(source_dir: Path) -> list[str]:
    """Download missing source files from the official site into ``source_dir``.

    Only called by :func:`prepare_dataset` when ``download=True``.
    Returns the names of the files that were downloaded.
    """
    from torchvision.datasets.utils import download_url  # verifies MD5 too

    source_dir = Path(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    missing = missing_source_files(source_dir)
    for name in missing:
        download_url(OFFICIAL_URL_PREFIX + name, str(source_dir), filename=name, md5=SOURCE_FILES[name])
    return missing


# ---------------------------------------------------------------------------
# Preparing the working copy (extraction)
# ---------------------------------------------------------------------------
def count_images(work_root: Path) -> int:
    """Number of ``.jpg`` files in ``<work_root>/flowers-102/jpg``."""
    jpg_dir = dataset_dir(work_root) / "jpg"
    if not jpg_dir.is_dir():
        return 0
    return sum(1 for p in jpg_dir.iterdir() if p.suffix == ".jpg")


def is_prepared(work_root: Path, expected_images: int = EXPECTED_TOTAL_IMAGES) -> bool:
    """True if the working copy is complete: all images and both ``.mat`` files."""
    base = dataset_dir(work_root)
    return (
        (base / LABELS_FILE).is_file()
        and (base / SETID_FILE).is_file()
        and count_images(work_root) == expected_images
    )


def extract_images(archive: Path, work_root: Path) -> int:
    """Safely extract the ``jpg/*.jpg`` images from the archive.

    Only regular files directly inside ``jpg/`` are written, so a malformed
    archive cannot write outside the working directory. Images are extracted
    to a temporary folder that is renamed to ``jpg/`` only when extraction
    finishes, so an interrupted run never looks complete.

    Returns the number of images extracted.
    """
    base = dataset_dir(work_root)
    base.mkdir(parents=True, exist_ok=True)
    final_dir = base / "jpg"
    partial_dir = base / "jpg.partial"
    shutil.rmtree(partial_dir, ignore_errors=True)
    partial_dir.mkdir()

    n = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            parts = PurePosixPath(member.name).parts
            if (
                not member.isfile()
                or len(parts) != 2
                or parts[0] != "jpg"
                or not parts[1].endswith(".jpg")
            ):
                continue
            src = tar.extractfile(member)
            if src is None:
                continue
            with src, open(partial_dir / parts[1], "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1

    shutil.rmtree(final_dir, ignore_errors=True)
    partial_dir.rename(final_dir)
    return n


def prepare_dataset(
    source_dir: Path,
    work_root: Path,
    download: bool = False,
    check_md5: bool = True,
    expected_images: int = EXPECTED_TOTAL_IMAGES,
) -> Path:
    """Make the dataset ready for :func:`load_split`.

    Steps:
        1. If ``download`` is True, fetch any *missing* source files from the
           official site into ``source_dir``. If False (default), never access
           the network.
        2. Verify the three source files (existence and, optionally, MD5).
        3. Extract the images into ``<work_root>/flowers-102/jpg`` unless a
           complete copy is already there.
        4. Copy the two ``.mat`` files next to the images.

    Args:
        source_dir: Folder holding the three source files (e.g. on Drive).
        work_root: Fast working folder for the extracted images
            (e.g. ``/content/flowers102`` in Colab).
        download: Allow downloading missing source files. Default False.
        check_md5: Verify official MD5 checksums of the source files.
        expected_images: Number of images a complete extraction contains.

    Returns:
        ``work_root`` (pass it to :func:`load_split`).

    Raises:
        FileNotFoundError: source files missing and ``download`` is False.
        ValueError: a source file fails the MD5 check.
        RuntimeError: extraction produced the wrong number of images.
    """
    source_dir, work_root = Path(source_dir), Path(work_root)

    if download:
        download_source_files(source_dir)

    verify_source_files(source_dir, check_md5=check_md5)

    base = dataset_dir(work_root)
    if count_images(work_root) != expected_images:
        n = extract_images(source_dir / ARCHIVE_FILE, work_root)
        if n != expected_images:
            raise RuntimeError(f"Extracted {n} images, expected {expected_images}.")

    for name in (LABELS_FILE, SETID_FILE):
        shutil.copy2(source_dir / name, base / name)

    return work_root


# ---------------------------------------------------------------------------
# Loading (never downloads)
# ---------------------------------------------------------------------------
def load_split(work_root: Path, split: str, transform=None):
    """Load one official split of Flowers-102 via torchvision. Never downloads.

    Args:
        work_root: Folder prepared by :func:`prepare_dataset` (the parent of
            ``flowers-102/``).
        split: One of ``"train"``, ``"val"``, ``"test"``.
        transform: Optional torchvision transform applied to each image.

    Returns:
        A ``torchvision.datasets.Flowers102`` instance.

    Raises:
        ValueError: unknown split name.
        FileNotFoundError: the working copy is incomplete; run
            :func:`prepare_dataset` first.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    base = dataset_dir(work_root)
    missing = [
        str(p)
        for p in (base / "jpg", base / LABELS_FILE, base / SETID_FILE)
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Flowers-102 working copy is incomplete. Missing:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\nRun prepare_dataset(source_dir, work_root) first."
        )

    # Imported here so the rest of this module works without torchvision.
    from torchvision.datasets import Flowers102

    return Flowers102(root=str(work_root), split=split, transform=transform, download=False)


# ---------------------------------------------------------------------------
# Inspecting a loaded split
# ---------------------------------------------------------------------------
def get_labels(dataset) -> list[int]:
    """Return the integer label (0-101) of every sample, without decoding images.

    torchvision stores labels in the private attribute ``_labels``. Reading
    it is much faster than iterating over the dataset (which opens every
    JPEG). If a future torchvision version renames it, we fall back to
    iterating.
    """
    labels = getattr(dataset, "_labels", None)
    if labels is not None:
        return [int(y) for y in labels]
    return [int(y) for _, y in dataset]


def get_image_files(dataset) -> list[str]:
    """Return the image file names (e.g. ``image_00001.jpg``) in a split.

    Uses torchvision's private ``_image_files`` attribute. Returns an empty
    list if it is not available.
    """
    files = getattr(dataset, "_image_files", None)
    if files is None:
        return []
    return [Path(f).name for f in files]


def class_counts(labels: Iterable[int], num_classes: int = NUM_CLASSES) -> list[int]:
    """Count how many samples belong to each class.

    Returns a list of length ``num_classes`` where index ``i`` is the number
    of samples with label ``i`` (classes with no samples get 0).
    """
    counter = Counter(int(y) for y in labels)
    return [counter.get(i, 0) for i in range(num_classes)]


# ---------------------------------------------------------------------------
# The 10 official-split integrity checks
# ---------------------------------------------------------------------------
def integrity_checks(
    labels: Mapping[str, Sequence[int]],
    files: Mapping[str, Sequence[str]],
) -> list[tuple[str, bool, str]]:
    """Run the 10 integrity checks on the official splits.

    Args:
        labels: ``{"train": [...], "val": [...], "test": [...]}`` labels per split.
        files: Same keys, image file names per split.

    Returns:
        A list of 10 ``(check_name, passed, detail)`` tuples:
        3 split sizes, total size, 3 class counts, 2 images-per-class checks,
        and split disjointness. If file names are unavailable, the
        disjointness check fails rather than being skipped.
    """
    results = []

    for s in SPLITS:
        n, expected = len(labels[s]), EXPECTED_SPLIT_SIZES[s]
        results.append((f"{s} size", n == expected, f"{n} (expected {expected})"))

    total = sum(len(labels[s]) for s in SPLITS)
    results.append(
        ("total images", total == EXPECTED_TOTAL_IMAGES, f"{total} (expected {EXPECTED_TOTAL_IMAGES})")
    )

    for s in SPLITS:
        n_classes = len(set(labels[s]))
        results.append((f"{s} classes", n_classes == NUM_CLASSES, f"{n_classes} (expected {NUM_CLASSES})"))

    for s, per_class in EXPECTED_IMAGES_PER_CLASS.items():
        counts = class_counts(labels[s])
        results.append(
            (
                f"{s} images per class",
                all(c == per_class for c in counts),
                f"min {min(counts)}, max {max(counts)} (expected {per_class})",
            )
        )

    sets = {s: set(files.get(s, ())) for s in SPLITS}
    if all(len(sets[s]) == len(labels[s]) for s in SPLITS):
        overlap = (sets["train"] & sets["val"]) | (sets["train"] & sets["test"]) | (sets["val"] & sets["test"])
        results.append(("splits are disjoint", not overlap, f"{len(overlap)} shared files"))
    else:
        results.append(("splits are disjoint", False, "image file names unavailable or duplicated"))

    return results
