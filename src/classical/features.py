"""Feature extraction for whole splits, with a validated on-disk cache.

Supported feature types (a :class:`FeatureSpec` each):

* ``"hog"``   -- :mod:`src.classical.hog` on the preprocessed grayscale image
* ``"color"`` -- :mod:`src.classical.color` on the preprocessed RGB image

One cache file per feature type, configuration and split::

    <cache_dir>/<kind>_<split>_<config-hash>.npz
        features   float32 (N, D)
        image_ids  str     (N,)   e.g. "image_00001"
        labels     int64   (N,)   0-101
        meta       JSON string (see below)

The metadata records the preprocessing and feature configuration, the split
name, the number of samples, a SHA-256 fingerprint of the (image_id, label)
list, the cache-format and implementation versions, and how long the original
extraction took. A cache is **reused only if all of these match** what the
caller expects; otherwise it is rebuilt (the reason is reported).

The HOG cache key is unchanged since Milestone 2, so existing HOG caches
(e.g. ``hog_<split>_b5525eb6a0c2.npz``) stay valid.

Images are read through the project's official loader
(:func:`src.data.flowers102.load_split`), so the split membership and the
image/label order are exactly torchvision's reading of ``setid.mat``.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

import numpy as np

from .color import ColorConfig, color_histogram
from .hog import HOGConfig, hog_features
from .preprocessing import PreprocessConfig, preprocess_image, preprocess_rgb

CACHE_FORMAT_VERSION = 1

# load(i) -> (image, label) for sample i of a split, e.g. a torchvision dataset's __getitem__.
SampleLoader = Callable[[int], tuple]


@dataclass
class FeatureSet:
    """Features of one split, aligned with their image IDs and labels."""

    split: str
    features: np.ndarray
    image_ids: np.ndarray
    labels: np.ndarray
    meta: dict
    from_cache: bool

    def __len__(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class FeatureSpec:
    """What to compute for every image: a feature type plus its configuration."""

    kind: str                                   # "hog" or "color"
    preprocess: PreprocessConfig
    config: Union[HOGConfig, ColorConfig]

    def __post_init__(self) -> None:
        expected = {"hog": HOGConfig, "color": ColorConfig}
        if self.kind not in expected or not isinstance(self.config, expected[self.kind]):
            raise ValueError(f"kind {self.kind!r} does not match config {type(self.config).__name__}")

    def payload(self) -> dict:
        """Everything that changes the numbers (preprocessing + feature config)."""
        return {"preprocess": self.preprocess.to_dict(), self.kind: self.config.to_dict()}

    @property
    def feature_length(self) -> int:
        if self.kind == "hog":
            size = self.preprocess.image_size
            return self.config.feature_length((size, size))
        return self.config.feature_length

    def compute(self, image) -> np.ndarray:
        if self.kind == "hog":
            return hog_features(preprocess_image(image, self.preprocess), self.config)
        return color_histogram(preprocess_rgb(image, self.preprocess), self.config)


def hog_spec(preprocess: PreprocessConfig = PreprocessConfig(), hog: HOGConfig = HOGConfig()) -> FeatureSpec:
    return FeatureSpec("hog", preprocess, hog)


def color_spec(preprocess: PreprocessConfig = PreprocessConfig(), color: ColorConfig = ColorConfig()) -> FeatureSpec:
    return FeatureSpec("color", preprocess, color)


# ---------------------------------------------------------------------------
# Identity of a split and of a configuration
# ---------------------------------------------------------------------------
def image_ids_from_files(files: Sequence[str]) -> list[str]:
    """``image_00001.jpg`` -> ``image_00001``."""
    return [Path(f).stem for f in files]


def samples_fingerprint(image_ids: Sequence[str], labels: Sequence[int]) -> str:
    """SHA-256 of the ordered (image_id, label) pairs."""
    h = hashlib.sha256()
    for image_id, label in zip(image_ids, labels):
        h.update(f"{image_id}:{int(label)}\n".encode())
    return h.hexdigest()


def spec_hash(spec: FeatureSpec) -> str:
    """Short hash of everything that changes the numbers in a feature vector."""
    payload = {**spec.payload(), "cache_format": CACHE_FORMAT_VERSION}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def spec_cache_path(cache_dir: Path, split: str, spec: FeatureSpec) -> Path:
    return Path(cache_dir) / f"{spec.kind}_{split}_{spec_hash(spec)}.npz"


def spec_expected_meta(split: str, image_ids: Sequence[str], labels: Sequence[int], spec: FeatureSpec) -> dict:
    """The metadata a valid cache for this split and feature spec must carry."""
    return {
        "cache_format": CACHE_FORMAT_VERSION,
        "split": split,
        "n_samples": len(image_ids),
        "samples_sha256": samples_fingerprint(image_ids, labels),
        **spec.payload(),
        "feature_length": spec.feature_length,
    }


# HOG-specific names kept for Milestone 2 code (notebook 02) -------------------
def config_hash(preprocess: PreprocessConfig, hog: HOGConfig) -> str:
    return spec_hash(hog_spec(preprocess, hog))


def cache_path(cache_dir: Path, split: str, preprocess: PreprocessConfig, hog: HOGConfig) -> Path:
    return spec_cache_path(cache_dir, split, hog_spec(preprocess, hog))


def expected_meta(split: str, image_ids: Sequence[str], labels: Sequence[int],
                  preprocess: PreprocessConfig, hog: HOGConfig) -> dict:
    return spec_expected_meta(split, image_ids, labels, hog_spec(preprocess, hog))


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def extract_spec(load: SampleLoader, labels: Sequence[int], spec: FeatureSpec,
                 progress: Optional[Callable] = None) -> tuple[np.ndarray, float]:
    """Compute one feature type for every sample of a split.

    Args:
        load: ``load(i) -> (image, label)``. The label is checked against ``labels[i]``.
        labels: Expected label of every sample (defines N and the order).
        progress: Optional wrapper such as ``tqdm`` applied to the index range.

    Returns:
        ``(features float32 (N, D), elapsed_seconds)``. The time covers image
        decoding, preprocessing and feature computation.
    """
    n = len(labels)
    out = np.empty((n, spec.feature_length), dtype=np.float32)
    indices = range(n) if progress is None else progress(range(n))
    t0 = time.perf_counter()
    for i in indices:
        image, label = load(i)
        if int(label) != int(labels[i]):
            raise RuntimeError(f"sample {i}: loader label {label} != expected {labels[i]}")
        out[i] = spec.compute(image)
    return out, time.perf_counter() - t0


def extract_features(load: SampleLoader, labels: Sequence[int], preprocess: PreprocessConfig = PreprocessConfig(),
                     hog: HOGConfig = HOGConfig(), progress: Optional[Callable] = None) -> tuple[np.ndarray, float]:
    """HOG features for every sample (Milestone 2 name)."""
    return extract_spec(load, labels, hog_spec(preprocess, hog), progress)


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------
def save_feature_cache(path: Path, fs: FeatureSet) -> None:
    """Write a feature set atomically (temporary file, then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez(tmp, features=fs.features, image_ids=np.asarray(fs.image_ids, dtype=str),
             labels=np.asarray(fs.labels, dtype=np.int64), meta=np.array(json.dumps(fs.meta, sort_keys=True)))
    tmp.replace(path)


def load_feature_cache(path: Path, expected: Mapping) -> tuple[Optional[FeatureSet], str]:
    """Load a cache file if it matches ``expected``.

    Returns:
        ``(FeatureSet, "ok")`` or ``(None, reason)`` when the file is missing,
        unreadable, or does not match the expected split/configuration/contents.
    """
    path = Path(path)
    if not path.is_file():
        return None, "no cache file"
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            features, image_ids, labels = data["features"], data["image_ids"], data["labels"]
    except Exception as exc:  # corrupted / truncated file
        return None, f"unreadable cache ({type(exc).__name__})"

    for key, value in expected.items():
        if meta.get(key) != value:
            return None, f"cache mismatch on '{key}'"
    if features.shape != (expected["n_samples"], expected["feature_length"]):
        return None, f"cache has feature shape {features.shape}"
    if samples_fingerprint(image_ids.tolist(), labels.tolist()) != expected["samples_sha256"]:
        return None, "cached image IDs / labels do not match the split"
    if not np.all(np.isfinite(features)):
        return None, "cache contains NaN or infinite values"
    return FeatureSet(expected["split"], features, image_ids, labels, meta, from_cache=True), "ok"


def get_or_extract(
    spec: FeatureSpec,
    split: str,
    image_ids: Sequence[str],
    labels: Sequence[int],
    load: SampleLoader,
    cache_dir: Path,
    force: bool = False,
    progress: Optional[Callable] = None,
) -> tuple[FeatureSet, str]:
    """Return one feature type of a split from a valid cache, or extract and cache it.

    Returns:
        ``(FeatureSet, status)`` where status explains whether the cache was used.
    """
    if len(image_ids) != len(labels):
        raise ValueError("image_ids and labels differ in length")
    expected = spec_expected_meta(split, image_ids, labels, spec)
    path = spec_cache_path(cache_dir, split, spec)

    if not force:
        fs, reason = load_feature_cache(path, expected)
        if fs is not None:
            return fs, f"loaded from cache {path.name}"
    else:
        reason = "force=True"

    features, seconds = extract_spec(load, labels, spec, progress)
    if not np.all(np.isfinite(features)):
        raise RuntimeError(f"{split}: {spec.kind} features contain NaN or infinite values")
    meta = {**expected, "extraction_seconds": seconds, "extraction_seconds_per_image": seconds / max(len(labels), 1),
            "created_unix": time.time()}
    fs = FeatureSet(split, features, np.asarray(image_ids, dtype=str), np.asarray(labels, dtype=np.int64),
                    meta, from_cache=False)
    save_feature_cache(path, fs)
    return fs, f"extracted ({reason}); saved {path.name}"


def get_or_extract_features(
    split: str,
    image_ids: Sequence[str],
    labels: Sequence[int],
    load: SampleLoader,
    cache_dir: Path,
    preprocess: PreprocessConfig = PreprocessConfig(),
    hog: HOGConfig = HOGConfig(),
    force: bool = False,
    progress: Optional[Callable] = None,
) -> tuple[FeatureSet, str]:
    """HOG features of a split from cache or freshly extracted (Milestone 2 name)."""
    return get_or_extract(hog_spec(preprocess, hog), split, image_ids, labels, load, cache_dir, force, progress)


# ---------------------------------------------------------------------------
# Split hygiene
# ---------------------------------------------------------------------------
def assert_disjoint_splits(feature_sets: Mapping[str, FeatureSet],
                           expected_sizes: Optional[Mapping[str, int]] = None) -> None:
    """Raise if any image appears in two splits, inside one split twice, or a size is wrong."""
    seen: dict[str, str] = {}
    for split, fs in feature_sets.items():
        ids = [str(i) for i in fs.image_ids]
        if len(set(ids)) != len(ids):
            raise AssertionError(f"{split}: duplicate image IDs")
        if expected_sizes is not None and split in expected_sizes and len(ids) != expected_sizes[split]:
            raise AssertionError(f"{split}: {len(ids)} samples, expected {expected_sizes[split]}")
        if not (len(fs.labels) == len(ids) == fs.features.shape[0]):
            raise AssertionError(f"{split}: features, IDs and labels are not aligned")
        for image_id in ids:
            if image_id in seen:
                raise AssertionError(f"{image_id} is in both {seen[image_id]} and {split}")
            seen[image_id] = split
