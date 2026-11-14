"""Image preprocessing and feature standardisation for the classical pipeline.

Pipeline for one image (Milestone 2 configuration)::

    RGB image (any size)  ->  resize to 128 x 128 (direct, aspect ratio NOT kept)
                          ->  grayscale (ITU-R BT.601 luma, NumPy)
                          ->  float64 array in [0, 1], shape (128, 128)

Resizing is done with Pillow (bilinear). Grayscale conversion is written out
in NumPy so the whole numerical path into HOG is visible.

Known limitation: Flowers-102 aspect ratios range from about 0.58 to 1.71, so
a direct resize to a square stretches or squashes many flowers. This is the
Milestone 2 setting (``resize_mode="direct"``). Milestone 2b adds
``resize_mode="center_crop"`` (largest centred square, then resize) as an
ablation option.

:class:`FeatureStandardizer` rescales each feature to zero mean and unit
variance. It must be fitted on the **training split only** and then applied
unchanged to validation and test features.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

# ITU-R BT.601 luma weights (same as Pillow's "L" mode and most textbooks).
GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114])

_RESAMPLE = {
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "nearest": Image.Resampling.NEAREST,
}

ImageLike = Union[Image.Image, np.ndarray, str, Path]


RESIZE_MODES = ("direct", "center_crop")


@dataclass(frozen=True)
class PreprocessConfig:
    """Settings of the preprocessing step (part of the feature-cache key).

    Attributes:
        image_size: Output side length in pixels (square output).
        resample: Pillow resampling filter: ``"bilinear"``, ``"bicubic"`` or ``"nearest"``.
        resize_mode: ``"direct"`` stretches the whole image to a square (aspect
            ratio not preserved; the Milestone 2 setting). ``"center_crop"``
            takes the largest centred square, then resizes it (aspect ratio
            preserved; the ends of the longer side are cut off).
    """

    image_size: int = 128
    resample: str = "bilinear"
    resize_mode: str = "direct"

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.resample not in _RESAMPLE:
            raise ValueError(f"resample must be one of {sorted(_RESAMPLE)}, got {self.resample!r}")
        if self.resize_mode not in RESIZE_MODES:
            raise ValueError(f"resize_mode must be one of {RESIZE_MODES}, got {self.resize_mode!r}")

    def to_dict(self) -> dict:
        # The "direct" description must stay byte-identical: it is part of the
        # cache key of the Milestone 2 HOG features already stored on Drive.
        mode = ("direct (aspect ratio not preserved)" if self.resize_mode == "direct"
                else "center_crop (aspect ratio preserved, longer side cropped)")
        return {"image_size": self.image_size, "resample": self.resample,
                "resize_mode": mode, "grayscale": "BT.601"}


def to_pil_rgb(image: ImageLike) -> Image.Image:
    """Return ``image`` as an RGB Pillow image.

    Accepts a Pillow image, a file path, or a uint8 array of shape (H, W) or (H, W, 3).
    """
    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            return im.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            raise ValueError("array images must be uint8")
        return Image.fromarray(image).convert("RGB")
    return image.convert("RGB")


def center_square_box(width: int, height: int) -> tuple[int, int, int, int]:
    """``(left, top, right, bottom)`` of the largest centred square inside a ``width x height`` image."""
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    return left, top, left + side, top + side


def resize_image(image: Image.Image, config: PreprocessConfig) -> Image.Image:
    """Resize to ``image_size x image_size`` according to ``config.resize_mode``."""
    size = (config.image_size, config.image_size)
    box = center_square_box(*image.size) if config.resize_mode == "center_crop" else None
    return image.resize(size, resample=_RESAMPLE[config.resample], box=box)


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB array (H, W, 3) to grayscale float64 in [0, 1].

    uint8 input is divided by 255. Float input is assumed to already be in [0, 1].
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) array, got shape {rgb.shape}")
    arr = rgb.astype(np.float64)
    if rgb.dtype == np.uint8:
        arr /= 255.0
    return arr @ GRAY_WEIGHTS


def preprocess_rgb(image: ImageLike, config: PreprocessConfig = PreprocessConfig()) -> np.ndarray:
    """RGB -> resize. Returns the resized colour image as uint8 ``(S, S, 3)``."""
    return np.asarray(resize_image(to_pil_rgb(image), config), dtype=np.uint8)


def preprocess_image(image: ImageLike, config: PreprocessConfig = PreprocessConfig()) -> np.ndarray:
    """Full preprocessing: RGB -> resize -> grayscale. Returns float64 (S, S) in [0, 1]."""
    return rgb_to_gray(preprocess_rgb(image, config))


class FeatureStandardizer:
    """Per-feature standardisation ``(x - mean) / std``, written in NumPy.

    Fit on training features only. Features with (near-)zero training variance
    keep a scale of 1 so they stay finite (they become constant 0).
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_fit_samples_: int | None = None

    def fit(self, X: np.ndarray) -> "FeatureStandardizer":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 1:
            raise ValueError(f"expected a non-empty 2-D array, got shape {X.shape}")
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.scale_ = np.where(std < self.eps, 1.0, std)
        self.n_fit_samples_ = X.shape[0]
        return self

    def transform(self, X: np.ndarray, dtype=np.float64, chunk_size: int = 1024) -> np.ndarray:
        """Standardise ``X`` with the fitted statistics, in row chunks to bound peak memory."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("FeatureStandardizer is not fitted; call fit() on the training features")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.mean_.shape[0]:
            raise ValueError(f"expected (N, {self.mean_.shape[0]}) features, got {X.shape}")
        out = np.empty(X.shape, dtype=dtype)
        for start in range(0, X.shape[0], chunk_size):
            block = X[start : start + chunk_size].astype(np.float64)
            out[start : start + chunk_size] = (block - self.mean_) / self.scale_
        return out

    def fit_transform(self, X: np.ndarray, dtype=np.float64) -> np.ndarray:
        return self.fit(X).transform(X, dtype=dtype)

    def save(self, path: Path) -> None:
        if self.mean_ is None:
            raise RuntimeError("cannot save an unfitted FeatureStandardizer")
        np.savez(Path(path), mean=self.mean_, scale=self.scale_,
                 meta=np.array(json.dumps({"eps": self.eps, "n_fit_samples": self.n_fit_samples_})))

    @classmethod
    def load(cls, path: Path) -> "FeatureStandardizer":
        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            obj = cls(eps=meta["eps"])
            obj.mean_, obj.scale_ = data["mean"], data["scale"]
            obj.n_fit_samples_ = meta["n_fit_samples"]
        return obj
