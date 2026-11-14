"""Colour histograms in HSV space, implemented from scratch in NumPy.

HOG works on grayscale and therefore cannot see colour, which separates many
Flowers-102 classes. This module adds a compact colour descriptor
(Milestone 2b):

1. :func:`rgb_to_hsv`       RGB in [0, 1] -> hue, saturation, value in [0, 1]
2. :func:`color_histogram`  joint hue x saturation histogram for the whole
                            image and for each cell of a ``grid x grid``
                            spatial layout; each region is L1-normalised.

Default: 16 hue bins x 4 saturation bins = 64 values per region,
1 global + 2 x 2 grid regions -> 5 x 64 = **320** values.

Value (brightness) is left out on purpose: it mostly reflects lighting.
Pixels with no saturation have an undefined hue; the formula assigns them
hue 0, so they all land in the lowest saturation bins of hue bin 0, which is
consistent across images.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

# Bump when the numerical output of this module changes; part of the cache key.
COLOR_IMPL_VERSION = 1


@dataclass(frozen=True)
class ColorConfig:
    """Colour-histogram parameters (part of the feature-cache key)."""

    h_bins: int = 16
    s_bins: int = 4
    grid: int = 2
    include_global: bool = True

    def __post_init__(self) -> None:
        if min(self.h_bins, self.s_bins, self.grid) < 1:
            raise ValueError("h_bins, s_bins and grid must be >= 1")

    @property
    def n_regions(self) -> int:
        return self.grid * self.grid + (1 if self.include_global else 0)

    @property
    def feature_length(self) -> int:
        return self.n_regions * self.h_bins * self.s_bins

    def to_dict(self) -> dict:
        return {**asdict(self), "space": "HSV (hue x saturation)", "region_norm": "L1",
                "impl_version": COLOR_IMPL_VERSION}

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to HSV, all channels in [0, 1].

    Args:
        rgb: ``(..., 3)`` array, uint8 (0-255) or float in [0, 1].

    Returns:
        ``(..., 3)`` float64 array of (hue, saturation, value); hue in [0, 1)
        (multiply by 360 for degrees). Same convention as Python's ``colorsys``.
    """
    x = np.asarray(rgb, dtype=np.float64)
    if np.asarray(rgb).dtype == np.uint8:
        x = x / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    v = x.max(axis=-1)
    c = v - x.min(axis=-1)                                    # chroma
    s = np.divide(c, v, out=np.zeros_like(v), where=v > 0)
    safe_c = np.where(c > 0, c, 1.0)
    h = np.where(v == r, (g - b) / safe_c,
        np.where(v == g, 2.0 + (b - r) / safe_c, 4.0 + (r - g) / safe_c))
    h = np.where(c > 0, (h / 6.0) % 1.0, 0.0)
    return np.stack([h, s, v], axis=-1)


def _hs_histogram(h: np.ndarray, s: np.ndarray, config: ColorConfig) -> np.ndarray:
    hi = np.minimum((h * config.h_bins).astype(np.int64), config.h_bins - 1)
    si = np.minimum((s * config.s_bins).astype(np.int64), config.s_bins - 1)
    hist = np.bincount((hi * config.s_bins + si).ravel(), minlength=config.h_bins * config.s_bins)
    return hist / max(h.size, 1)


def color_histogram(rgb: np.ndarray, config: ColorConfig = ColorConfig()) -> np.ndarray:
    """Spatial hue x saturation histogram of an RGB image.

    Layout: ``[global, cell(0,0), cell(0,1), ..., cell(g-1,g-1)]`` (cells row
    by row); inside a region the index is ``hue_bin * s_bins + sat_bin``.
    Each region sums to 1.

    Args:
        rgb: ``(H, W, 3)`` image, uint8 or float in [0, 1]. H and W should be
            divisible by ``grid`` (extra rows/columns are ignored by the grid).
    """
    hsv = rgb_to_hsv(rgb)
    h, s = hsv[..., 0], hsv[..., 1]
    parts = [_hs_histogram(h, s, config)] if config.include_global else []
    g = config.grid
    rh, rw = h.shape[0] // g, h.shape[1] // g
    for i in range(g):
        for j in range(g):
            sl = (slice(i * rh, (i + 1) * rh), slice(j * rw, (j + 1) * rw))
            parts.append(_hs_histogram(h[sl], s[sl], config))
    return np.concatenate(parts)


def flip_permutation(config: ColorConfig = ColorConfig()) -> np.ndarray:
    """Index array ``p`` with ``color_histogram(fliplr(img)) == color_histogram(img)[p]``.

    Colours are unchanged by mirroring; only the grid columns swap. Exact when
    the image width is divisible by ``grid``.
    """
    per = config.h_bins * config.s_bins
    order = []
    if config.include_global:
        order.append(0)
    offset = 1 if config.include_global else 0
    for i in range(config.grid):
        for j in range(config.grid):
            order.append(offset + i * config.grid + (config.grid - 1 - j))
    return np.concatenate([np.arange(r * per, (r + 1) * per) for r in order])
