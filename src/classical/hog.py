"""Histogram of Oriented Gradients (HOG), implemented from scratch in NumPy.

No ready-made HOG (scikit-image, OpenCV, ...) is used here. scikit-image is
only used in an optional development test as a loose reference.

Milestone 2 configuration (Dalal & Triggs, CVPR 2005, style)::

    input            128 x 128 grayscale, float in [0, 1]
    cells            8 x 8 pixels      -> 16 x 16 cells
    orientations     9 bins over 0-180 degrees (unsigned), 20 degrees wide
    blocks           2 x 2 cells, stride 1 cell -> 15 x 15 blocks
    normalisation    L2-Hys (L2, clip at 0.2, L2 again)
    feature length   15 * 15 * (2 * 2 * 9) = 8,100

Steps, one function each:

1. :func:`compute_gradients`    centred [-1, 0, 1] differences
2. :func:`magnitude_orientation` magnitude and unsigned angle in [0, 180)
3. :func:`cell_histograms`       magnitude-weighted votes, linear interpolation
                                 between the two nearest orientation bins
4. :func:`build_blocks`          overlapping 2 x 2 groups of cell histograms
5. :func:`l2_hys`                block normalisation
6. :func:`hog_features`          everything above, flattened into one vector

Conventions: rows (y) increase downwards, columns (x) to the right. An angle
of 0 degrees means intensity changes along x (a vertical edge); 90 degrees
means it changes along y (a horizontal edge). Bin ``k`` is centred at
``(k + 0.5) * 20`` degrees: 10, 30, ..., 170.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

# Bump when the numerical output of this module changes; part of the cache key.
HOG_IMPL_VERSION = 1


@dataclass(frozen=True)
class HOGConfig:
    """HOG parameters (part of the feature-cache key).

    Attributes:
        cell_size: Cell side length in pixels.
        n_bins: Number of orientation bins over 0-180 degrees.
        block_size: Block side length in cells.
        block_stride: Step between blocks, in cells.
        clip: L2-Hys clipping threshold.
        eps: Small constant in the L2 norms (avoids division by zero).
    """

    cell_size: int = 8
    n_bins: int = 9
    block_size: int = 2
    block_stride: int = 1
    clip: float = 0.2
    eps: float = 1e-5

    def __post_init__(self) -> None:
        for name in ("cell_size", "n_bins", "block_size", "block_stride"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if not 0 < self.clip <= 1:
            raise ValueError("clip must be in (0, 1]")

    @property
    def bin_width(self) -> float:
        """Width of one orientation bin in degrees (unsigned range of 180)."""
        return 180.0 / self.n_bins

    def grid_shape(self, image_shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return ``((cells_y, cells_x), (blocks_y, blocks_x))`` for an image shape."""
        cy, cx = image_shape[0] // self.cell_size, image_shape[1] // self.cell_size
        by = (cy - self.block_size) // self.block_stride + 1
        bx = (cx - self.block_size) // self.block_stride + 1
        if cy < self.block_size or cx < self.block_size:
            raise ValueError(f"image {image_shape} is too small for one block")
        return (cy, cx), (by, bx)

    def feature_length(self, image_shape: tuple[int, int] = (128, 128)) -> int:
        """Length of the HOG vector for an image of ``image_shape``."""
        _, (by, bx) = self.grid_shape(image_shape)
        return by * bx * self.block_size**2 * self.n_bins

    def to_dict(self) -> dict:
        return {**asdict(self), "orientation_range_deg": 180, "signed": False,
                "block_norm": "L2-Hys", "impl_version": HOG_IMPL_VERSION}

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# 1. Gradients
# ---------------------------------------------------------------------------
def compute_gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal and vertical gradients with the centred [-1, 0, 1] kernel.

    ``gx[y, x] = I[y, x+1] - I[y, x-1]`` and ``gy[y, x] = I[y+1, x] - I[y-1, x]``.
    Borders are handled by replicating the edge pixel, which turns the
    centred difference into a one-sided one there.

    Args:
        gray: 2-D float image.

    Returns:
        ``(gx, gy)``, each with the shape of ``gray``.
    """
    if gray.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image, got shape {gray.shape}")
    p = np.pad(np.asarray(gray, dtype=np.float64), 1, mode="edge")
    gx = p[1:-1, 2:] - p[1:-1, :-2]
    gy = p[2:, 1:-1] - p[:-2, 1:-1]
    return gx, gy


# ---------------------------------------------------------------------------
# 2. Magnitude and orientation
# ---------------------------------------------------------------------------
def magnitude_orientation(gx: np.ndarray, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gradient magnitude and unsigned orientation in degrees, in [0, 180).

    Opposite gradient directions (e.g. dark-to-bright and bright-to-dark across
    the same edge) map to the same orientation.
    """
    magnitude = np.hypot(gx, gy)
    orientation = np.degrees(np.arctan2(gy, gx)) % 180.0
    orientation[orientation >= 180.0] = 0.0  # guard: -tiny % 180 can round to 180.0
    return magnitude, orientation


# ---------------------------------------------------------------------------
# 3. Cell histograms
# ---------------------------------------------------------------------------
def cell_histograms(magnitude: np.ndarray, orientation: np.ndarray, config: HOGConfig = HOGConfig()) -> np.ndarray:
    """Orientation histogram of every cell.

    Each pixel votes with its gradient magnitude. The vote is split linearly
    between the two orientation bins whose centres are nearest to its angle,
    wrapping around at 0/180 degrees (e.g. 5 degrees is shared between the
    bins centred at 170 and 10). Pixels beyond the last full cell are ignored.

    Returns:
        Array of shape ``(cells_y, cells_x, n_bins)``.
    """
    (cy, cx), _ = config.grid_shape(magnitude.shape)
    cs, n = config.cell_size, config.n_bins
    mag = magnitude[: cy * cs, : cx * cs]
    ori = orientation[: cy * cs, : cx * cs]

    pos = ori / config.bin_width - 0.5          # position in units of bins, relative to centres
    lo = np.floor(pos)
    frac = pos - lo                              # in [0, 1): share of the upper bin
    lo = lo.astype(np.int64) % n
    hi = (lo + 1) % n

    votes = np.zeros(mag.shape + (n,))
    rows, cols = np.indices(mag.shape)
    votes[rows, cols, lo] = mag * (1.0 - frac)
    votes[rows, cols, hi] += mag * frac          # with n_bins == 1, lo == hi and the shares add up to mag
    return votes.reshape(cy, cs, cx, cs, n).sum(axis=(1, 3))


# ---------------------------------------------------------------------------
# 4. Blocks
# ---------------------------------------------------------------------------
def build_blocks(cell_hist: np.ndarray, config: HOGConfig = HOGConfig()) -> np.ndarray:
    """Group cell histograms into overlapping blocks.

    Block ``(i, j)`` covers cells ``[i*s : i*s+b, j*s : j*s+b]`` (``s`` = stride,
    ``b`` = block size). Inside a block the cells are ordered row by row and
    each contributes its ``n_bins`` values.

    Returns:
        Array of shape ``(blocks_y, blocks_x, block_size, block_size, n_bins)``.
    """
    cy, cx, n = cell_hist.shape
    b, s = config.block_size, config.block_stride
    by, bx = (cy - b) // s + 1, (cx - b) // s + 1
    blocks = np.empty((by, bx, b, b, n), dtype=cell_hist.dtype)
    for dy in range(b):
        for dx in range(b):
            blocks[:, :, dy, dx, :] = cell_hist[dy : dy + (by - 1) * s + 1 : s, dx : dx + (bx - 1) * s + 1 : s, :]
    return blocks


# ---------------------------------------------------------------------------
# 5. L2-Hys normalisation
# ---------------------------------------------------------------------------
def l2_hys(v: np.ndarray, clip: float = 0.2, eps: float = 1e-5) -> np.ndarray:
    """L2-Hys normalisation along the last axis.

    1. ``v <- v / sqrt(||v||^2 + eps^2)``
    2. ``v <- min(v, clip)``  (HOG values are non-negative)
    3. ``v <- v / sqrt(||v||^2 + eps^2)``

    An all-zero vector stays all-zero (no NaN).
    """
    v = v / np.sqrt(np.sum(v * v, axis=-1, keepdims=True) + eps**2)
    v = np.minimum(v, clip)
    return v / np.sqrt(np.sum(v * v, axis=-1, keepdims=True) + eps**2)


# ---------------------------------------------------------------------------
# 6. Full descriptor
# ---------------------------------------------------------------------------
def hog_features(gray: np.ndarray, config: HOGConfig = HOGConfig()) -> np.ndarray:
    """HOG descriptor of one grayscale image, as a 1-D float64 vector.

    Order: blocks row by row, then cells inside a block row by row, then
    orientation bins (same layout as scikit-image's ``feature_vector=True``).
    """
    gx, gy = compute_gradients(gray)
    magnitude, orientation = magnitude_orientation(gx, gy)
    cells = cell_histograms(magnitude, orientation, config)
    blocks = build_blocks(cells, config)
    by, bx = blocks.shape[:2]
    normed = l2_hys(blocks.reshape(by, bx, -1), clip=config.clip, eps=config.eps)
    return normed.ravel()
