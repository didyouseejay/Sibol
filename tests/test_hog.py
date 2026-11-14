"""Tests for the from-scratch HOG implementation (src/classical/hog.py).

The primary correctness checks use small hand-checkable inputs (ramps and
edges whose gradients and bin votes can be worked out on paper). The
scikit-image comparison at the end is an optional, loose development check.
"""

import numpy as np
import pytest

from src.classical import hog as H
from src.classical.hog import HOGConfig

CFG = HOGConfig()


# --------------------------------------------------------------- dimensions
def test_milestone_configuration_gives_8100_features():
    assert CFG.grid_shape((128, 128)) == ((16, 16), (15, 15))
    assert CFG.feature_length((128, 128)) == 8100
    rng = np.random.default_rng(0)
    assert H.hog_features(rng.random((128, 128)), CFG).shape == (8100,)


def test_intermediate_shapes():
    gray = np.random.default_rng(1).random((128, 128))
    gx, gy = H.compute_gradients(gray)
    mag, ori = H.magnitude_orientation(gx, gy)
    cells = H.cell_histograms(mag, ori, CFG)
    blocks = H.build_blocks(cells, CFG)
    assert gx.shape == gy.shape == mag.shape == ori.shape == (128, 128)
    assert cells.shape == (16, 16, 9)
    assert blocks.shape == (15, 15, 2, 2, 9)


def test_other_configurations_have_consistent_length():
    cfg = HOGConfig(cell_size=16, block_size=3, block_stride=2)
    gray = np.random.default_rng(2).random((128, 96))
    (cy, cx), (by, bx) = cfg.grid_shape(gray.shape)
    assert (cy, cx, by, bx) == (8, 6, 3, 2)
    assert H.hog_features(gray, cfg).shape == (cfg.feature_length(gray.shape),) == (3 * 2 * 9 * 9,)


# ------------------------------------------------------------- gradients
def test_gradients_of_a_horizontal_ramp():
    # I(y, x) = 0.1 * x  ->  interior gx = I[x+1] - I[x-1] = 0.2, gy = 0
    gray = np.tile(0.1 * np.arange(10.0), (6, 1))
    gx, gy = H.compute_gradients(gray)
    np.testing.assert_allclose(gx[:, 1:-1], 0.2)
    np.testing.assert_allclose(gx[:, 0], 0.1)   # replicated border -> one-sided difference
    np.testing.assert_allclose(gx[:, -1], 0.1)
    np.testing.assert_allclose(gy, 0.0)


def test_orientation_is_unsigned_and_in_range():
    gx = np.array([[1.0, -1.0, 0.0, 0.0, 1.0, -1.0, -1e-17]])
    gy = np.array([[0.0, 0.0, 1.0, -1.0, 1.0, -1.0, -1e-17]])
    mag, ori = H.magnitude_orientation(gx, gy)
    np.testing.assert_allclose(ori[0, :6], [0, 0, 90, 90, 45, 45], atol=1e-9)
    assert np.all((ori >= 0) & (ori < 180))
    np.testing.assert_allclose(mag[0, 4], np.sqrt(2))


# ------------------------------------------------------- histogram voting
def _single_cell_hist(gray):
    cfg = HOGConfig(cell_size=gray.shape[0], block_size=1)
    mag, ori = H.magnitude_orientation(*H.compute_gradients(gray))
    return H.cell_histograms(mag, ori, cfg)[0, 0], mag


def test_horizontal_edge_votes_only_into_the_90_degree_bin():
    gray = np.zeros((8, 8))
    gray[4:, :] = 1.0                       # intensity changes along y -> 90 degrees
    hist, mag = _single_cell_hist(gray)
    assert hist[4] == pytest.approx(mag.sum())   # bin 4 is centred at 90 degrees
    assert np.count_nonzero(hist) == 1


def test_vertical_edge_splits_equally_between_bins_0_and_8():
    gray = np.zeros((8, 8))
    gray[:, 4:] = 1.0                       # 0 degrees: halfway between centres 170 and 10
    hist, mag = _single_cell_hist(gray)
    assert hist[0] == pytest.approx(mag.sum() / 2)
    assert hist[8] == pytest.approx(mag.sum() / 2)
    assert hist[1:8].sum() == pytest.approx(0.0)


def test_45_degree_ramp_interpolates_between_bins_1_and_2():
    # I = 0.05 * (x + y): angle 45 degrees. Position 45/20 - 0.5 = 1.75,
    # so bin 1 (30 deg) gets 25 % and bin 2 (50 deg) gets 75 % of the magnitude.
    # Border pixels have one-sided differences (not 45 degrees), so the cell is
    # taken from the interior of a larger ramp.
    y, x = np.mgrid[0:10, 0:10]
    gray = 0.05 * (x + y).astype(float)
    mag, ori = H.magnitude_orientation(*H.compute_gradients(gray))
    mag, ori = mag[1:9, 1:9], ori[1:9, 1:9]
    np.testing.assert_allclose(ori, 45.0)
    np.testing.assert_allclose(mag, 0.1 * np.sqrt(2))
    hist = H.cell_histograms(mag, ori, HOGConfig(cell_size=8, block_size=1))[0, 0]
    np.testing.assert_allclose(hist[1], 0.25 * mag.sum())
    np.testing.assert_allclose(hist[2], 0.75 * mag.sum())
    assert hist.sum() == pytest.approx(mag.sum())    # votes conserve magnitude


def test_cell_histograms_sum_pixels_of_their_own_cell_only():
    mag = np.zeros((16, 16))
    mag[9, 2] = 3.0                         # pixel inside cell (1, 0)
    ori = np.full((16, 16), 90.0)
    cells = H.cell_histograms(mag, ori, HOGConfig(cell_size=8))
    assert cells[1, 0, 4] == pytest.approx(3.0)
    assert cells.sum() == pytest.approx(3.0)


# ------------------------------------------------------------ blocks / norm
def test_build_blocks_order_and_overlap():
    cfg = HOGConfig(n_bins=1)
    cells = np.arange(9.0).reshape(3, 3, 1)
    blocks = H.build_blocks(cells, cfg)
    assert blocks.shape == (2, 2, 2, 2, 1)
    np.testing.assert_array_equal(blocks[0, 0, :, :, 0], [[0, 1], [3, 4]])
    np.testing.assert_array_equal(blocks[1, 1, :, :, 0], [[4, 5], [7, 8]])


def test_l2_hys_clips_and_renormalises():
    v = np.array([[10.0, 1.0, 1.0, 0.0]])
    out = H.l2_hys(v, clip=0.2, eps=0.0)
    # after the first L2 step the large entry is clipped to 0.2, then renormalised:
    first = v / np.linalg.norm(v)
    clipped = np.minimum(first, 0.2)
    np.testing.assert_allclose(out, clipped / np.linalg.norm(clipped))
    assert np.linalg.norm(out) == pytest.approx(1.0)


def test_l2_hys_keeps_zero_blocks_finite():
    out = H.l2_hys(np.zeros((3, 36)))
    assert np.all(out == 0)


# ------------------------------------------------------- numerical validity
@pytest.mark.parametrize("image", ["zeros", "ones", "noise", "checker"])
def test_hog_is_finite_and_bounded(image):
    rng = np.random.default_rng(3)
    gray = {"zeros": np.zeros((128, 128)), "ones": np.ones((128, 128)), "noise": rng.random((128, 128)),
            "checker": (np.indices((128, 128)).sum(axis=0) % 2).astype(float)}[image]
    f = H.hog_features(gray)
    assert np.all(np.isfinite(f))
    assert f.min() >= 0
    block_norms = np.linalg.norm(f.reshape(225, 36), axis=1)
    assert np.all(block_norms <= 1.0 + 1e-9)


def test_constant_image_gives_zero_descriptor():
    assert np.all(H.hog_features(np.full((128, 128), 0.7)) == 0)


def test_hog_is_invariant_to_brightness_offset_and_contrast_scale():
    gray = np.random.default_rng(4).random((128, 128)) * 0.5
    a, b = H.hog_features(gray), H.hog_features(0.2 + 1.5 * gray)
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_config_fingerprint_changes_with_parameters():
    assert HOGConfig().fingerprint() == HOGConfig().fingerprint()
    assert HOGConfig().fingerprint() != HOGConfig(n_bins=12).fingerprint()


# ------------------------------------- optional development-time reference
def test_close_to_scikit_image_reference():
    """Loose agreement with skimage.feature.hog (development check only).

    The implementations differ on purpose (skimage uses hard binning and zero
    gradients on the border; ours interpolates between bins and replicates
    edges), so only a high correlation is required, not equality.
    scikit-image is never used to compute SIBOL's features.
    """
    skfeature = pytest.importorskip("skimage.feature")
    rng = np.random.default_rng(5)
    # smooth random image: sum of a few blurred blobs
    y, x = np.mgrid[0:128, 0:128]
    gray = np.zeros((128, 128))
    for _ in range(12):
        cy, cx, s = rng.uniform(0, 128), rng.uniform(0, 128), rng.uniform(6, 20)
        gray += rng.uniform(0.3, 1.0) * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * s * s))
    gray /= gray.max()

    ref = skfeature.hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                        block_norm="L2-Hys", feature_vector=True)
    ours = H.hog_features(gray)
    assert ref.shape == ours.shape == (8100,)
    corr = np.corrcoef(ref, ours)[0, 1]
    assert corr > 0.8, f"correlation with scikit-image HOG only {corr:.3f}"
