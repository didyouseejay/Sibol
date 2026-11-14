"""Tests for the from-scratch HSV colour histograms (src/classical/color.py)."""

import colorsys

import numpy as np
import pytest

from src.classical import color as C
from src.classical.color import ColorConfig


def test_rgb_to_hsv_matches_colorsys():
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(500, 3), dtype=np.uint8)
    pixels[:5] = [[0, 0, 0], [255, 255, 255], [128, 128, 128], [255, 0, 0], [0, 0, 255]]   # edge cases
    ours = C.rgb_to_hsv(pixels)
    ref = np.array([colorsys.rgb_to_hsv(*(p / 255.0)) for p in pixels])
    np.testing.assert_allclose(ours, ref, atol=1e-12)
    assert np.all((ours >= 0) & (ours <= 1)) and np.all(ours[:, 0] < 1)


def test_feature_length_and_region_sums():
    cfg = ColorConfig()
    assert cfg.feature_length == 320 == 5 * 16 * 4
    rgb = np.random.default_rng(1).integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    f = C.color_histogram(rgb, cfg)
    assert f.shape == (320,) and np.all(np.isfinite(f)) and f.min() >= 0
    np.testing.assert_allclose(f.reshape(5, 64).sum(axis=1), 1.0)


def test_pure_colours_land_in_expected_bins():
    cfg = ColorConfig(grid=1, include_global=False)
    red = np.zeros((8, 8, 3), np.uint8)
    red[..., 0] = 255                           # hue 0, saturation 1
    f = C.color_histogram(red, cfg)
    assert f[0 * 4 + 3] == pytest.approx(1.0)   # hue bin 0, top saturation bin
    blue = np.zeros((8, 8, 3), np.uint8)
    blue[..., 2] = 255                          # hue 240/360 = 0.667 -> bin 10 of 16
    assert C.color_histogram(blue, cfg)[10 * 4 + 3] == pytest.approx(1.0)
    gray = np.full((8, 8, 3), 100, np.uint8)    # no saturation -> hue 0, saturation bin 0
    assert C.color_histogram(gray, cfg)[0] == pytest.approx(1.0)


def test_spatial_layout_separates_left_and_right():
    img = np.zeros((128, 128, 3), np.uint8)
    img[:, :64, 0] = 255                        # left half red
    img[:, 64:, 2] = 255                        # right half blue
    f = C.color_histogram(img).reshape(5, 64)
    red_bin, blue_bin = 0 * 4 + 3, 10 * 4 + 3
    assert f[0, red_bin] == pytest.approx(0.5) and f[0, blue_bin] == pytest.approx(0.5)  # global
    assert f[1, red_bin] == pytest.approx(1.0) and f[2, blue_bin] == pytest.approx(1.0)  # top-left, top-right


@pytest.mark.parametrize("cfg", [ColorConfig(), ColorConfig(grid=3), ColorConfig(include_global=False)])
def test_flip_permutation_is_exact(cfg):
    rgb = np.random.default_rng(2).integers(0, 256, size=(129, 129, 3), dtype=np.uint8)[:126, :126]
    f = C.color_histogram(rgb, cfg)
    np.testing.assert_array_equal(C.color_histogram(rgb[:, ::-1], cfg), f[C.flip_permutation(cfg)])


def test_config_validation_and_fingerprint():
    with pytest.raises(ValueError):
        ColorConfig(h_bins=0)
    assert ColorConfig().fingerprint() != ColorConfig(h_bins=8).fingerprint()
