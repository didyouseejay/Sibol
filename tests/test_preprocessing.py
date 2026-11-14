"""Tests for preprocessing and feature standardisation (src/classical/preprocessing.py)."""

import numpy as np
import pytest
from PIL import Image

from src.classical.preprocessing import (FeatureStandardizer, PreprocessConfig, preprocess_image, rgb_to_gray)


def rgb_image(w=500, h=667, seed=0):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8))


def test_output_shape_dtype_and_range():
    out = preprocess_image(rgb_image())
    assert out.shape == (128, 128)
    assert out.dtype == np.float64
    assert 0.0 <= out.min() and out.max() <= 1.0


@pytest.mark.parametrize("size", [(500, 500), (857, 500), (500, 861), (37, 12)])
def test_any_input_size_becomes_square(size):
    assert preprocess_image(rgb_image(*size)).shape == (128, 128)


def test_preprocessing_is_deterministic_and_input_type_independent(tmp_path):
    img = rgb_image()
    path = tmp_path / "x.png"
    img.save(path)
    a = preprocess_image(img)
    np.testing.assert_array_equal(a, preprocess_image(img))
    np.testing.assert_array_equal(a, preprocess_image(path))
    np.testing.assert_array_equal(a, preprocess_image(np.asarray(img)))


def test_grayscale_weights_bt601():
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
    np.testing.assert_allclose(rgb_to_gray(rgb)[0], [0.299, 0.587, 0.114, 1.0])


def test_grayscale_close_to_pillow_L_mode():
    img = rgb_image(64, 64)
    ours = rgb_to_gray(np.asarray(img))
    pillow = np.asarray(img.convert("L"), dtype=np.float64) / 255.0
    assert np.abs(ours - pillow).max() <= 1.0 / 255 + 1e-9   # Pillow rounds to integers


def test_grayscale_input_is_converted_to_rgb_first():
    gray_img = Image.fromarray(np.full((50, 80), 128, dtype=np.uint8), mode="L")
    np.testing.assert_allclose(preprocess_image(gray_img), 128 / 255)


def test_config_validation():
    with pytest.raises(ValueError):
        PreprocessConfig(image_size=0)
    with pytest.raises(ValueError):
        PreprocessConfig(resample="lanczos-ish")
    with pytest.raises(ValueError):
        PreprocessConfig(resize_mode="pad")


def test_milestone2_cache_key_is_unchanged():
    """The direct-mode description and the HOG cache hash must not change: Drive caches depend on them."""
    from src.classical import features as F
    from src.classical.hog import HOGConfig

    assert PreprocessConfig().to_dict() == {"image_size": 128, "resample": "bilinear",
                                            "resize_mode": "direct (aspect ratio not preserved)",
                                            "grayscale": "BT.601"}
    assert F.config_hash(PreprocessConfig(), HOGConfig()) == "b5525eb6a0c2"
    assert F.config_hash(PreprocessConfig(resize_mode="center_crop"), HOGConfig()) != "b5525eb6a0c2"


def test_center_square_box():
    from src.classical.preprocessing import center_square_box

    assert center_square_box(300, 100) == (100, 0, 200, 100)
    assert center_square_box(100, 301) == (0, 100, 100, 200)
    assert center_square_box(50, 50) == (0, 0, 50, 50)


def test_center_crop_removes_the_ends_of_the_long_side():
    from src.classical.preprocessing import preprocess_rgb

    arr = np.zeros((100, 300, 3), np.uint8)
    arr[:, :100] = [255, 0, 0]       # left third red
    arr[:, 100:200] = [0, 255, 0]    # middle third green
    arr[:, 200:] = [0, 0, 255]       # right third blue
    img = Image.fromarray(arr)
    crop = preprocess_rgb(img, PreprocessConfig(resize_mode="center_crop"))
    assert crop.shape == (128, 128, 3)
    np.testing.assert_array_equal(crop[2:-2, 2:-2], np.broadcast_to([0, 255, 0], (124, 124, 3)))
    direct = preprocess_rgb(img, PreprocessConfig())
    assert direct[64, 5, 0] == 255 and direct[64, -5, 2] == 255     # direct keeps (squashes) all three


def test_grayscale_path_is_unchanged_by_refactor():
    from src.classical.preprocessing import preprocess_rgb

    img = rgb_image()
    np.testing.assert_array_equal(preprocess_image(img), rgb_to_gray(preprocess_rgb(img)))


# ------------------------------------------------------------- standardiser
def test_standardizer_uses_training_statistics_only():
    rng = np.random.default_rng(0)
    train = rng.normal(5.0, 2.0, size=(200, 6))
    other = rng.normal(-3.0, 0.5, size=(50, 6))
    sc = FeatureStandardizer().fit(train)
    np.testing.assert_allclose(sc.mean_, train.mean(axis=0))
    np.testing.assert_allclose(sc.scale_, train.std(axis=0))
    z = sc.transform(train)
    np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(z.std(axis=0), 1, atol=1e-12)
    # transforming other data must not refit: its mean is NOT zero afterwards
    assert np.all(sc.transform(other).mean(axis=0) < -2)
    np.testing.assert_allclose(sc.mean_, train.mean(axis=0))
    assert sc.n_fit_samples_ == 200


def test_standardizer_constant_features_stay_finite():
    X = np.column_stack([np.ones(10), np.arange(10.0)])
    z = FeatureStandardizer().fit_transform(X)
    assert np.all(np.isfinite(z))
    np.testing.assert_array_equal(z[:, 0], 0)


def test_standardizer_requires_fit_and_matching_width():
    with pytest.raises(RuntimeError):
        FeatureStandardizer().transform(np.zeros((2, 3)))
    sc = FeatureStandardizer().fit(np.random.default_rng(1).random((5, 3)))
    with pytest.raises(ValueError):
        sc.transform(np.zeros((2, 4)))


def test_standardizer_save_load(tmp_path):
    X = np.random.default_rng(2).random((20, 4))
    sc = FeatureStandardizer().fit(X)
    sc.save(tmp_path / "s.npz")
    np.testing.assert_array_equal(FeatureStandardizer.load(tmp_path / "s.npz").transform(X), sc.transform(X))
