"""Tests for the from-scratch one-vs-rest linear SVM (src/classical/linear_svm.py)."""

import numpy as np
import pytest

from src.classical.linear_svm import LinearSVMOvR, SVMConfig


def blobs(n_per_class=30, n_classes=4, dim=20, spread=0.5, seed=0):
    """Well-separated Gaussian blobs: an easy, linearly separable problem."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 3, size=(n_classes, dim))
    X = np.concatenate([c + spread * rng.normal(size=(n_per_class, dim)) for c in centres])
    y = np.repeat(np.arange(n_classes), n_per_class)
    return X, y


def test_learns_separable_synthetic_data():
    X, y = blobs()
    model = LinearSVMOvR(SVMConfig(learning_rate=0.05, reg_strength=1e-3, n_epochs=30, batch_size=16)).fit(X, y)
    assert (model.predict(X) == y).mean() == 1.0
    Xt, yt = blobs(seed=0, n_per_class=10)  # same centres (same seed), fresh-ish noise
    assert (model.predict(Xt) == yt).mean() >= 0.95


def test_score_and_prediction_dimensions():
    X, y = blobs(n_classes=5, dim=12)
    model = LinearSVMOvR(SVMConfig(n_epochs=3)).fit(X, y)
    scores = model.decision_function(X[:7])
    assert scores.shape == (7, 5)
    assert model.predict(X[:7]).shape == (7,)
    assert model.W.shape == (12, 5) and model.b.shape == (5,)


def test_prediction_is_argmax_of_scores():
    X, y = blobs()
    model = LinearSVMOvR(SVMConfig(n_epochs=5)).fit(X, y)
    np.testing.assert_array_equal(model.predict(X), model.classes_[model.decision_function(X).argmax(axis=1)])


def test_parameter_count_for_flowers_configuration():
    model = LinearSVMOvR()
    model.W, model.b = np.zeros((8100, 102)), np.zeros(102)
    assert model.n_parameters == 826_302 == 8100 * 102 + 102


def test_training_is_reproducible_with_seed():
    X, y = blobs()
    cfg = SVMConfig(n_epochs=5, batch_size=8, seed=7)
    a, b = LinearSVMOvR(cfg).fit(X, y), LinearSVMOvR(cfg).fit(X, y)
    np.testing.assert_array_equal(a.W, b.W)
    c = LinearSVMOvR(SVMConfig(n_epochs=5, batch_size=8, seed=8)).fit(X, y)
    assert not np.array_equal(a.W, c.W)


def test_objective_decreases():
    X, y = blobs()
    model = LinearSVMOvR(SVMConfig(learning_rate=0.01, n_epochs=20, batch_size=None)).fit(X, y)
    obj = model.history.objective
    assert len(obj) == 20
    assert obj[-1] < obj[0]
    # starting from W = 0, b = 0 every hinge term is 1, so the first objective is below 1 after one step
    assert obj[0] < 1.0


def test_relative_change_measures_convergence():
    from src.classical.linear_svm import TrainingHistory

    h = TrainingHistory(objective=[2.0, 1.5, 1.2, 1.0])
    assert h.relative_change(window=2) == pytest.approx((1.5 - 1.0) / 1.0)
    assert np.isnan(h.relative_change(window=4))      # needs window + 1 recorded epochs
    assert TrainingHistory(objective=[1.0] * 12).relative_change(10) == 0.0


def test_full_batch_gradient_matches_finite_differences():
    """The update rule must be the gradient of the documented objective (away from hinge kinks)."""
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(12, 4)), np.repeat([0, 1, 2], 4)
    cfg = SVMConfig(learning_rate=1.0, reg_strength=0.3, n_epochs=1, batch_size=None, lr_decay=0.0,
                    positive_weight=2.5)
    model = LinearSVMOvR(cfg)
    model.classes_, y_idx = np.unique(y, return_inverse=True)
    Y = model._signed_targets(y_idx)
    model.positive_weights_ = model._class_positive_weights(Y)
    S = np.where(Y > 0, model.positive_weights_, 1.0)
    W0, b0 = rng.normal(scale=0.3, size=(4, 3)), rng.normal(scale=0.3, size=3)

    def objective_sum(W, b):
        model.W, model.b = W, b
        return model._objective(X, Y, S) * 3  # _objective is the mean over 3 classes

    eps = 1e-6
    num_W = np.zeros_like(W0)
    for idx in np.ndindex(W0.shape):
        Wp, Wm = W0.copy(), W0.copy()
        Wp[idx] += eps
        Wm[idx] -= eps
        num_W[idx] = (objective_sum(Wp, b0) - objective_sum(Wm, b0)) / (2 * eps)
    num_b = np.array([(objective_sum(W0, b0 + eps * e) - objective_sum(W0, b0 - eps * e)) / (2 * eps)
                      for e in np.eye(3)])

    active = (Y * (X @ W0 + b0)) < 1
    coef = S * Y * active
    np.testing.assert_allclose(cfg.reg_strength * W0 - X.T @ coef / 12, num_W, atol=1e-5)
    np.testing.assert_allclose(-coef.sum(axis=0) / 12, num_b, atol=1e-5)


def test_bias_is_not_regularised():
    """With every sample on the positive side of the margin, only the weights shrink."""
    X = np.array([[1.0], [-1.0]])
    y = np.array([0, 1])
    model = LinearSVMOvR(SVMConfig(learning_rate=0.1, reg_strength=1.0, n_epochs=1, batch_size=None, lr_decay=0))
    model.fit(X, y)
    # one manual full-batch step from zero: all margins are 0 < 1, so all samples are active
    Y = np.array([[1.0, -1.0], [-1.0, 1.0]])
    expected_W = 0.1 * (X.T @ Y) / 2
    expected_b = 0.1 * Y.sum(axis=0) / 2
    np.testing.assert_allclose(model.W, expected_W)
    np.testing.assert_allclose(model.b, expected_b)   # = 0 here; regularisation would not touch it either way
    # Second check: with all-zero features W stays 0 and all margins stay < 1, so
    # every epoch adds the same hinge step to b. A regularised bias would be
    # pulled back by eta * lambda * b (= 10 * b here) in epoch 2; ours is not.
    X2, y2 = np.zeros((3, 1)), np.array([0, 0, 1])
    m2 = LinearSVMOvR(SVMConfig(learning_rate=0.1, reg_strength=100.0, n_epochs=2, batch_size=None,
                                lr_decay=0.0)).fit(X2, y2)
    one_step = np.array([0.1 * (1 + 1 - 1) / 3, 0.1 * (-1 - 1 + 1) / 3])
    np.testing.assert_allclose(m2.b, 2 * one_step)


def test_balanced_positive_weight_is_negatives_over_positives():
    X, y = blobs(n_per_class=10, n_classes=5)
    model = LinearSVMOvR(SVMConfig(n_epochs=1, positive_weight="balanced")).fit(X, y)
    np.testing.assert_allclose(model.positive_weights_, 40 / 10)
    model = LinearSVMOvR(SVMConfig(n_epochs=1, positive_weight=3.0)).fit(X, y)
    np.testing.assert_allclose(model.positive_weights_, 3.0)
    model = LinearSVMOvR(SVMConfig(n_epochs=1)).fit(X, y)
    np.testing.assert_allclose(model.positive_weights_, 1.0)


def test_positive_weight_scales_positive_hinge_terms():
    """Positive weight p multiplies the positive samples' gradient contribution by p."""
    X = np.array([[1.0], [2.0], [-1.0]])
    y = np.array([0, 1, 1])
    step = dict(learning_rate=1.0, reg_strength=0.0, n_epochs=1, batch_size=None, lr_decay=0.0)
    w1 = LinearSVMOvR(SVMConfig(**step)).fit(X, y).W[:, 0]
    w3 = LinearSVMOvR(SVMConfig(**step, positive_weight=3.0)).fit(X, y).W[:, 0]
    # class 0: positive x=1, negatives x=2, -1. Unweighted grad sum = 1 - 2 + 1 = 0 -> 0; weighted: 3 - 2 + 1 = 2
    np.testing.assert_allclose(w1, [0.0])
    np.testing.assert_allclose(w3, [2.0 / 3])


def test_non_contiguous_labels_are_mapped_back():
    X, y = blobs(n_classes=3)
    labels = np.array([5, 17, 101])[y]
    model = LinearSVMOvR(SVMConfig(learning_rate=0.05, n_epochs=20, batch_size=16)).fit(X, labels)
    assert set(model.predict(X)) <= {5, 17, 101}
    assert (model.predict(X) == labels).mean() == 1.0


@pytest.mark.parametrize("bad", [dict(learning_rate=0), dict(reg_strength=-1), dict(n_epochs=0),
                                 dict(batch_size=0), dict(positive_weight="auto"), dict(positive_weight=-2.0)])
def test_invalid_configuration_rejected(bad):
    with pytest.raises(ValueError):
        SVMConfig(**bad)


def test_rejects_nonfinite_features():
    X, y = blobs()
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        LinearSVMOvR().fit(X, y)


def test_save_and_load_round_trip(tmp_path):
    X, y = blobs()
    model = LinearSVMOvR(SVMConfig(n_epochs=3, positive_weight="balanced")).fit(X, y)
    model.save(tmp_path / "svm.npz")
    loaded = LinearSVMOvR.load(tmp_path / "svm.npz")
    np.testing.assert_array_equal(loaded.decision_function(X), model.decision_function(X))
    assert loaded.config == model.config
