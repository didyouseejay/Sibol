"""Multiclass one-vs-rest linear SVM, implemented from scratch in NumPy.

No ready-made SVM (``sklearn.svm``, ``LinearSVC``, ...) is used.

Model
-----
For ``C`` classes we learn ``C`` binary classifiers at once, stored as a
weight matrix ``W`` (``D x C``) and a bias vector ``b`` (``C``). Class ``c``
scores sample ``x`` with ``f_c(x) = x . W[:, c] + b[c]``. The prediction is
the class with the highest score.

Objective (one binary problem per class, all trained together)
--------------------------------------------------------------
With ``y_ic = +1`` if sample ``i`` belongs to class ``c`` and ``-1`` otherwise::

    J_c(w, b) = (lambda / 2) * ||w||^2
              + (1 / N) * sum_i  s_ic * max(0, 1 - y_ic * (w . x_i + b))

* The hinge loss is the standard soft-margin SVM loss.
* L2 regularisation is applied to ``w`` only; **the bias is not regularised**.
* ``s_ic`` is the sample weight: ``positive_weight_c`` for the positive
  (one) class and ``1`` for the negatives (the rest).
  - ``positive_weight=None`` -> 1 (unweighted).
  - a number ``p`` -> ``p`` for every class.
  - ``"balanced"`` -> ``n_negative_c / n_positive_c`` for each class, so the
    positives and negatives carry equal total weight (``101`` on the
    Flowers-102 training split: 1,010 negatives vs 10 positives).
  The weighting multiplies each sample's hinge term (the same formulation as
  a class weight scaling ``C`` in a standard SVM); the average stays over
  ``N``, so ``lambda`` keeps the same meaning in both settings.

Optimisation
------------
Mini-batch subgradient descent with seeded shuffling. For a batch ``B``::

    active_ic = 1  if y_ic * f_c(x_i) < 1  else 0
    grad_W    = lambda * W - (1/|B|) * X_B^T (S * Y * active)
    grad_b    =          - (1/|B|) * sum_i (S * Y * active)_i
    eta_t     = learning_rate / (1 + lr_decay * epoch)

The full training objective (mean over classes) is recorded after every epoch.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

PositiveWeight = Union[None, float, str]


@dataclass
class SVMConfig:
    """Hyperparameters of :class:`LinearSVMOvR`.

    Attributes:
        learning_rate: Initial step size ``eta_0``.
        reg_strength: L2 regularisation strength ``lambda`` (weights only).
        n_epochs: Passes over the training set.
        batch_size: Mini-batch size; ``None`` = full batch.
        lr_decay: Step-size decay per epoch, ``eta_t = eta_0 / (1 + lr_decay * t)``.
        positive_weight: ``None``, a positive number, or ``"balanced"``.
        seed: Seed for the shuffling order (makes training reproducible).
    """

    learning_rate: float = 0.01
    reg_strength: float = 1e-3
    n_epochs: int = 50
    batch_size: Optional[int] = 128
    lr_decay: float = 0.05
    positive_weight: PositiveWeight = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.reg_strength < 0:
            raise ValueError("reg_strength must be >= 0")
        if self.n_epochs < 1:
            raise ValueError("n_epochs must be >= 1")
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("batch_size must be >= 1 or None")
        if self.lr_decay < 0:
            raise ValueError("lr_decay must be >= 0")
        pw = self.positive_weight
        if isinstance(pw, str):
            if pw != "balanced":
                raise ValueError("positive_weight string must be 'balanced'")
        elif pw is not None and not pw > 0:
            raise ValueError("positive_weight must be None, 'balanced' or a positive number")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainingHistory:
    """Per-epoch record of training."""

    objective: list = field(default_factory=list)       # mean over classes of J_c
    learning_rate: list = field(default_factory=list)
    fit_seconds: float = 0.0

    def relative_change(self, window: int = 10) -> float:
        """Relative decrease of the objective over the last ``window`` epochs.

        ``(J[-window-1] - J[-1]) / |J[-1]|``. Close to 0 means training has
        levelled off; a large value means it was still improving (not converged).
        Returns NaN if fewer than ``window + 1`` epochs were recorded.
        """
        if len(self.objective) <= window:
            return float("nan")
        old, new = self.objective[-window - 1], self.objective[-1]
        return float((old - new) / max(abs(new), 1e-12))


class LinearSVMOvR:
    """One-vs-rest linear SVM with hinge loss and L2 regularisation (NumPy)."""

    def __init__(self, config: SVMConfig = SVMConfig()) -> None:
        self.config = config
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.positive_weights_: np.ndarray | None = None
        self.history = TrainingHistory()

    # ------------------------------------------------------------------ helpers
    @property
    def n_parameters(self) -> int:
        """Number of learned coefficients: ``D * C`` weights + ``C`` biases."""
        if self.W is None:
            raise RuntimeError("model is not fitted")
        return int(self.W.size + self.b.size)

    def _signed_targets(self, y_idx: np.ndarray) -> np.ndarray:
        """``(N, C)`` matrix with +1 for the true class and -1 elsewhere."""
        Y = -np.ones((y_idx.shape[0], self.classes_.shape[0]))
        Y[np.arange(y_idx.shape[0]), y_idx] = 1.0
        return Y

    def _class_positive_weights(self, Y: np.ndarray) -> np.ndarray:
        pw = self.config.positive_weight
        n_pos = (Y > 0).sum(axis=0).astype(np.float64)
        if pw is None:
            return np.ones(Y.shape[1])
        if pw == "balanced":
            return (Y.shape[0] - n_pos) / n_pos
        return np.full(Y.shape[1], float(pw))

    def _objective(self, X: np.ndarray, Y: np.ndarray, S: np.ndarray) -> float:
        """Mean over classes of the full training objective ``J_c``."""
        margins = Y * (X @ self.W + self.b)
        hinge = (S * np.maximum(0.0, 1.0 - margins)).mean(axis=0)
        reg = 0.5 * self.config.reg_strength * np.sum(self.W * self.W, axis=0)
        return float(np.mean(reg + hinge))

    # --------------------------------------------------------------------- fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVMOvR":
        """Train all one-vs-rest classifiers.

        Args:
            X: ``(N, D)`` features (standardised beforehand).
            y: ``(N,)`` integer labels. Every class present becomes one classifier.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or y.shape != (X.shape[0],):
            raise ValueError(f"bad shapes X {X.shape}, y {y.shape}")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values")

        cfg = self.config
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        if self.classes_.shape[0] < 2:
            raise ValueError("need at least two classes")
        n, d = X.shape
        c = self.classes_.shape[0]

        Y = self._signed_targets(y_idx)
        self.positive_weights_ = self._class_positive_weights(Y)
        S = np.where(Y > 0, self.positive_weights_, 1.0)

        self.W = np.zeros((d, c))
        self.b = np.zeros(c)
        self.history = TrainingHistory()
        rng = np.random.default_rng(cfg.seed)
        batch = n if cfg.batch_size is None else min(cfg.batch_size, n)

        t0 = time.perf_counter()
        for epoch in range(cfg.n_epochs):
            eta = cfg.learning_rate / (1.0 + cfg.lr_decay * epoch)
            order = rng.permutation(n)
            for start in range(0, n, batch):
                idx = order[start : start + batch]
                Xb, Yb, Sb = X[idx], Y[idx], S[idx]
                active = (Yb * (Xb @ self.W + self.b)) < 1.0
                coef = Sb * Yb * active                       # (B, C)
                grad_W = cfg.reg_strength * self.W - (Xb.T @ coef) / idx.shape[0]
                grad_b = -coef.sum(axis=0) / idx.shape[0]     # bias: no regularisation term
                self.W -= eta * grad_W
                self.b -= eta * grad_b
            self.history.objective.append(self._objective(X, Y, S))
            self.history.learning_rate.append(eta)
        self.history.fit_seconds = time.perf_counter() - t0
        return self

    # ----------------------------------------------------------------- predict
    def decision_function(self, X: np.ndarray, chunk_size: int = 2048) -> np.ndarray:
        """Scores of every class, shape ``(N, C)``. Computed in chunks to bound memory."""
        if self.W is None:
            raise RuntimeError("model is not fitted")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.W.shape[0]:
            raise ValueError(f"expected (N, {self.W.shape[0]}) features, got {X.shape}")
        out = np.empty((X.shape[0], self.W.shape[1]))
        for start in range(0, X.shape[0], chunk_size):
            out[start : start + chunk_size] = X[start : start + chunk_size].astype(np.float64) @ self.W + self.b
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Label of the highest-scoring class for each sample (ties: lowest class index)."""
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    # ----------------------------------------------------------------- persist
    def save(self, path: Path) -> None:
        if self.W is None:
            raise RuntimeError("model is not fitted")
        meta = {"config": self.config.to_dict(), "objective": self.history.objective,
                "fit_seconds": self.history.fit_seconds}
        np.savez(Path(path), W=self.W, b=self.b, classes=self.classes_,
                 positive_weights=self.positive_weights_, meta=np.array(json.dumps(meta)))

    @classmethod
    def load(cls, path: Path) -> "LinearSVMOvR":
        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            model = cls(SVMConfig(**meta["config"]))
            model.W, model.b, model.classes_ = data["W"], data["b"], data["classes"]
            model.positive_weights_ = data["positive_weights"]
        model.history.objective = meta["objective"]
        model.history.fit_seconds = meta["fit_seconds"]
        return model
