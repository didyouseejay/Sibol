"""Milestone 2b: feature recipes and validation-only ablation for the classical pipeline.

A :class:`Recipe` describes one variant of the classical pipeline:

=================  ==============================================================
``resize_mode``    ``"direct"`` (M2) or ``"center_crop"`` (aspect ratio preserved)
``sqrt``           element-wise square root of the histogram features before
                   standardisation ("power" / Hellinger normalisation)
``color``          append the HSV colour histogram (320 values) to HOG (8,100)
``color_weight``   multiplier applied to the standardised colour columns, so the
                   320 colour values are not swamped by the 8,100 HOG values
``flip``           horizontal-flip augmentation of the **training** split only
                   (done exactly by permuting cached features; val/test untouched)
=================  ==============================================================

:func:`build_design` turns cached features into SVM inputs. The standardiser
is fitted on the (possibly augmented) **training** matrix only.
:func:`svm_grid_search` trains every SVM configuration on train and scores it
on validation; the test split is never used here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Mapping, Optional, Sequence

import numpy as np

from . import color as color_mod
from . import hog as hog_mod
from .color import ColorConfig
from .hog import HOGConfig
from .linear_svm import LinearSVMOvR, SVMConfig
from .preprocessing import FeatureStandardizer


@dataclass(frozen=True)
class Recipe:
    """One variant of the classical feature pipeline."""

    name: str
    resize_mode: str = "direct"
    sqrt: bool = False
    color: bool = False
    color_weight: float = 1.0
    flip: bool = False

    def with_changes(self, name: str, **changes) -> "Recipe":
        return replace(self, name=name, **changes)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Design:
    """SVM inputs for one recipe."""

    X: dict           # split -> standardised float matrix
    y: dict           # split -> labels (train doubled when flip=True)
    scaler: FeatureStandardizer
    column_weights: np.ndarray
    n_features: int


def raw_blocks(recipe: Recipe, hog: np.ndarray, color: Optional[np.ndarray]) -> np.ndarray:
    """Concatenate the recipe's feature blocks (HOG, then colour) and apply sqrt if requested."""
    blocks = [hog]
    if recipe.color:
        if color is None:
            raise ValueError("recipe uses colour but no colour features were given")
        blocks.append(color)
    X = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
    if recipe.sqrt:
        if X.min() < 0:
            raise ValueError("sqrt normalisation expects non-negative histogram features")
        X = np.sqrt(X)
    return X


def flip_index(recipe: Recipe, hog_cfg: HOGConfig, color_cfg: ColorConfig, image_size: int) -> np.ndarray:
    """Column permutation that turns features of an image into features of its mirror image."""
    perm = hog_mod.flip_permutation(hog_cfg, (image_size, image_size))
    if recipe.color:
        perm = np.concatenate([perm, perm.size + color_mod.flip_permutation(color_cfg)])
    return perm


def build_design(
    recipe: Recipe,
    hog: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    color: Optional[Mapping[str, np.ndarray]] = None,
    hog_cfg: HOGConfig = HOGConfig(),
    color_cfg: ColorConfig = ColorConfig(),
    image_size: int = 128,
    splits: Sequence[str] = ("train", "val"),
) -> Design:
    """Build standardised SVM inputs for ``splits`` (must include ``"train"``).

    * Flip augmentation is applied to ``train`` only: the training matrix is
      ``[X; X[:, flip_index]]`` and the labels are repeated.
    * The standardiser is fitted on that training matrix only.
    * Colour columns are multiplied by ``recipe.color_weight`` after standardising.
    """
    if "train" not in splits:
        raise ValueError("splits must include 'train' (the standardiser is fitted on it)")
    raw = {s: raw_blocks(recipe, hog[s], None if color is None else color[s]) for s in splits}
    y = {s: np.asarray(labels[s]) for s in splits}
    if recipe.flip:
        perm = flip_index(recipe, hog_cfg, color_cfg, image_size)
        raw["train"] = np.concatenate([raw["train"], raw["train"][:, perm]])
        y["train"] = np.concatenate([y["train"], y["train"]])

    scaler = FeatureStandardizer().fit(raw["train"])
    n_features = raw["train"].shape[1]
    weights = np.ones(n_features)
    if recipe.color:
        weights[hog[splits[0]].shape[1]:] = recipe.color_weight
    X = {}
    for s in splits:
        dtype = np.float64 if s == "train" else np.float32
        X[s] = scaler.transform(raw[s], dtype=dtype)
        if recipe.color and recipe.color_weight != 1.0:
            X[s] *= weights.astype(dtype)
    return Design(X=X, y=y, scaler=scaler, column_weights=weights, n_features=n_features)


def transform_split(design: Design, recipe: Recipe, hog: np.ndarray, color: Optional[np.ndarray]) -> np.ndarray:
    """Apply a design's fitted standardiser and column weights to another split (e.g. test). No augmentation."""
    X = design.scaler.transform(raw_blocks(recipe, hog, color), dtype=np.float32)
    if recipe.color and recipe.color_weight != 1.0:
        X *= design.column_weights.astype(np.float32)
    return X


def svm_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    configs: Sequence[SVMConfig],
    n_classes: int,
    metric_fn,
    log=None,
) -> tuple[list[dict], dict, LinearSVMOvR]:
    """Train every configuration on train, score on validation, keep the best.

    Selection: highest validation macro-F1, ties to higher validation top-1,
    then to the earlier configuration.

    Args:
        metric_fn: ``metric_fn(y_true, y_pred, n_classes) -> dict`` with
            ``"macro_f1"`` and ``"top1_accuracy"`` (e.g. ``classification_metrics``).
        log: optional ``print``-like callable for one line per configuration.

    Returns:
        ``(rows, best_row, best_model)``.
    """
    rows, best = [], None
    for cfg in configs:
        model = LinearSVMOvR(cfg).fit(X_train, y_train)
        val_m = metric_fn(y_val, model.predict(X_val), n_classes)
        row = {"positive_weight": str(cfg.positive_weight), "learning_rate": cfg.learning_rate,
               "reg_strength": cfg.reg_strength, "n_epochs": cfg.n_epochs,
               "fit_seconds": model.history.fit_seconds,
               "objective_rel_change_last10": model.history.relative_change(10),
               "weight_norm": float(np.linalg.norm(model.W)),
               "train_top1_accuracy": float((model.predict(X_train) == y_train).mean()),
               **{f"val_{k}": v for k, v in val_m.items() if k != "n_samples"}}
        rows.append(row)
        key = (row["val_macro_f1"], row["val_top1_accuracy"])
        if best is None or key > best[0]:
            best = (key, row, model)
        if log is not None:
            log(f"  lr={cfg.learning_rate:<6g} reg={cfg.reg_strength:<6g} | fit {row['fit_seconds']:5.1f} s | "
                f"train acc {row['train_top1_accuracy']:.3f} | val acc {row['val_top1_accuracy']:.3f} | "
                f"val macro-F1 {row['val_macro_f1']:.4f}")
    return rows, best[1], best[2]
