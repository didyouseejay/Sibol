"""Static figures for reports (Matplotlib, PNG).

All functions return the figure and optionally save it. Styling is kept
plain: one hue for single-series charts, a single-hue sequential map for the
confusion matrix, recessive grids, text in neutral ink.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

INK = "#262626"
MUTED = "#6b6b6b"
GRID = "#e3e3e3"
SERIES = ["#2f6db5", "#d9772b", "#3a9a5b", "#8a5cc2"]  # fixed order, never cycled past 4
SEQUENTIAL_CMAP = "Blues"


def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def _finish(fig, path: Optional[Path], dpi: int = 150):
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig


# ---------------------------------------------------------------------------
# HOG
# ---------------------------------------------------------------------------
def draw_hog_glyphs(ax, cell_hist: np.ndarray, cell_size: int) -> None:
    """Draw each cell's histogram as line segments perpendicular to the gradient (edge direction)."""
    cy, cx, n = cell_hist.shape
    scale = cell_hist.max() or 1.0
    centres = (np.arange(n) + 0.5) * 180.0 / n
    ax.set_facecolor("black")
    for i in range(cy):
        for j in range(cx):
            yc, xc = (i + 0.5) * cell_size, (j + 0.5) * cell_size
            for k, angle in enumerate(centres):
                w = cell_hist[i, j, k] / scale
                if w <= 0:
                    continue
                edge = np.radians(angle + 90.0)  # edges run perpendicular to the gradient
                dx, dy = np.cos(edge) * cell_size / 2, np.sin(edge) * cell_size / 2
                ax.plot([xc - dx, xc + dx], [yc - dy, yc + dy], color="white", alpha=float(w), linewidth=1.0)
    ax.set_xlim(0, cx * cell_size)
    ax.set_ylim(cy * cell_size, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_hog_demo(rgb: np.ndarray, gray: np.ndarray, magnitude: np.ndarray, cell_hist: np.ndarray,
                  cell_size: int, title: str = "", path: Optional[Path] = None):
    """Four panels: original, 128x128 grayscale, gradient magnitude, cell histograms."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    panels = [(rgb, None, "Original"), (gray, "gray", f"Resized grayscale {gray.shape[1]}x{gray.shape[0]}"),
              (magnitude, "magma", "Gradient magnitude")]
    for ax, (img, cmap, label) in zip(axes[:3], panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(label, color=INK, fontsize=11)
        ax.axis("off")
    draw_hog_glyphs(axes[3], cell_hist, cell_size)
    axes[3].set_title(f"Cell histograms ({cell_hist.shape[0]}x{cell_hist.shape[1]} cells)", color=INK, fontsize=11)
    if title:
        fig.suptitle(title, color=INK, fontsize=13)
    return _finish(fig, path)


# ---------------------------------------------------------------------------
# Training and tuning
# ---------------------------------------------------------------------------
def plot_training_curve(objective: Sequence[float], title: str = "Training objective", path: Optional[Path] = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = np.arange(1, len(objective) + 1)
    ax.plot(epochs, objective, color=SERIES[0], linewidth=2)
    ax.set_xlabel("epoch", color=INK)
    ax.set_ylabel("mean one-vs-rest objective", color=INK)
    ax.set_title(title, color=INK)
    _style(ax)
    return _finish(fig, path)


def plot_validation_grid(rows: Sequence[dict], path: Optional[Path] = None,
                         metric: str = "val_macro_f1", title: str = "Validation macro-F1"):
    """One panel per positive-weight setting; x = regularisation strength (log), one line per learning rate."""
    weights = sorted({str(r["positive_weight"]) for r in rows})
    lrs = sorted({r["learning_rate"] for r in rows})
    if len(lrs) > len(SERIES):
        raise ValueError("at most 4 learning rates can be shown")
    y_max = max(r[metric] for r in rows)
    fig, axes = plt.subplots(1, len(weights), figsize=(5.5 * len(weights), 4), sharey=True, squeeze=False)
    for ax, w in zip(axes[0], weights):
        for color, lr in zip(SERIES, lrs):
            pts = sorted((r["reg_strength"], r[metric]) for r in rows
                         if str(r["positive_weight"]) == w and r["learning_rate"] == lr)
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=7,
                    markeredgecolor="white", markeredgewidth=1.5, label=f"learning rate {lr:g}")
        ax.set_xscale("log")
        ax.set_xlabel("regularisation strength λ", color=INK)
        ax.set_title(f"positive weight: {w}", color=INK)
        ax.set_ylim(0, y_max * 1.15 if y_max > 0 else 1)
        _style(ax)
    axes[0][0].set_ylabel(metric.replace("_", " "), color=INK)
    axes[0][-1].legend(frameon=False, labelcolor=INK)
    fig.suptitle(title, color=INK)
    return _finish(fig, path)


def plot_ablation(steps: Sequence[dict], title: str = "Ablation: validation macro-F1",
                  path: Optional[Path] = None):
    """Horizontal bars, one per evaluated recipe, in evaluation order.

    Each step dict needs ``name``, ``val_macro_f1`` and ``kept`` (bool). Kept
    steps are drawn in the series colour, rejected ones in grey; the value is
    printed at the end of each bar.
    """
    names = [s["name"] for s in steps][::-1]
    values = [s["val_macro_f1"] for s in steps][::-1]
    colors = [SERIES[0] if s["kept"] else "#b5b5b5" for s in steps][::-1]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(steps) + 1.4))
    ax.barh(np.arange(len(steps)), values, color=colors, height=0.65)
    for y, v in enumerate(values):
        ax.text(v + max(values) * 0.01, y, f"{v:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(np.arange(len(steps)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, max(values) * 1.15)
    ax.set_xlabel("validation macro-F1 (best SVM configuration per recipe)", color=INK)
    ax.set_title(f"{title}\n(blue = kept in the pipeline, grey = not kept)", color=INK, fontsize=11)
    _style(ax)
    ax.grid(False, axis="y")
    return _finish(fig, path)


def plot_color_demo(rgb: np.ndarray, hist: np.ndarray, h_bins: int, s_bins: int,
                    title: str = "", path: Optional[Path] = None):
    """The resized image next to its global hue x saturation histogram (as a heat map)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.6]})
    axes[0].imshow(rgb)
    axes[0].set_title("Resized image", color=INK)
    axes[0].axis("off")
    grid = hist[: h_bins * s_bins].reshape(h_bins, s_bins).T
    im = axes[1].imshow(grid, cmap=SEQUENTIAL_CMAP, aspect="auto", origin="lower", vmin=0)
    axes[1].set_xlabel("hue bin (0 = red, ~5 = green, ~10 = blue)", color=INK)
    axes[1].set_ylabel("saturation bin (0 = grey)", color=INK)
    axes[1].set_yticks(np.arange(s_bins))
    axes[1].set_title("Global hue x saturation histogram", color=INK)
    axes[1].tick_params(colors=MUTED, labelcolor=INK)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04).set_label("share of pixels", color=INK)
    if title:
        fig.suptitle(title, color=INK)
    return _finish(fig, path)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion matrix (row-normalised)",
                          path: Optional[Path] = None, tick_every: int = 10):
    """Row-normalised confusion matrix: cell = share of the true class predicted as each class."""
    rows = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros(cm.shape, dtype=float), where=rows > 0)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1, interpolation="nearest")
    ticks = np.arange(0, cm.shape[0], tick_every)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("predicted class index", color=INK)
    ax.set_ylabel("true class index", color=INK)
    ax.set_title(title, color=INK)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("share of true-class images", color=INK)
    return _finish(fig, path)


def plot_per_class_f1(per_class: Sequence[dict], n_show: int = 15, title: str = "Per-class F1 (test)",
                      path: Optional[Path] = None):
    """Two panels: the ``n_show`` lowest- and highest-F1 classes, as horizontal bars."""
    ordered = sorted(per_class, key=lambda r: (r["f1"], r["class_id"]))
    groups = [("Lowest F1", ordered[:n_show]), ("Highest F1", ordered[-n_show:])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.32 * n_show + 1.8), sharex=True)
    for ax, (label, items) in zip(axes, groups):
        names = [f'{r["class_name"]} ({r["support"]})' for r in items]
        values = [r["f1"] for r in items]
        ax.barh(np.arange(len(items)), values, color=SERIES[0], height=0.7)
        ax.set_yticks(np.arange(len(items)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_title(label, color=INK)
        ax.set_xlabel("F1", color=INK)
        _style(ax)
        ax.grid(False, axis="y")
    fig.suptitle(f"{title}; number of test images in brackets", color=INK)
    return _finish(fig, path)
