"""
plotting/style.py
-----------------
Shared visual style constants and helpers used across plotting modules.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


# Colormap used to encode trial duration (short = orange, long = blue).
DURATION_CMAP = LinearSegmentedColormap.from_list(
    "duration",
    [(1.0, 0.549, 0.0), (0.392, 0.584, 0.929)],
)

# Colormap used for multi-window plots (sequential green).
WINDOW_CMAP = "BuGn"

# Default colormap for magnitude-colored scatter points.
MAGNITUDE_CMAP = "Greys"


def _clipped_cmap(
    cmap_src: mpl.colors.Colormap,
    lo: float = 0.0,
    hi: float = 1.0,
    n: int = 256,
) -> mpl.colors.ListedColormap:
    """Return a colormap clipped to the [lo, hi] range of `cmap_src`.

    This is useful for avoiding very light or very dark extremes that can
    reduce contrast against a white background.

    Parameters
    ----------
    cmap_src : matplotlib.colors.Colormap
        Source colormap.
    lo : float, default=0.0
        Lower fraction of the colormap range to use.
    hi : float, default=1.0
        Upper fraction of the colormap range to use.
    n : int, default=256
        Number of colors in the output colormap.

    Returns
    -------
    matplotlib.colors.ListedColormap
        Clipped colormap.
    """
    colors = cmap_src(np.linspace(lo, hi, n))
    return mpl.colors.ListedColormap(colors)


_SPEED_BASE = LinearSegmentedColormap.from_list(
    "_speed_base",
    ["#ffffffff", "#30303086"],
    N=256,
)

CMAP_SPEED_BG = _clipped_cmap(_SPEED_BASE, lo=0.05)

CMAP_SPEED_LINE = _clipped_cmap(_SPEED_BASE, lo=0.20)

CMAP_SPEED = CMAP_SPEED_BG

CMAP_DELTA = plt.cm.RdBu_r

CMAP_ANGLE = plt.cm.cividis

# Named colors for representative trials. Unknown trials fall back to a gradient.
TRIAL_COLORS: dict[str, tuple[float, float, float]] = {
    "trial0_baseline": (1.0, 0.549, 0.0),       # orange
    "trial0_perturbed": (0.85, 0.25, 0.15),     # red
    "trial15_baseline": (0.392, 0.584, 0.929),  # blue
    "trial15_perturbed": (0.30, 0.30, 0.75),    # indigo
}

# Gradient fallback across trials.
CMAP_TRIALS = LinearSegmentedColormap.from_list(
    "trial_gradient",
    [TRIAL_COLORS["trial0_baseline"], TRIAL_COLORS["trial15_baseline"]],
)

def get_trial_color(trial_idx: int, condition: str) -> tuple[float, float, float]:
    """Return the RGB color for a (trial, condition) pair.

    Parameters
    ----------
    trial_idx : int
        Trial index.
    condition : str
        Typically 'baseline' or 'perturbed'.

    Returns
    -------
    tuple[float, float, float]
        RGB triplet. Falls back to `CMAP_TRIALS` for unknown trial indices.
    """
    key = f"trial{trial_idx}_{condition}"
    if key in TRIAL_COLORS:
        return TRIAL_COLORS[key]

    # Normalize unknown trial indices to the [0, 1] range expected by the cmap.
    t = np.clip(trial_idx / 15.0, 0.0, 1.0)
    return CMAP_TRIALS(t)[:3]


def style_axes(ax, fontsize_labels: int = 16, fontsize_ticks: int = 14) -> None:
    """Apply a consistent visual style to an axis.

    Removes the top/right spines and standardizes tick appearance.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to style.
    fontsize_labels : int, default=16
        Reference font size for axis labels (labels are applied externally).
    fontsize_ticks : int, default=14
        Font size for tick labels.
    """
    del fontsize_labels  # kept for API compatibility / external reference

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=fontsize_ticks,
        length=8,
        width=2,
        direction="out",
    )
    ax.tick_params(
        axis="both",
        which="minor",
        labelsize=fontsize_ticks - 2,
        length=4,
        width=1.5,
        direction="out",
    )