"""
plotting/plot_perturbation.py
------------------------------
Visualisations for input-perturbation analyses.

Each function has a `multi_window` flag:
  - multi_window=False (default): results come from a single perturbation
    window, structured as a list of dicts with keys
    {'predictions', 'mean_times', 'slopes', 'performance'}.
  - multi_window=True: results come from multiple perturbation windows,
    structured as a list of dicts keyed by window start time, each
    containing the same inner keys.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.spatial.distance import pdist

from style import DURATION_CMAP, WINDOW_CMAP, MAGNITUDE_CMAP, style_axes


# Mean produced time vs. perturbation magnitude

def plot_mean_times_vs_magnitude(
    results,
    perturb_magnitudes,
    multi_window=False,
    perturb_starts=None,
    window_length=300,
    cmap_name=WINDOW_CMAP,
):
    """Plot mean produced time (Tp) ± SD across repeats vs. perturbation magnitude.

    Parameters
    ----------
    results : list of dict
        Output of generate_perturbation_results (length = n_repeats).
    perturb_magnitudes : list of float
        Perturbation magnitudes that were tested.
    multi_window : bool
        If True, plot one line per window defined by `perturb_starts`.
        If False, plot a single line for the single perturbation window.
    perturb_starts : list of int, optional
        Window onset times (ms). Required when multi_window=True.
    window_length : int
        Duration of each window in ms, used for legend labels.
    cmap_name : str
        Colormap name for multi-window mode.
    """
    mags = np.array(perturb_magnitudes)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)

    if multi_window:
        colors = plt.get_cmap(cmap_name)(np.linspace(0.3, 1.0, len(perturb_starts)))
        for color, start in zip(colors, perturb_starts):
            mu, sd = _mean_sd(results, start, "mean_times")
            ax.plot(mags, mu, "-o", markersize=10, color=color,
                    label=f"{start}–{start + window_length} ms")
            ax.fill_between(mags, mu - sd, mu + sd, alpha=0.5, color=color)
        ax.legend(fontsize=11)
    else:
        mu, sd = _mean_sd(results, None, "mean_times")
        ax.plot(mags, mu, "-", color="grey", zorder=1)
        ax.scatter(mags, mu, c=mags, cmap=MAGNITUDE_CMAP,
                   vmin=mags.min(), vmax=mags.max(),
                   s=80, edgecolor="black", zorder=2)
        ax.fill_between(mags, mu - sd, mu + sd, color="grey", alpha=0.3, zorder=0)

    ax.set_xticks(mags)
    ax.set_xlabel("Perturbation magnitude", fontsize=16)
    ax.set_ylabel("Mean $T_p$ (ms)", fontsize=16)
    style_axes(ax)
    fig.tight_layout()
    plt.show()


# Regression slope (Ts / Tp) vs. perturbation magnitude

def plot_slopes_vs_magnitude(
    results,
    perturb_magnitudes,
    multi_window=False,
    perturb_starts=None,
    window_length=300,
    cmap_name=WINDOW_CMAP,
):
    """Plot Ts/Tp regression slope ± SD across repeats vs. perturbation magnitude.

    Parameters
    ----------
    results : list of dict
        Output of generate_perturbation_results.
    perturb_magnitudes : list of float
        Perturbation magnitudes that were tested.
    multi_window : bool
        If True, plot one line per window defined by `perturb_starts`.
    perturb_starts : list of int, optional
        Window onset times (ms). Required when multi_window=True.
    window_length : int
        Duration of each window in ms, used for legend labels.
    cmap_name : str
        Colormap name for multi-window mode.
    """
    mags = np.array(perturb_magnitudes)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)

    if multi_window:
        colors = plt.get_cmap(cmap_name)(np.linspace(0.3, 1.0, len(perturb_starts)))
        for color, start in zip(colors, perturb_starts):
            mu, sd = _mean_sd(results, start, "slopes")
            ax.plot(mags, mu, "-o", markersize=10, color=color,
                    label=f"{start}–{start + window_length} ms")
            ax.fill_between(mags, mu - sd, mu + sd, alpha=0.4, color=color)
        ax.legend(fontsize=11)
    else:
        mu, sd = _mean_sd(results, None, "slopes")
        ax.plot(mags, mu, "-", color="grey", zorder=1)
        ax.scatter(mags, mu, c=mags, cmap=MAGNITUDE_CMAP,
                   vmin=mags.min(), vmax=mags.max(),
                   s=80, edgecolor="black", zorder=2)
        ax.fill_between(mags, mu - sd, mu + sd, color="grey", alpha=0.3, zorder=0)

    ax.set_xticks(mags)
    ax.set_xlabel("Perturbation magnitude", fontsize=16)
    ax.set_ylabel("$T_s$/$T_p$ slope", fontsize=16)
    style_axes(ax)
    fig.tight_layout()
    plt.show()


# Model performance vs. perturbation magnitude

def plot_performance_vs_magnitude(
    results,
    perturb_magnitudes,
    multi_window=False,
    perturb_starts=None,
    window_length=300,
    cmap_name=WINDOW_CMAP,
):
    """Plot mean model performance ± SD across repeats vs. perturbation magnitude.

    Parameters
    ----------
    results : list of dict
        Output of generate_perturbation_results.
    perturb_magnitudes : list of float
        Perturbation magnitudes that were tested.
    multi_window : bool
        If True, plot one line per window defined by `perturb_starts`.
    perturb_starts : list of int, optional
        Window onset times (ms). Required when multi_window=True.
    window_length : int
        Duration of each window in ms, used for legend labels.
    cmap_name : str
        Colormap name for multi-window mode.
    """
    mags = np.array(perturb_magnitudes)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)

    if multi_window:
        colors = plt.get_cmap(cmap_name)(np.linspace(0.3, 1.0, len(perturb_starts)))
        for color, start in zip(colors, perturb_starts):
            mu, sd = _mean_sd(results, start, "performance")
            ax.plot(mags, mu, "-o", markersize=10, color=color,
                    label=f"{start}–{start + window_length} ms")
            ax.fill_between(mags, mu - sd, mu + sd, alpha=0.5, color=color)
        ax.legend(fontsize=11)
    else:
        mu, sd = _mean_sd(results, None, "performance")
        ax.plot(mags, mu, "-o", color="indigo")
        ax.fill_between(mags, mu - sd, mu + sd, alpha=0.4, color="indigo")

    ax.set_xticks(mags)
    ax.set_xlabel("Perturbation magnitude", fontsize=16)
    ax.set_ylabel("Mean performance", fontsize=16)
    style_axes(ax)
    fig.tight_layout()
    plt.show()


# Produced time vs. target time scatter

def plot_scatter_produced_vs_target(
    results,
    perturb_magnitudes,
    targ_times,
    multi_window=False,
    perturb_starts=None,
    duration_cmap=DURATION_CMAP,
    magnitude_cmap=MAGNITUDE_CMAP,
    tick_values=None,
):
    """Scatter plot of mean produced time vs. target time, colored by trial duration.

    In single-window mode, all magnitudes are overlaid on one plot.
    In multi-window mode, one plot is produced per perturbation window.

    Parameters
    ----------
    results : list of dict
        Output of generate_perturbation_results.
    perturb_magnitudes : list of float
        Perturbation magnitudes that were tested.
    targ_times : array-like
        Target interval durations (ms), one per trial.
    multi_window : bool
        If True, produce one figure per window in `perturb_starts`.
    perturb_starts : list of int, optional
        Window onset times (ms). Required when multi_window=True.
    duration_cmap : Colormap
        Colormap encoding trial duration (applied to scatter points).
    magnitude_cmap : str
        Colormap name for magnitude-colored mean lines (multi_window only).
    tick_values : array-like, optional
        Explicit tick positions for both axes. Defaults to [450, 550, 720, 880].
    """
    targ_times = np.asarray(targ_times)
    mn, mx = targ_times.min(), targ_times.max()
    ticks = np.array(tick_values) if tick_values is not None else np.array([450, 550, 720, 880])

    if multi_window:
        mag_colors = plt.get_cmap(magnitude_cmap)(
            np.linspace(0.4, 1.0, len(perturb_magnitudes))
        )
        trial_colors = duration_cmap(np.linspace(0, 1, len(targ_times)))

        for start in perturb_starts:
            fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
            for mag_color, mag in zip(mag_colors, perturb_magnitudes):
                stack = np.vstack([r[start]["predictions"][mag] for r in results])
                mean_p, sd_p = stack.mean(axis=0), stack.std(axis=0, ddof=1)
                ax.plot(targ_times, mean_p, "-", lw=3.0, color=mag_color, zorder=1)
                ax.fill_between(targ_times, mean_p - sd_p, mean_p + sd_p,
                                color=mag_color, alpha=0.4)
                for j, t in enumerate(targ_times):
                    ax.scatter(t, mean_p[j], s=80, c=[trial_colors[j]],
                               edgecolors="k", linewidth=0.4, zorder=2)
            _finish_scatter(ax, mn, mx, ticks)
            fig.tight_layout()
            plt.show()
    else:
        fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
        for mag in perturb_magnitudes:
            stack = np.vstack([r["predictions"][mag] for r in results])
            mean_p, sd_p = stack.mean(axis=0), stack.std(axis=0, ddof=1)
            ax.plot(targ_times, mean_p, "-", lw=1.0, color="grey", zorder=0)
            ax.fill_between(targ_times, mean_p - sd_p, mean_p + sd_p,
                            color="grey", alpha=0.25, zorder=1)
            ax.scatter(targ_times, mean_p, c=targ_times, cmap=duration_cmap,
                       vmin=mn, vmax=mx, s=80, edgecolors="k", linewidth=0.5, zorder=2)
        _finish_scatter(ax, mn, mx, ticks)
        fig.tight_layout()
        plt.show()


# Trajectory displacement over time

def plot_displacement_over_time(
    hid_clean,
    hid_perturbed,
    pcs,
    n_components=3,
    time_axis=None,
    figsize=(6, 4),
    dpi=300,
    color_full="black",
    color_pca="gray",
):
    """Plot normalised neural trajectory displacement between clean and perturbed runs.

    Displacement is shown in both the full D-dimensional space and the leading
    PCA subspace, normalised by the trajectory diameter in each space.

    Parameters
    ----------
    hid_clean : np.ndarray, shape (T, B, D)
        Hidden states from unperturbed trials.
    hid_perturbed : np.ndarray, shape (T, B, D)
        Hidden states from perturbed trials.
    pcs : np.ndarray, shape (D, K), K >= n_components
        Principal component axes (columns), e.g. from sklearn PCA.
    n_components : int
        Number of PCs used to define the subspace.
    time_axis : array-like, optional
        Time labels for the x-axis. Defaults to sample indices.
    figsize : tuple
        Figure size in inches.
    dpi : int
        Figure resolution.
    color_full : str
        Line color for full-space displacement.
    color_pca : str
        Line color for PCA-space displacement.

    Returns
    -------
    norm_full : np.ndarray, shape (T, B)
        Per-trial, per-timepoint normalised displacement in full space.
    """
    T, B, D = hid_clean.shape
    if time_axis is None:
        time_axis = np.arange(T)

    # Displacement vectors
    delta = hid_clean - hid_perturbed                           # (T, B, D)
    dist_full = np.linalg.norm(delta, axis=2)                   # (T, B)

    delta_pca = (delta.reshape(-1, D) @ pcs[:, :n_components])  # (T*B, K)
    dist_pca = np.linalg.norm(delta_pca, axis=1).reshape(T, B)  # (T, B)

    # Normalise by trajectory diameter in each space
    diam_full = max(pdist(hid_clean[:, b, :]).max() for b in range(B))

    baseline_pca = (hid_clean.reshape(-1, D) @ pcs[:, :n_components]).reshape(T, B, n_components)
    diam_pca = max(pdist(baseline_pca[:, b, :]).max() for b in range(B))

    norm_full = dist_full / diam_full   # (T, B)
    norm_pca  = dist_pca  / diam_pca   # (T, B)

    mean_full = norm_full.mean(axis=1)  # (T,)
    mean_pca  = norm_pca.mean(axis=1)   # (T,)

    # Plot individual trials (faint) and trial-mean (bold)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(time_axis, norm_full, lw=3, color=color_full, alpha=0.2)
    ax.plot(time_axis, norm_pca,  lw=3, color=color_pca,  alpha=0.2)
    ax.plot(time_axis, mean_full, "-", lw=2.0, color=color_full, label="Full space")
    ax.plot(time_axis, mean_pca,  "-", lw=2.0, color=color_pca,  label="PCA space")

    ax.set_xlabel("Time (ms)", fontsize=14)
    ax.set_ylabel("Normalised displacement", fontsize=14)
    ax.legend(fontsize=11)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    style_axes(ax, fontsize_ticks=12)
    fig.tight_layout()
    plt.show()

    return norm_full



def _mean_sd(results, start, key):
    """Extract mean and SD of a metric across repeats.

    Parameters
    ----------
    results : list of dict
        Repeat-level results.
    start : int or None
        Window start key for multi-window results; None for single-window.
    key : str
        Metric name ('mean_times', 'slopes', or 'performance').

    Returns
    -------
    mu : np.ndarray
    sd : np.ndarray
    """
    if start is not None:
        data = np.array([r[start][key] for r in results])
    else:
        data = np.array([r[key] for r in results])
    return data.mean(axis=0), data.std(axis=0, ddof=1)


def _finish_scatter(ax, mn, mx, ticks):
    """Add identity line, axis labels, ticks and style to a scatter axes."""
    ax.plot([mn, mx], [mn, mx], "k--", lw=0.8)
    ax.set_xlabel("Target time $T_s$ (ms)", fontsize=16)
    ax.set_ylabel("Produced time $T_p$ (ms)", fontsize=16)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    style_axes(ax)