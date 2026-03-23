"""
plotting/plot_trajectories.py
------------------------------
Visualisations for model outputs, hidden-state trajectories, PCA projections,
eigenvalue spectra, and stability analyses.

All functions follow the same style conventions as plot_perturbation.py and
import shared constants from style.py.
"""

from __future__ import annotations

from typing import List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from analysis.metrics import decode_time, compute_trajectory_speed
from plotting.style import DURATION_CMAP, style_axes

# Global rcParams
mpl.rcParams["xtick.major.width"] = 1
mpl.rcParams["ytick.major.width"] = 1
mpl.rcParams["xtick.major.size"]  = 6
mpl.rcParams["ytick.major.size"]  = 6


# Output and hidden-unit traces
def plot_output_targets(
    output: np.ndarray,
    target: np.ndarray,
    set_onset: int,
    t_start: int = 1000,
    t_end: int = 2700,
    cmap=DURATION_CMAP,
):
    """Plot model output traces alongside target traces for all trials.

    Parameters
    ----------
    output : np.ndarray, shape (T, B, 1)
    target : np.ndarray, shape (T, B, 1)
    set_onset : int
        Time index of the set signal (used to centre the x-axis).
    t_start : int
        First time index to display.
    t_end : int
        Last time index to display (exclusive).
    cmap : Colormap
        Colormap encoding trial duration.
    """
    if hasattr(output, "detach"):
        output = output.detach().cpu().numpy()
    if hasattr(target, "detach"):
        target = target.detach().cpu().numpy()

    output = output[t_start:t_end]
    target = target[t_start:t_end]
    T, B, D = output.shape
    assert D == 1, "Expected 1-dimensional output (D=1)."

    trial_colors = cmap(np.linspace(0, 1, B))
    time_ax = np.arange(T) - set_onset

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    for i in range(B):
        col = trial_colors[i]
        ax.plot(time_ax, output[:, i, 0], color=col, lw=1)
        ax.plot(time_ax, target[:, i, 0], color=col, lw=3, alpha=0.3)

    # Threshold and event markers
    ax.hlines(1.0, xmin=time_ax[0], xmax=time_ax[-1],
              colors="black", lw=0.5, linestyles="dashed")
    for t_event in [0, 500, 800]:
        ax.axvline(t_event, lw=0.5, color="black", linestyle="--")

    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks([0, 500, 800])
    ax.set_xlabel("Time (ms)", fontsize=16)

    # Grey axis styling consistent with the task aesthetic
    for element in [ax.xaxis.label, ax.yaxis.label]:
        element.set_color("grey")
    ax.tick_params(axis="both", colors="grey")
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("grey")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    return fig


def plot_hidden_units(
    hidden_acts: np.ndarray,
    unit_list: List[int],
    set_onset: int,
    hidden_acts_perturbed: Optional[np.ndarray] = None,
    cmap=DURATION_CMAP,
):
    """Plot the activity of selected hidden units across trials over time.

    If `hidden_acts_perturbed` is provided, the clean traces are shown in
    grey and the perturbed traces are overlaid in colour.

    Parameters
    ----------
    hidden_acts : np.ndarray, shape (T, B, H)
        Clean hidden-state activations.
    unit_list : list of int
        Indices of hidden units to plot (one figure per unit).
    set_onset : int
        Time index of the set signal (used to centre the x-axis).
    hidden_acts_perturbed : np.ndarray, shape (T, B, H), optional
        Perturbed hidden states. If provided, both clean and perturbed
        traces are drawn.
    cmap : Colormap
        Colormap encoding trial duration.
    """
    T, B, _ = hidden_acts.shape
    trial_colors = cmap(np.linspace(0, 1, B))
    time_ax = np.arange(T) - set_onset

    for idx in unit_list:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=300)

        for trial in range(B):
            color = trial_colors[trial]
            if hidden_acts_perturbed is None:
                ax.plot(time_ax, hidden_acts[:, trial, idx],
                        color=color, lw=0.75)
            else:
                ax.plot(time_ax, hidden_acts[:, trial, idx],
                        color="gray", lw=2, alpha=0.2)
                ax.plot(time_ax, hidden_acts_perturbed[:, trial, idx],
                        color=color, lw=1)

        for t_event in [0, 500, 800]:
            ax.axvline(t_event, lw=0.5, color="black", linestyle="--")

        ax.set_xlabel("Time (ms)", fontsize=16)
        ax.set_ylabel("Activity (a.u.)", fontsize=16)
        ax.set_xticks([0, 500, 800])

        for element in [ax.xaxis.label, ax.yaxis.label]:
            element.set_color("grey")
        ax.tick_params(axis="both", colors="grey")
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("grey")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        plt.show()


# PCA trajectory plots

def _run_pca(hidden_seq: np.ndarray, num_components: int):
    """Centre hidden_seq and return PCA axes, projections, and singular values.

    Parameters
    ----------
    hidden_seq : np.ndarray, shape (T*B, neurons) or (T, B, neurons)
        Will be flattened to 2D if needed.
    num_components : int

    Returns
    -------
    pcs : np.ndarray, shape (neurons, num_components)
    proj : np.ndarray, shape (T*B, num_components)
    mean_hidden : np.ndarray, shape (neurons,)
    S : np.ndarray  — singular values (for variance explained)
    """
    if hidden_seq.ndim == 3:
        T, B, neurons = hidden_seq.shape
        flat = hidden_seq.reshape(-1, neurons)
    else:
        flat = hidden_seq

    mean_hidden = flat.mean(axis=0)
    centered = flat - mean_hidden
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pcs = Vt[:num_components].T
    proj = centered @ pcs
    return pcs, proj, mean_hidden, S


def _style_3d_ax(ax):
    """Remove 3-D pane backgrounds, grids, and tick labels."""
    ax.tick_params(which="both", length=0,
                   labelbottom=False, labelleft=False,
                   labelright=False, labeltop=False)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def plot_pca_trajectories(
    hidden_seq: np.ndarray,
    num_components: int = 3,
    x_star: Optional[np.ndarray] = None,
    x_star_2: Optional[np.ndarray] = None,
    set_onset: Optional[int] = None,
    cmap=DURATION_CMAP,
    elev: float = -30,
    azim: float = 75,
):
    """Plot 3-D PCA trajectories of hidden states, coloured by trial duration.

    Also plots a companion scree-style cumulative-variance figure.

    Parameters
    ----------
    hidden_seq : np.ndarray, shape (T, B, H)
        Hidden-state sequence.
    num_components : int
        Number of PCs to project onto (must be ≤ 3 for 3-D plotting).
    x_star : np.ndarray, shape (N, H), optional
        First set of fixed points to overlay.
    x_star_2 : np.ndarray, shape (N, H), optional
        Second set of fixed points to overlay.
    set_onset : int, optional
        Time index after which to mark trial states (marker at onset + 200).
    cmap : Colormap
    elev : float
        Elevation angle for the 3-D view.
    azim : float
        Azimuth angle for the 3-D view.
    """
    T, B, neurons = hidden_seq.shape
    pcs, proj, mean_hidden, S = _run_pca(hidden_seq, num_components)
    proj_reshaped = proj.reshape(T, B, num_components)

    # Per-trial colours tiled over time
    trial_rgba = cmap(np.linspace(0, 1, B))
    point_colors = np.tile(trial_rgba, (T, 1))

    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2],
               c=point_colors, alpha=0.6, s=1, linewidths=0.05)

    if set_onset is not None and 0 <= set_onset + 200 < T:
        pts = proj_reshaped[set_onset + 200, :, :]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   marker="^", s=50, color="gray")

    for fp_arr in [x_star, x_star_2]:
        if fp_arr is not None:
            fp_proj = (fp_arr - mean_hidden) @ pcs
            ax.scatter(fp_proj[:, 0], fp_proj[:, 1], fp_proj[:, 2],
                       marker="X", s=100, color="k", label="Fixed points")

    _style_3d_ax(ax)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("PC1", fontsize=18)
    ax.set_ylabel("PC2", fontsize=18)
    ax.set_zlabel("PC3", fontsize=18)
    fig.tight_layout()
    plt.show()

    # Cumulative variance explained
    var_ratio = S**2 / np.sum(S**2)
    cum_var = np.cumsum(var_ratio)
    fig2, ax2 = plt.subplots(figsize=(3, 3), dpi=300)
    ax2.scatter(np.arange(1, num_components + 1), cum_var[:num_components],
                s=50, color="black")
    ax2.set_xlabel("Number of components", fontsize=12)
    ax2.set_ylabel("Cumulative explained variance", fontsize=12)
    ax2.set_ylim(0, 1.02)
    style_axes(ax2, fontsize_ticks=11)
    fig2.tight_layout()
    plt.show()

    return proj_reshaped, proj, pcs


def plot_pca_variable_lengths(
    trial_seqs: List[np.ndarray],
    num_components: int = 3,
    x_star: Optional[np.ndarray] = None,
    x_star_2: Optional[np.ndarray] = None,
    set_onset: Optional[int] = None,
    cmap=DURATION_CMAP,
    alpha: float = 0.6,
    s: float = 1,
    elev: float = 25,
    azim: float = 90,
):
    """Plot 3-D PCA trajectories for trials with variable lengths.

    PCA is fit on all trials jointly (stacked), then each trial is projected
    and plotted as a separate line.

    Parameters
    ----------
    trial_seqs : list of np.ndarray, each shape (T_i, H)
        Variable-length hidden-state sequences, one per trial.
    num_components : int
    x_star : np.ndarray, shape (N, H), optional
    x_star_2 : np.ndarray, shape (N, H), optional
    set_onset : int, optional
    cmap : Colormap
    alpha : float
    s : float
        Marker size for scatter.
    elev : float
    azim : float

    Returns
    -------
    proj_by_trial : list of np.ndarray, each shape (T_i, num_components)
    pcs : np.ndarray, shape (H, num_components)
    """
    n_trials = len(trial_seqs)
    lengths = [seq.shape[0] for seq in trial_seqs]

    flat = np.vstack(trial_seqs)
    pcs, proj_flat, mean_hidden, _ = _run_pca(flat, num_components)

    splits = np.cumsum(lengths)[:-1]
    proj_by_trial = np.split(proj_flat, splits, axis=0)
    trial_rgba = cmap(np.linspace(0, 1, n_trials))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1.5, 1, 1])

    for i, traj in enumerate(proj_by_trial):
        xs, ys = traj[:, 0], traj[:, 1] if num_components > 1 else np.zeros(len(traj))
        zs = traj[:, 2] if num_components > 2 else np.zeros(len(traj))
        ax.plot(xs, ys, zs, color=trial_rgba[i], alpha=alpha, linewidth=1)
        ax.scatter(xs, ys, zs, color=trial_rgba[i], s=s, alpha=alpha)

    if set_onset is not None:
        for i, traj in enumerate(proj_by_trial):
            if 0 <= set_onset < traj.shape[0]:
                pt = traj[set_onset]
                ax.scatter(pt[0],
                           pt[1] if num_components > 1 else 0,
                           pt[2] if num_components > 2 else 0,
                           marker="^", s=50, color="gray")

    for fp_arr in [x_star, x_star_2]:
        if fp_arr is not None:
            fp_proj = (fp_arr - mean_hidden) @ pcs
            ax.scatter(fp_proj[:, 0], fp_proj[:, 1], fp_proj[:, 2],
                       marker="X", s=100, color="k", label="Fixed points")

    ax.tick_params(colors="black", labelsize=12, width=1, length=4)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    fig.tight_layout()
    plt.show()

    return proj_by_trial, pcs


def plot_overlayed_pca(
    baseline_hidden: np.ndarray,
    perturbed_hidden: np.ndarray,
    num_components: int = 3,
    cmap=DURATION_CMAP,
    elev: float = 160,
    azim: float = -90,
):
    """Overlay clean and perturbed PCA trajectories in a shared PC space.

    PCA axes are fit on the baseline data; both datasets are then projected
    onto those axes. Baseline points are shown in grey; perturbed points
    are coloured by trial duration.

    Parameters
    ----------
    baseline_hidden : np.ndarray, shape (T, B, H)
    perturbed_hidden : np.ndarray, shape (T, B, H)
    num_components : int
    cmap : Colormap
    elev : float
    azim : float

    Returns
    -------
    PB : np.ndarray, shape (T, B, num_components) — baseline projections
    PP : np.ndarray, shape (T, B, num_components) — perturbed projections
    """
    assert baseline_hidden.shape == perturbed_hidden.shape, \
        "baseline_hidden and perturbed_hidden must have the same shape."

    T, B, H = baseline_hidden.shape

    flat_base = baseline_hidden.reshape(-1, H)
    flat_pert = perturbed_hidden.reshape(-1, H)

    mean_base = flat_base.mean(axis=0)
    pcs, proj_base, _, _ = _run_pca(flat_base, num_components)
    proj_pert = (flat_pert - mean_base) @ pcs

    PB = proj_base.reshape(T, B, num_components)
    PP = proj_pert.reshape(T, B, num_components)

    trial_rgba = cmap(np.linspace(0, 1, B))
    pert_colors = np.tile(trial_rgba, (T, 1))  # (T*B, 4)

    fig = plt.figure(figsize=(10, 7), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])

    ax.scatter(*[PB[:, :, k].reshape(-1) for k in range(3)],
               color="gray", alpha=0.1, s=1, zorder=1, label="Baseline")
    ax.scatter(*[PP[:, :, k].reshape(-1) for k in range(3)],
               c=pert_colors, alpha=0.8, s=2, zorder=2, label="Perturbed")

    _style_3d_ax(ax)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("PC1", fontsize=18)
    ax.set_ylabel("PC2", fontsize=18)
    ax.set_zlabel("PC3", fontsize=18)
    fig.tight_layout()
    plt.show()

    return PB, PP


# Behavioural readouts


def plot_produced_vs_target(
    output: np.ndarray,
    target: np.ndarray,
    set_onset: int,
    threshold: float = 0.95,
    tick_values: Optional[List[int]] = None,
    cmap=DURATION_CMAP,
):
    """Scatter plot of produced time Tp vs target time Ts, one point per trial.

    Parameters
    ----------
    output : np.ndarray, shape (T, B, D)
    target : np.ndarray, shape (T, B, D)
    set_onset : int
    threshold : float
    tick_values : list of int, optional
        Explicit axis tick positions. Defaults to [450, 550, 720, 880].
    cmap : Colormap

    Returns
    -------
    Tp : np.ndarray, shape (B,)
    Ts : np.ndarray, shape (B,)
    """
    Tp = np.array([t if t is not None else -1
                   for t in decode_time(output, set_onset, threshold)])
    Ts = np.array([t if t is not None else -1
                   for t in decode_time(target, set_onset, threshold)])

    mask = (Tp >= 0) & (Ts >= 0)
    if mask.sum() < 2:
        raise ValueError("Need at least two valid trials for plotting.")

    n_trials = len(Tp)
    trial_colors = cmap(np.linspace(0, 1, n_trials))
    ticks = np.array(tick_values if tick_values is not None else [450, 550, 720, 880])

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    ax.scatter(Ts, Tp, c=trial_colors, alpha=1, s=80,
               edgecolor="k", linewidth=0.5, zorder=2)

    all_min = min(Ts.min(), Tp.min())
    all_max = max(Ts.max(), Tp.max())
    ax.plot([all_min, all_max], [all_min, all_max],
            lw=0.5, color="black", zorder=1, label="Identity: $T_p=T_s$")

    ax.set_xlabel("Target time $T_s$ (ms)", fontsize=16)
    ax.set_ylabel("Produced time $T_p$ (ms)", fontsize=16)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    style_axes(ax)
    fig.tight_layout()
    plt.show()

    return Tp, Ts


def plot_speed_vs_produced_time(
    pca_trajectories: np.ndarray,
    produced_times: np.ndarray,
    dt: float = 1.0,
    tick_values: Optional[List[int]] = None,
    cmap=DURATION_CMAP,
):
    """Scatter normalised average trajectory speed vs. produced interval.

    Parameters
    ----------
    pca_trajectories : np.ndarray, shape (T, B, K)
        PCA-projected hidden states.
    produced_times : np.ndarray, shape (B,)
        Produced interval durations (ms).
    dt : float
        Time step size in ms.
    tick_values : list of int, optional
        Explicit x-axis tick positions. Defaults to [450, 550, 720, 880].
    cmap : Colormap
    """
    norm_speed = compute_trajectory_speed(pca_trajectories, dt=dt)
    B = len(produced_times)
    trial_colors = [cmap(i / B) for i in range(B)]
    ticks = np.array(tick_values if tick_values is not None else [450, 550, 720, 880])

    coeffs = np.polyfit(produced_times, norm_speed, 1)
    x_fit = np.linspace(produced_times.min(), produced_times.max(), 100)
    y_fit = np.polyval(coeffs, x_fit)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    for i in range(B):
        ax.scatter(produced_times[i], norm_speed[i],
                   s=80, edgecolor="k", color=trial_colors[i], zorder=2)
    ax.plot(x_fit, y_fit, color="black", linewidth=0.5, zorder=1)

    ax.set_xlabel("Produced interval (ms)", fontsize=16)
    ax.set_ylabel("Normalised average speed", fontsize=16)
    ax.set_xticks(ticks)
    style_axes(ax)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Eigenvalue and stability plots
# ---------------------------------------------------------------------------

def plot_eigenvalue_spectra(
    eigs: List[np.ndarray],
    cmap=DURATION_CMAP,
):
    """Scatter eigenvalues of per-fixed-point Jacobians in the complex plane.

    Parameters
    ----------
    eigs : list of np.ndarray, each shape (H,)
        Eigenvalues for each fixed point, e.g. from
        `CustomFixedPointFinder.compute_jacobian`.
    cmap : Colormap
    """
    n_fps = len(eigs)
    fp_colors = cmap(np.linspace(0, 1, n_fps))

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    for idx, eigvals in enumerate(eigs):
        ax.scatter(eigvals.real, eigvals.imag,
                   color=fp_colors[idx], alpha=0.7, s=10)

    ax.set_xlabel("Real", fontsize=14)
    ax.set_ylabel("Imaginary", fontsize=14)
    ax.set_aspect("equal", adjustable="box")
    style_axes(ax)
    fig.tight_layout()
    plt.show()


def plot_leading_eigenvalue_heatmap(
    leading: np.ndarray,
    cmap: str = "PRGn",
    alpha_label: str = r"$\lambda$",
):
    """Heatmap of the leading eigenvalue across trials and time.

    Parameters
    ----------
    leading : np.ndarray, shape (B, T)
        Leading eigenvalue at each trial and timestep.
    cmap : str
    alpha_label : str
        Colorbar label.
    """
    n_trials, n_times = leading.shape

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    im = ax.imshow(leading, aspect="auto", origin="upper", cmap=cmap)
    ax.set_yticks(np.arange(-0.5, n_trials, 1), minor=True)
    ax.grid(which="minor", axis="y", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("Time step", fontsize=14)
    ax.set_ylabel("Trial index", fontsize=14)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(alpha_label, rotation=270, labelpad=15)
    fig.tight_layout()
    plt.show()


def plot_delta_lambda(
    x: np.ndarray,
    x_pert: np.ndarray,
    times: Optional[np.ndarray] = None,
    t_mark: Optional[float] = None,
    flip_threshold: float = 1.0,
    vclip: float = 99.0,
    figsize: tuple = (8, 4.5),
):
    """Two-panel plot of Δλ = λ_perturbed − λ_original across trials and time.

    The top panel shows a heatmap of Δλ; the bottom panel shows the
    trial-mean ± SEM over time, with a secondary axis showing the
    fraction of trials that cross the stability boundary (λ = 1).

    Parameters
    ----------
    x : np.ndarray, shape (B, T)
        Leading eigenvalues from the unperturbed model.
    x_pert : np.ndarray, shape (B, T)
        Leading eigenvalues from the perturbed model.
    times : np.ndarray, shape (T,), optional
        Time labels for the x-axis. Defaults to sample indices.
    t_mark : float, optional
        Time value at which to draw a vertical dashed marker.
    flip_threshold : float
        λ value defining the stability boundary.
    vclip : float
        Percentile for robust symmetric colour-limit clipping.
    figsize : tuple

    Returns
    -------
    fig : matplotlib.figure.Figure
    Δ : np.ndarray — raw difference array
    mean_Δ : np.ndarray — trial-mean Δλ over time
    sem_Δ : np.ndarray — standard error of mean
    flip_rate : np.ndarray — fraction of trials crossing the boundary per timestep
    """
    assert x.shape == x_pert.shape, "x and x_pert must have the same shape."
    delta = x_pert - x

    finite = np.isfinite(delta)
    if not np.any(finite):
        raise ValueError("Δλ contains no finite values.")

    vmax = np.percentile(np.abs(delta[finite]), vclip)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    n_trials, n_times = delta.shape
    if times is None:
        times = np.arange(n_times)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.2], hspace=0.35)

    # Panel A — Δλ heatmap
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(delta, aspect="auto", interpolation="nearest",
                    origin="upper", norm=norm)
    tick_idx = np.linspace(0, n_times - 1, min(10, n_times)).astype(int)
    ax0.set_xticks(tick_idx)
    ax0.set_xticklabels([f"{times[i]:g}" for i in tick_idx])
    ax0.set_ylabel("Trial #", fontsize=12)
    ax0.set_title("Δλ (perturbed − original)", fontsize=12)
    cbar = fig.colorbar(im, ax=ax0)
    cbar.set_label("Δλ", rotation=270, labelpad=15)

    # Optional vertical marker
    t_mark_idx = None
    if t_mark is not None:
        t_mark_idx = int(np.argmin(np.abs(times - t_mark)))
        ax0.axvline(t_mark_idx, linestyle="--", linewidth=1)

    # Panel B — mean Δλ ± SEM and stability flip rate
    mean_delta = np.nanmean(delta, axis=0)
    n_valid = np.sum(np.isfinite(delta), axis=0)
    sem_delta = np.nanstd(delta, axis=0) / np.sqrt(np.maximum(n_valid, 1))

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(times, mean_delta, linewidth=2)
    ax1.fill_between(times, mean_delta - sem_delta, mean_delta + sem_delta,
                     alpha=0.25, linewidth=0)
    ax1.axhline(0, linewidth=1)
    if t_mark_idx is not None:
        ax1.axvline(times[t_mark_idx], linestyle="--", linewidth=1)
    ax1.set_xlabel("Time", fontsize=12)
    ax1.set_ylabel("Mean Δλ", fontsize=12)

    with np.errstate(invalid="ignore"):
        flips = ((x < flip_threshold) != (x_pert < flip_threshold)) \
                & np.isfinite(x) & np.isfinite(x_pert)
    flip_rate = flips.sum(axis=0) / np.maximum(1, n_valid)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(times, 100 * flip_rate, linestyle=":", linewidth=1.5)
    ax1_twin.set_ylabel("% stability flips", rotation=270, labelpad=15, fontsize=11)

    fig.tight_layout()
    return fig, (delta, mean_delta, sem_delta, flip_rate)