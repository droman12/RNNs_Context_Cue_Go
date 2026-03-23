"""
plotting/plot_flow_field.py
----------------------------
Flow field visualisations for RNN perturbation experiments.

Each public function produces one figure and saves it as both .png and .pdf.
All computation (velocity fields, PCA, phase binning) is delegated to
analysis/flow_field.py; this module only handles rendering.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from analysis.flow_field import compute_velocity_field, compute_field_delta
from plotting.style import (
    CMAP_ANGLE, CMAP_DELTA, CMAP_SPEED, CMAP_SPEED_BG, CMAP_SPEED_LINE,
    get_trial_color,
)


# ---------------------------------------------------------------------------
# Private rendering helpers
# ---------------------------------------------------------------------------

def _lighten(color, amount: float = 0.55) -> tuple:
    """Blend `color` toward white by `amount` (0 = original, 1 = white).

    Parameters
    ----------
    color : color-like
    amount : float

    Returns
    -------
    tuple of float — lightened RGB
    """
    rgb = np.array(mpl.colors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * amount)


def _draw_trajectory(ax, traj, b1, b2, color_active, color_ghost=None, lw_active=3.5):
    """Draw a trajectory with a highlighted active-phase segment.

    The full trajectory is drawn as a faint dashed ghost; the segment
    [b1:b2] is overlaid as a solid line with a white halo for legibility
    on dark backgrounds.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    traj : np.ndarray, shape (T, 2)
    b1, b2 : int
        Start and end indices of the active phase segment.
    color_active : color-like
        Colour for the active segment and markers.
    color_ghost : color-like, optional
        Colour for the full ghost path. Defaults to a lightened version of
        `color_active` to preserve hue identity.
    lw_active : float
        Line width for the active segment.
    """
    if color_ghost is None:
        color_ghost = _lighten(color_active, amount=0.5)

    stroke = lambda w: [pe.withStroke(linewidth=w, foreground="black")]

    ax.plot(traj[:, 0], traj[:, 1],
            ls="--", lw=2, color=color_ghost, alpha=1, zorder=7)
    ax.plot(traj[b1:b2, 0], traj[b1:b2, 1],
            ls="-", lw=lw_active, path_effects=stroke(w=4.0),
            color=color_active, zorder=8)
    ax.scatter(traj[b1, 0],     traj[b1, 1],
               s=45, color=color_active, zorder=10)
    ax.scatter(traj[b2 - 1, 0], traj[b2 - 1, 1],
               s=45, facecolor=color_active, edgecolor="black",
               linewidths=1.0, zorder=10)


def _draw_speed_field(ax, X1, X2, U, V, S, norm):
    """Render a speed field with a contourf background and streamplot overlay.

    The background uses CMAP_SPEED_BG (white at zero speed, so empty regions
    stay white). Streamlines use CMAP_SPEED_LINE, which is clipped to never
    reach white, guaranteeing contrast at all speed values.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    X1, X2 : np.ndarray, shape (n, n) — grid coordinates
    U, V : np.ndarray, shape (n, n) — velocity components
    S : np.ndarray, shape (n, n) — speed (may contain NaNs for masked regions)
    norm : matplotlib.colors.Normalize
    """
    S_safe = np.nan_to_num(S)
    U_safe = np.nan_to_num(U)
    V_safe = np.nan_to_num(V)

    ax.contourf(X1, X2, S_safe, levels=14, norm=norm,
                cmap=CMAP_SPEED_BG, alpha=0.80, zorder=1)

    lw_field = np.clip(0.6 + 2.0 * (S_safe / (norm.vmax + 1e-12)), 0.6, 2.6)
    ax.streamplot(
        X1, X2, U_safe, V_safe,
        color=S_safe,
        cmap=CMAP_SPEED_LINE,
        norm=norm,
        linewidth=lw_field,
        density=0.75,
        arrowsize=1.1,
        minlength=0.25,
        zorder=3,
    )


def _style_flow_ax(ax, bounds, add_pc_labels: bool = False):
    """Apply publication-style formatting to a flow field axes.

    Sets equal aspect ratio, tight limits from `bounds`, hides ticks, and
    optionally adds PC axis labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    bounds : ((x1min, x1max), (x2min, x2max))
    add_pc_labels : bool
        If True, label the axes 'PC 1' / 'PC 2'. Use only for the outermost panel.
    """
    (x1min, x1max), (x2min, x2max) = bounds
    ax.set_aspect("equal")
    ax.set_xlim(x1min, x1max)
    ax.set_ylim(x2min, x2max)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    if add_pc_labels:
        ax.set_xlabel("PC 1", fontsize=7, labelpad=2)
        ax.set_ylabel("PC 2", fontsize=7, labelpad=2)


def _add_speed_colorbar(fig, axes_subset, norm, cmap):
    """Attach a compact speed colorbar to the right of the given axes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    axes_subset : list of matplotlib.axes.Axes
    norm : matplotlib.colors.Normalize
    cmap : matplotlib Colormap
    """
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes_subset, shrink=0.7, pad=0.02,
                        aspect=20, fraction=0.046)
    cbar.ax.set_title("speed", fontsize=7, pad=4)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_visible(False)


def _global_speed_norm(model, pca, bounds, x_phase_trial, device, n_phases):
    """Compute a shared speed Normalize across all phases for one trial.

    Using a shared normalisation ensures that speed is comparable across
    phase panels within a figure.

    Parameters
    ----------
    model : Run_Model
    pca : sklearn.decomposition.PCA
    bounds : tuple
    x_phase_trial : np.ndarray, shape (n_phases, input_dim)
        Phase-averaged inputs for one trial.
    device : torch.device
    n_phases : int

    Returns
    -------
    norm : matplotlib.colors.Normalize
    thr : float — low-speed masking threshold (5% of vmax)
    """
    peak_speeds = []
    for k in range(n_phases):
        _, _, _, _, S = compute_velocity_field(model, pca, bounds, x_phase_trial[k], device)
        peak_speeds.append(np.nanpercentile(S, 99))
    vmax = float(np.nanmax(peak_speeds))
    return mpl.colors.Normalize(vmin=0, vmax=vmax), 0.05 * vmax


# ---------------------------------------------------------------------------
# Public figure functions
# ---------------------------------------------------------------------------

def plot_baseline_vs_perturbed(
    trial_idx: int,
    model,
    pca,
    bounds: tuple,
    x_phase_base: np.ndarray,
    x_phase_pert: np.ndarray,
    hid,
    hid_pert,
    phase_meta: list,
    fp_project: np.ndarray,
    fixed_point_indices: np.ndarray,
    device,
    t_end: int = 2700,
    save: bool = True,
):
    """Multi-phase comparison of baseline vs. perturbed flow fields for one trial.

    Produces a (K_phases × 4) figure with columns:
      0 — baseline field + baseline trajectory
      1 — perturbed field + both trajectories
      2 — Δspeed heatmap (diverging)
      3 — angle change heatmap (sequential)

    Parameters
    ----------
    trial_idx : int
        Index of the trial to visualise.
    model : Run_Model
    pca : sklearn.decomposition.PCA
    bounds : tuple
    x_phase_base : np.ndarray, shape (B, K, input_dim)
        Phase-averaged baseline inputs, indexed [trial, phase].
    x_phase_pert : np.ndarray, shape (B, K, input_dim)
        Phase-averaged perturbed inputs.
    hid : array-like, shape (T, B, H)
        Baseline hidden states.
    hid_pert : array-like, shape (T, B, H)
        Perturbed hidden states.
    phase_meta : list of (int, int)
        Phase time boundaries [(b1, b2), ...].
    fp_project : np.ndarray, shape (nTfix, B, 2)
        Fixed points projected into PC space.
    fixed_point_indices : np.ndarray of int
        Indices into `fp_project` selecting which fixed points to display.
    device : torch.device
    t_end : int
        Last time index to include in trajectory projection.
    save : bool
        If True, save as .png and .pdf.
    """
    from analysis.flow_field import project_trajectory, filter_points_near_trajectory

    K = len(phase_meta)
    col_base = get_trial_color(trial_idx, "baseline")
    col_pert = get_trial_color(trial_idx, "perturbed")

    norm, thr = _global_speed_norm(
        model, pca, bounds, x_phase_base[trial_idx], device, K
    )

    traj_base = project_trajectory(pca, hid[:t_end],      batch_idx=trial_idx)
    traj_pert = project_trajectory(pca, hid_pert[:t_end], batch_idx=trial_idx)

    pts_raw = fp_project[fixed_point_indices, trial_idx, :]
    mask, _ = filter_points_near_trajectory(pts_raw, traj_base, threshold=0.5)

    fig, axes = plt.subplots(K, 4, figsize=(11.2, 2.6 * K),
                              dpi=300, constrained_layout=True)
    axes = np.atleast_2d(axes)

    col_labels = [f"Trial {trial_idx}  baseline", "Perturbed", "Δ speed", "Angle (°)"]
    for j, label in enumerate(col_labels):
        axes[0, j].set_title(label, fontsize=8, fontweight="bold", pad=5)

    for k in range(K):
        b1, b2 = phase_meta[k]
        X1, X2, U0, V0, S0, U1, V1, S1, _, _, dS, ang = compute_field_delta(
            model, pca, bounds,
            x_phase_base[trial_idx, k],
            x_phase_pert[trial_idx, k],
            device,
        )
        S0_m = np.where(S0 >= thr, S0, np.nan)
        S1_m = np.where(S1 >= thr, S1, np.nan)

        axes[k, 0].set_ylabel(f"Phase {k}", fontsize=7, rotation=0,
                               labelpad=28, va="center")

        # Baseline field
        ax = axes[k, 0]
        _draw_speed_field(ax, X1, X2, U0, V0, S0_m, norm)
        _draw_trajectory(ax, traj_base, b1, b2, color_active=col_base)

        # Perturbed field with both trajectories
        ax = axes[k, 1]
        _draw_speed_field(ax, X1, X2, U1, V1, S1_m, norm)
        _draw_trajectory(ax, traj_base, b1, b2, color_active=col_base)
        _draw_trajectory(ax, traj_pert, b1, b2, color_active=col_pert)

        # Δspeed
        ax = axes[k, 2]
        v = np.nanmax(np.abs(dS)) or 1.0
        ax.contourf(X1, X2, dS, levels=14, cmap=CMAP_DELTA, vmin=-v, vmax=v, zorder=1)
        for traj, col in [(traj_base, col_base), (traj_pert, col_pert)]:
            ax.plot(traj[:, 0], traj[:, 1], ls="--", lw=0.9, color=col, alpha=0.6, zorder=7)

        # Angle change
        ax = axes[k, 3]
        ax.contourf(X1, X2, ang * 180 / np.pi, levels=14, cmap=CMAP_ANGLE, zorder=1)
        for traj, col in [(traj_base, col_base), (traj_pert, col_pert)]:
            ax.plot(traj[:, 0], traj[:, 1], ls="--", lw=0.9, color=col, alpha=0.6, zorder=7)

        for j in range(4):
            _style_flow_ax(axes[k, j], bounds, add_pc_labels=False)

    _add_speed_colorbar(fig, axes[:, :2].ravel().tolist(), norm, CMAP_SPEED)

    if save:
        fname = f"flow_compare_trial{trial_idx}"
        fig.savefig(f"{fname}.png", bbox_inches="tight")
        fig.savefig(f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_trial_comparison(
    trial_a: int,
    trial_b: int,
    model,
    pca,
    bounds: tuple,
    x_phase_base: np.ndarray,
    hid,
    phase_meta: list,
    fp_project: np.ndarray,
    fp_indices_a: np.ndarray,
    fp_indices_b: np.ndarray,
    device,
    t_end: int = 2700,
    save: bool = True,
):
    """Multi-phase comparison of two trials under baseline conditions.

    Produces a (K_phases × 4) figure with columns:
      0 — trial A field + trajectory
      1 — trial B field + trajectory
      2 — Δspeed (B − A)
      3 — angle change between A and B

    Parameters
    ----------
    trial_a : int
    trial_b : int
    model : Run_Model
    pca : sklearn.decomposition.PCA
    bounds : tuple
    x_phase_base : np.ndarray, shape (B, K, input_dim)
    hid : array-like, shape (T, B, H)
    phase_meta : list of (int, int)
    fp_project : np.ndarray, shape (nTfix, B, 2)
    fp_indices_a : np.ndarray of int
        Fixed point indices for trial A.
    fp_indices_b : np.ndarray of int
        Fixed point indices for trial B.
    device : torch.device
    t_end : int
    save : bool
    """
    from analysis.flow_field import (
        project_trajectory, filter_points_near_trajectory, compute_velocity_field,
    )

    K = len(phase_meta)
    color_a = get_trial_color(trial_a, "baseline")
    color_b = get_trial_color(trial_b, "baseline")

    # Shared normalisation across both trials and all phases
    peak_speeds = []
    for k in range(K):
        for t_idx in [trial_a, trial_b]:
            _, _, _, _, S = compute_velocity_field(
                model, pca, bounds, x_phase_base[t_idx, k], device
            )
            peak_speeds.append(np.nanpercentile(S, 99))
    vmax = float(np.nanmax(peak_speeds))
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    thr  = 0.05 * vmax

    traj_a = project_trajectory(pca, hid[:t_end], batch_idx=trial_a)
    traj_b = project_trajectory(pca, hid[:t_end], batch_idx=trial_b)

    fig, axes = plt.subplots(K, 4, figsize=(11.2, 2.6 * K),
                              dpi=300, constrained_layout=True)
    axes = np.atleast_2d(axes)

    col_labels = [f"Trial {trial_a}", f"Trial {trial_b}", "Δ speed (B − A)", "Angle (°)"]
    for j, label in enumerate(col_labels):
        axes[0, j].set_title(label, fontsize=8, fontweight="bold", pad=5)

    for k in range(K):
        b1, b2 = phase_meta[k]

        X1, X2, Ua, Va, Sa = compute_velocity_field(
            model, pca, bounds, x_phase_base[trial_a, k], device
        )
        _, _, Ub, Vb, Sb = compute_velocity_field(
            model, pca, bounds, x_phase_base[trial_b, k], device
        )
        Sa_m = np.where(Sa >= thr, Sa, np.nan)
        Sb_m = np.where(Sb >= thr, Sb, np.nan)

        dS = Sb - Sa
        eps = 1e-12
        cos = np.clip(
            (Ua*Ub + Va*Vb) / ((np.hypot(Ua, Va) + eps) * (np.hypot(Ub, Vb) + eps)),
            -1.0, 1.0,
        )
        ang_deg = np.arccos(cos) * 180 / np.pi

        axes[k, 0].set_ylabel(f"Phase {k}", fontsize=7, rotation=0,
                               labelpad=28, va="center")

        ax = axes[k, 0]
        _draw_speed_field(ax, X1, X2, Ua, Va, Sa_m, norm)
        _draw_trajectory(ax, traj_a, b1, b2, color_active=color_a)

        ax = axes[k, 1]
        _draw_speed_field(ax, X1, X2, Ub, Vb, Sb_m, norm)
        _draw_trajectory(ax, traj_b, b1, b2, color_active=color_b)

        ax = axes[k, 2]
        v = np.nanmax(np.abs(dS)) or 1.0
        ax.contourf(X1, X2, dS, levels=14, cmap=CMAP_DELTA, vmin=-v, vmax=v, zorder=1)
        for traj, col in [(traj_a, color_a), (traj_b, color_b)]:
            ax.plot(traj[:, 0], traj[:, 1], ls="--", lw=0.9, color=col, alpha=0.6, zorder=7)

        ax = axes[k, 3]
        ax.contourf(X1, X2, ang_deg, levels=14, cmap=CMAP_ANGLE, zorder=1)
        for traj, col in [(traj_a, color_a), (traj_b, color_b)]:
            ax.plot(traj[:, 0], traj[:, 1], ls="--", lw=0.9, color=col, alpha=0.6, zorder=7)

        for j in range(4):
            _style_flow_ax(axes[k, j], bounds,
                           add_pc_labels=(k == K - 1 and j == 0))

    _add_speed_colorbar(fig, axes[:, :2].ravel().tolist(), norm, CMAP_SPEED)

    if save:
        fname = f"flow_compare_baseline_trials_{trial_a}_vs_{trial_b}"
        fig.savefig(f"{fname}.png", bbox_inches="tight")
        fig.savefig(f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)