"""
Flow field analysis for RNN perturbation experiments.
Compares baseline vs perturbed dynamics in PCA-reduced state space.
"""

# Libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

# Project
import torch
from input_target import *
from hp import *
from model import *
from plotting_functions import *
from perturbations import *
from fixed_points_finder import *


COLORS: dict[str, tuple] = {
    "trial0_baseline":   (1.0,   0.549, 0.0  ),   # orange
    "trial0_perturbed":  (0.85,  0.25,  0.15 ),   # red
    "trial15_baseline":  (0.392, 0.584, 0.929),   # blue
    "trial15_perturbed": (0.30,  0.30,  0.75 ),   # indigo
}


def traj_color(trial_idx: int, condition: str) -> tuple:
    """Return the RGB colour for a given (trial, condition) pair."""
    key = f"trial{trial_idx}_{condition}"
    try:
        return COLORS[key]
    except KeyError:
        # Graceful fallback: derive from CMAP_TRIALS for unseen trial indices
        t = trial_idx / 15.0
        return CMAP_TRIALS(t)[:3]


def _lighten(color, amount: float = 0.55) -> tuple:
    """
    Return a lighter version of `color` by blending toward white.
    `amount=0` → original; `amount=1` → white.
    """
    rgb = np.array(mpl.colors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * amount)


#   CMAP_SPEED_BG   : for contourf background 
#   CMAP_SPEED_LINE : for streamlines 
#   CMAP_DELTA  : diverging for Δ speed  (RdBu_r: red = faster, blue = slower)
#   CMAP_ANGLE  : sequential for angle change (cividis: colorblind-safe)

def _clipped_cmap(cmap_src, lo: float = 0.0, hi: float = 1.0, N: int = 256):
    """
    Return a new ListedColormap using only the [lo, hi] portion of `cmap_src`.
    Prevents a sequential map from reaching its white or black extreme.
    """
    colors = cmap_src(np.linspace(lo, hi, N))
    return mpl.colors.ListedColormap(colors)


_speed_base = LinearSegmentedColormap.from_list(
    "_speed_base",
    ["#ffffffff", "#30303086"],            # white → very dark blue-black
    N=256,
)
CMAP_SPEED_BG   = _clipped_cmap(_speed_base, lo=0.05)                       # full range: white → dark (for bg fill)
CMAP_SPEED_LINE = _clipped_cmap(_speed_base, lo=0.2)  # clipped: never reaches white (for lines)

CMAP_DELTA = plt.cm.RdBu_r            # publication-standard diverging
CMAP_ANGLE = plt.cm.cividis           # perceptually uniform, colorblind-safe

# Alias used by the colorbar and any external callers
CMAP_SPEED = CMAP_SPEED_BG

# Legacy gradient kept for any remaining callers
START_COLOR  = COLORS["trial0_baseline"]
END_COLOR    = COLORS["trial15_baseline"]
CMAP_TRIALS  = LinearSegmentedColormap.from_list("custom_gradient", [START_COLOR, END_COLOR])


# ═══════════════════════════════════════════════════════════════════════════
#  Core helpers
# ═══════════════════════════════════════════════════════════════════════════

def _to_numpy(x):
    """Convert tensor or array-like to a numpy array."""
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


@torch.no_grad()
def _velocity_batch(model, H, x_const, use_continuous=False):
    """Compute hidden-state velocity dH/dt (or Δh) for a batch of points H."""
    cell  = model.model.rnn.rnncell
    r     = x_const @ cell.weight_ih.T + H @ cell.weight_hh.T
    if getattr(cell, "bias", None) is not None:
        r = r + cell.bias
    a = cell.nonlinearity(r)
    if use_continuous:
        tau = 1.0 / (1.0 - cell.decay)
        return (-H + a) / tau
    return (1.0 - cell.decay) * (a - H)


def split_points(pts):
    """Split an (N,2) array into first, last, and middle sub-arrays."""
    n = len(pts)
    if n == 0:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2))
    if n == 1:
        return pts[0:1], np.empty((0, 2)), np.empty((0, 2))
    return pts[0:1], pts[-1:], pts[1:-1] if n > 2 else np.empty((0, 2))


def filter_points_near_traj(points, traj, thresh=0.5):
    """Return a boolean mask for points within `thresh` of trajectory `traj`."""
    dist, _ = cKDTree(traj).query(points)
    return dist <= thresh, dist


# ═══════════════════════════════════════════════════════════════════════════
#  PCA / grid utilities
# ═══════════════════════════════════════════════════════════════════════════

def prepare_baseline_pca_with_union_bounds(hid_base, hid_pert, q_lo=1, q_hi=99):
    """
    Fit PCA on baseline hidden states; derive axis bounds from the union of
    baseline + perturbed projections.
    """
    Ha  = hid_base.reshape(-1, hid_base.shape[-1])
    pca = PCA(n_components=2).fit(Ha)

    H2_base = pca.transform(Ha)
    H2_pert = pca.transform(hid_pert.reshape(-1, hid_pert.shape[-1]))
    H2_all  = np.vstack([H2_base, H2_pert])

    x1min, x1max = np.percentile(H2_all[:, 0], [q_lo, q_hi])
    x2min, x2max = np.percentile(H2_all[:, 1], [q_lo, q_hi])
    return pca, ((x1min, x1max), (x2min, x2max))


def _make_grid(bounds, n):
    (x1min, x1max), (x2min, x2max) = bounds
    X1, X2 = np.meshgrid(
        np.linspace(x1min, x1max, n),
        np.linspace(x2min, x2max, n),
    )
    return X1, X2, np.stack([X1.ravel(), X2.ravel()], axis=1)


@torch.no_grad()
def project_traj(pca, hid_seq, batch_idx=0, t_slice=None):
    """Project hidden states of one trial onto the PCA plane."""
    H = hid_seq[:, batch_idx, :]
    if t_slice is not None:
        H = H[t_slice]
    return pca.transform(_to_numpy(H))


# ═══════════════════════════════════════════════════════════════════════════
#  Flow-field computation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_field_with_input(
    model, pca, bounds, x_const, n=81, smooth_sigma=0.2, use_continuous=False
):
    """Compute the 2-D velocity field on a grid given a constant input x_const."""
    X1, X2, grid2d = _make_grid(bounds, n)
    H_high  = torch.tensor(pca.inverse_transform(grid2d), dtype=torch.float32, device=device)
    x_const = torch.as_tensor(x_const, dtype=torch.float32, device=device).view(-1)

    Vh    = _velocity_batch(model, H_high, x_const, use_continuous=use_continuous)
    comps = torch.tensor(pca.components_[:2], dtype=torch.float32, device=device)
    V2d   = (Vh @ comps.T).cpu().numpy()

    U, V  = V2d[:, 0].reshape(n, n), V2d[:, 1].reshape(n, n)
    if smooth_sigma:
        U, V = gaussian_filter(U, sigma=smooth_sigma), gaussian_filter(V, sigma=smooth_sigma)
    return X1, X2, U, V, np.hypot(U, V)


def compute_field_delta(
    model, pca, bounds, x_base, x_pert, n=81, smooth_sigma=0.2, use_continuous=False
):
    """
    Compute baseline and perturbed fields, plus delta metrics (ΔU, ΔV, Δspeed,
    angle between vectors).
    """
    X1, X2, U0, V0, S0 = compute_field_with_input(
        model, pca, bounds, x_base, n=n, smooth_sigma=smooth_sigma,
        use_continuous=use_continuous,
    )
    _, _, U1, V1, S1 = compute_field_with_input(
        model, pca, bounds, x_pert, n=n, smooth_sigma=smooth_sigma,
        use_continuous=use_continuous,
    )
    dU, dV, dS = U1 - U0, V1 - V0, S1 - S0

    eps  = 1e-12
    cos  = np.clip((U0*U1 + V0*V1) / ((np.hypot(U0, V0) + eps) * (np.hypot(U1, V1) + eps)), -1.0, 1.0)
    ang  = np.arccos(cos)

    return X1, X2, U0, V0, S0, U1, V1, S1, dU, dV, dS, ang


# ═══════════════════════════════════════════════════════════════════════════
#  Input binning
# ═══════════════════════════════════════════════════════════════════════════

def phase_binned_inputs_time_locked(inputs, K=None, *, edges=None, bins=None,
                                    axis_time=0, axis_batch=1, per_trial=False):
    """
    Average inputs within time bins.

    Parameters
    ----------
    inputs     : array-like, shape (T, B, Nin)
    K          : int – number of equal-width bins (mutually exclusive with edges/bins)
    edges      : 1-D int array of bin edges, length M+1
    bins       : list of (t0, t1) pairs
    per_trial  : if True, shape (B, M, Nin); else (M, Nin)

    Returns
    -------
    x_phase    : averaged inputs
    out_bins   : list of (t0, t1) pairs actually used
    """
    Xnp = inputs.detach().cpu().numpy() if torch.is_tensor(inputs) else np.asarray(inputs)
    Xnp = np.moveaxis(Xnp, [axis_time, axis_batch], [0, 1])
    T, B, Nin = Xnp.shape[0], Xnp.shape[1], int(np.prod(Xnp.shape[2:]))
    X = Xnp.reshape(T, B, Nin)

    if bins is not None:
        out_bins = [(int(t0), int(t1)) for t0, t1 in bins]
    elif edges is not None:
        e = np.asarray(edges, dtype=int)
        out_bins = [(int(e[i]), int(e[i+1])) for i in range(len(e) - 1)]
    elif K is not None:
        e = np.linspace(0, T, int(K) + 1, dtype=int)
        out_bins = [(int(e[i]), int(e[i+1])) for i in range(len(e) - 1)]
    else:
        raise ValueError("Provide `bins`, `edges`, or `K`.")

    M = len(out_bins)
    if per_trial:
        x_phase = np.full((B, M, Nin), np.nan, dtype=X.dtype)
        for m, (t0, t1) in enumerate(out_bins):
            t0c, t1c = max(0, t0), min(T, t1)
            if t1c > t0c:
                x_phase[:, m, :] = X[t0c:t1c].mean(axis=0)
    else:
        x_phase = np.full((M, Nin), np.nan, dtype=X.dtype)
        for m, (t0, t1) in enumerate(out_bins):
            t0c, t1c = max(0, t0), min(T, t1)
            if t1c > t0c:
                x_phase[m, :] = X[t0c:t1c].mean(axis=(0, 1))

    return x_phase, out_bins


# ═══════════════════════════════════════════════════════════════════════════
#  Trajectory-experienced field change
# ═══════════════════════════════════════════════════════════════════════════

def _build_interpolators(X1, X2, U, V):
    x, y = X1[0, :], X2[:, 0]
    fu = RegularGridInterpolator((y, x), U, bounds_error=False, fill_value=np.nan)
    fv = RegularGridInterpolator((y, x), V, bounds_error=False, fill_value=np.nan)
    return fu, fv


def _interp_vectors(fu, fv, pts2d):
    """Interpolate (U, V) at (T, 2) points with columns [x, y]."""
    return np.stack([fu(pts2d[:, [1, 0]]), fv(pts2d[:, [1, 0]])], axis=1)


def _traj_tangent_normal(traj2d, eps=1e-12):
    d     = np.gradient(traj2d, axis=0)
    speed = np.linalg.norm(d, axis=1, keepdims=True)
    t_hat = d / (speed + eps)
    n_hat = np.stack([-t_hat[:, 1], t_hat[:, 0]], axis=1)
    return t_hat, n_hat, speed.squeeze(-1)


def experienced_change_along_trajectory(X1, X2, U0, V0, U1, V1, traj2d, dt=1.0, eps=1e-12):
    """
    Quantify how the field change is experienced along a trajectory.

    Returns a dict with integrated (I_par, I_norm), angular deflection (dtheta),
    and normalised variants (per unit time / path length).
    """
    v0 = _interp_vectors(*_build_interpolators(X1, X2, U0, V0), traj2d)
    v1 = _interp_vectors(*_build_interpolators(X1, X2, U1, V1), traj2d)
    dv = v1 - v0

    t_hat, n_hat, base_path_rate = _traj_tangent_normal(traj2d, eps=eps)
    v0_norm = np.linalg.norm(v0, axis=1) + eps

    dv_par  = np.sum(dv * t_hat, axis=1)
    dv_norm = np.sum(dv * n_hat, axis=1)
    time    = np.arange(traj2d.shape[0]) * dt

    I_par  = np.trapz(dv_par,  time)
    I_norm = np.trapz(dv_norm, time)
    dtheta = np.trapz(dv_norm / v0_norm, time)

    T = time[-1] - time[0] + eps
    L = np.trapz(base_path_rate, time) + eps

    return dict(
        I_par=I_par, I_norm=I_norm, dtheta=dtheta,
        par_mean=I_par / T,   norm_mean=I_norm / T,
        par_per_len=I_par / L, norm_per_len=I_norm / L,
        dv_par_series=dv_par, dv_norm_series=dv_norm, time=time,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Plotting helpers
# ═══════════════════════════════════════════════════════════════════════════

def _plot_traj(ax, traj, b1, b2, color_active, color_ghost=None, lw_active=3.5):
    """
    Draw a trajectory with a highlighted phase segment.

    Parameters
    ----------
    color_active : colour for the active phase segment (solid line + markers).
    color_ghost  : colour for the full ghost path (dashed).  If None, a
                   lightened version of `color_active` is used so hue identity
                   is preserved even when faded.
    lw_active    : line-width for the active segment.
    """
    if color_ghost is None:
        color_ghost = _lighten(color_active, amount=0.5)

    stroke = lambda w: [pe.withStroke(linewidth=w, foreground="black")]

    # Full trajectory — thin dashed ghost, no stroke (keeps it subtle)
    ax.plot(traj[:, 0], traj[:, 1],
            ls="--", lw=2, color=color_ghost, alpha=1, zorder=7)

    # Active phase segment — solid with white halo for legibility on dark bg
    ax.plot(traj[b1:b2, 0], traj[b1:b2, 1],
            ls="-", lw=lw_active, path_effects= stroke(w=4.0), color=color_active, zorder=8)

    # Start dot
    ax.scatter(traj[b1, 0], traj[b1, 1],
               s=45, color=color_active, zorder=10)
    # End arrow marker
    ax.scatter(traj[b2 - 1, 0], traj[b2 - 1, 1],
               s=45, facecolor=color_active, edgecolor="black",
               linewidths=1.0, zorder=10)


def _apply_ax_style(ax, bounds, add_pc_labels=False):
    """
    Publication-style axis: equal aspect, tight limits, spine-free, no ticks.
    Optionally adds 'PC 1' / 'PC 2' axis labels (for outermost panels only).
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


def _draw_speed_field(ax, X1, X2, U, V, S, norm, cmap=None):
    """
    Render a speed field with:
      • contourf   — filled background using CMAP_SPEED_BG (white at zero,
                     so empty regions stay white)
      • streamplot — lines coloured by CMAP_SPEED_LINE, which is clipped to
                     never reach white, guaranteeing contrast against the
                     background at all speed values. Width scales with speed
                     for an additional depth cue.

    The `cmap` argument is accepted for API compatibility but ignored;
    CMAP_SPEED_BG / CMAP_SPEED_LINE are always used.
    """
    S_safe = np.nan_to_num(S)
    U_safe = np.nan_to_num(U)
    V_safe = np.nan_to_num(V)

    # Background: full white-to-dark range so the zero-speed region is white
    ax.contourf(X1, X2, S_safe, levels=14, norm=norm,
                cmap=CMAP_SPEED_BG, alpha=0.80, zorder=1)

    # Streamlines: clipped cmap (min ~light gray) + speed-scaled linewidth
    lw_field = np.clip(0.6 + 2.0 * (S_safe / (norm.vmax + 1e-12)), 0.6, 2.6)

    # Build a matching Normalize for the clipped cmap so color mapping is
    # consistent with the background (both anchored to the same vmax)
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


def _add_speed_colorbar(fig, axes_subset, norm, cmap):
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes_subset, shrink=0.7, pad=0.02,
                        aspect=20, fraction=0.046)
    cbar.ax.set_title("speed", fontsize=7, pad=4)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_visible(False)


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 1 – Baseline vs Perturbed (per trial)
# ═══════════════════════════════════════════════════════════════════════════

def _fp_selection_for_trial(trial_idx, fp_project, nTfix):
    """Return PCA-projected fixed points and middle slice for a given trial."""
    if trial_idx == 15:
        idx = np.r_[1, 20:80, nTfix - 1]
        mid_slice = slice(7, None)
    else:  # trial 0 (and fallback)
        idx = np.r_[1, 30:58, nTfix - 1]
        mid_slice = slice(5, None)
    pts = fp_project[idx, trial_idx, :]
    return pts, mid_slice


def plot_baseline_vs_perturbed(trial_idx, model, pca, bounds, x_phase_base,
                               x_phase_pert, hid, hid_pert, phase_meta,
                               fp_project, nTfix, trials_norm, cmap_trials):
    K_phases = len(phase_meta)

    # ── resolve condition colours for this trial ─────────────────────────
    col_base = traj_color(trial_idx, "baseline")
    col_pert = traj_color(trial_idx, "perturbed")

    # ── global speed normalisation ───────────────────────────────────────
    all_speeds = []
    for k in range(K_phases):
        _, _, _, _, S0 = compute_field_with_input(model, pca, bounds, x_phase_base[trial_idx, k])
        _, _, _, _, S1 = compute_field_with_input(model, pca, bounds, x_phase_pert[trial_idx, k])
        all_speeds += [np.nanpercentile(S0, 99), np.nanpercentile(S1, 99)]
    vmax = float(np.nanmax(all_speeds))
    thr  = 0.05 * vmax
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)

    traj_base = project_traj(pca, hid[:2700],      batch_idx=trial_idx)
    traj_pert = project_traj(pca, hid_pert[:2700],  batch_idx=trial_idx)

    pts_raw, _ = _fp_selection_for_trial(trial_idx, fp_project, nTfix)
    mask, _    = filter_points_near_traj(pts_raw, traj_base, thresh=0.5)

    # ── figure ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(K_phases, 4, figsize=(11.2, 2.6*K_phases),
                             dpi=300, constrained_layout=True)
    axes = np.atleast_2d(axes)

    # Column headers (written once, above the top row)
    col_labels = [f"Trial {trial_idx}  baseline", "perturbed", "Δ speed", "angle (°)"]
    for j, label in enumerate(col_labels):
        axes[0, j].set_title(label, fontsize=8, fontweight="bold", pad=5)

    for k in range(K_phases):
        b1, b2 = phase_meta[k]
        X1, X2, U0, V0, S0, U1, V1, S1, dU, dV, dS, ang = compute_field_delta(
            model, pca, bounds, x_phase_base[trial_idx, k], x_phase_pert[trial_idx, k]
        )
        Sm0 = np.where(S0 >= thr, S0, np.nan)
        Sm1 = np.where(S1 >= thr, S1, np.nan)

        # Phase label on the leftmost panel
        axes[k, 0].set_ylabel(f"phase {k}", fontsize=7, rotation=0,
                               labelpad=28, va="center")

        # Col 0 – baseline field + baseline trajectory
        ax = axes[k, 0]
        _draw_speed_field(ax, X1, X2, U0, V0, Sm0, norm, CMAP_SPEED)
        _plot_traj(ax, traj_base, b1, b2, color_active=col_base)

        # Col 1 – perturbed field; show pert traj (active) + baseline (ghost)
        ax = axes[k, 1]
        _draw_speed_field(ax, X1, X2, U1, V1, Sm1, norm, CMAP_SPEED)
        # Baseline ghost first (beneath perturbed)
        _plot_traj(ax, traj_base, b1, b2, color_active=col_base)
        _plot_traj(ax, traj_pert, b1, b2, color_active=col_pert)

        # Col 2 – Δ speed (diverging)
        ax = axes[k, 2]
        v = np.nanmax(np.abs(dS)) or 1.0
        ax.contourf(X1, X2, dS, levels=14, cmap=CMAP_DELTA, vmin=-v, vmax=v, zorder=1)
        ax.plot(traj_base[:, 0], traj_base[:, 1],
                ls="--", lw=0.9, color=col_base, alpha=0.6, zorder=7)
        ax.plot(traj_pert[:, 0], traj_pert[:, 1],
                ls="--", lw=0.9, color=col_pert, alpha=0.6, zorder=7)

        # Col 3 – angle change (sequential, always 0–180°)
        ax = axes[k, 3]
        im = ax.contourf(X1, X2, ang * 180 / np.pi, levels=14, cmap=CMAP_ANGLE, zorder=1)
        ax.plot(traj_base[:, 0], traj_base[:, 1],
                ls="--", lw=0.9, color=col_base, alpha=0.6, zorder=7)
        ax.plot(traj_pert[:, 0], traj_pert[:, 1],
                ls="--", lw=0.9, color=col_pert, alpha=0.6, zorder=7)

        # Axis cosmetics — add PC labels only on bottom-left panel
        for j in range(4):
            _apply_ax_style(axes[k, j], bounds,
                            add_pc_labels=False)

    _add_speed_colorbar(fig, axes[:, :2].ravel().tolist(), norm, CMAP_SPEED)

    fname = f"flow_compare_trial{trial_idx}"
    fig.savefig(f"{fname}.png", bbox_inches="tight")
    fig.savefig(f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)



# ═══════════════════════════════════════════════════════════════════════════
#  Figure 2 – Trial 0 vs Trial 15 (baseline only, across phases)
# ═══════════════════════════════════════════════════════════════════════════

def plot_trial_comparison(trial_a, trial_b, model, pca, bounds,
                          x_phase_base, hid, phase_meta,
                          fp_project, nTfix, trials_norm, cmap_trials):
    K_phases = len(phase_meta)

    # ── condition colours — both are baselines ───────────────────────────
    color_a = traj_color(trial_a, "baseline")
    color_b = traj_color(trial_b, "baseline")

    # ── global speed normalisation ───────────────────────────────────────
    all_pctl99 = []
    for k in range(K_phases):
        _, _, _, _, Sa = compute_field_with_input(model, pca, bounds, x_phase_base[trial_a, k])
        _, _, _, _, Sb = compute_field_with_input(model, pca, bounds, x_phase_base[trial_b, k])
        all_pctl99 += [np.nanpercentile(Sa, 99), np.nanpercentile(Sb, 99)]
    vmax = float(np.nanmax(all_pctl99))
    thr  = 0.05 * vmax
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)

    traj_a = project_traj(pca, hid[:2700], batch_idx=trial_a)
    traj_b = project_traj(pca, hid[:2700], batch_idx=trial_b)

    # Fixed points
    pts_a_raw = fp_project[np.r_[1, 30:58, nTfix - 1], trial_a, :]
    pts_b_raw = fp_project[np.r_[1, 20:80, nTfix - 1], trial_b, :]
    pts_a = pts_a_raw[filter_points_near_traj(pts_a_raw, traj_a, thresh=0.5)[0]]
    pts_b = pts_b_raw[filter_points_near_traj(pts_b_raw, traj_b, thresh=0.5)[0]]

    # ── figure ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(K_phases, 4, figsize=(11.2, 2.6*K_phases),
                             dpi=300, constrained_layout=True)
    axes = np.atleast_2d(axes)

    col_labels = [f"Trial {trial_a}", f"Trial {trial_b}", "Δ speed  (B − A)", "angle (°)"]
    for j, label in enumerate(col_labels):
        axes[0, j].set_title(label, fontsize=8, fontweight="bold", pad=5)

    for k in range(K_phases):
        b1, b2 = phase_meta[k]

        X1, X2, Ua, Va, Sa = compute_field_with_input(
            model, pca, bounds, x_phase_base[trial_a, k]
        )
        _, _, Ub, Vb, Sb = compute_field_with_input(
            model, pca, bounds, x_phase_base[trial_b, k]
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

        axes[k, 0].set_ylabel(f"phase {k}", fontsize=7, rotation=0,
                               labelpad=28, va="center")

        # Col 0 – Trial A
        ax = axes[k, 0]
        _draw_speed_field(ax, X1, X2, Ua, Va, Sa_m, norm, CMAP_SPEED)
        _plot_traj(ax, traj_a, b1, b2, color_active=color_a)

        # Col 1 – Trial B
        ax = axes[k, 1]
        _draw_speed_field(ax, X1, X2, Ub, Vb, Sb_m, norm, CMAP_SPEED)
        _plot_traj(ax, traj_b, b1, b2, color_active=color_b)

        # Col 2 – Δ speed (B − A)
        ax = axes[k, 2]
        v = np.nanmax(np.abs(dS)) or 1.0
        ax.contourf(X1, X2, dS, levels=14, cmap=CMAP_DELTA, vmin=-v, vmax=v, zorder=1)
        ax.plot(traj_a[:, 0], traj_a[:, 1], ls="--", lw=0.9, color=color_a, alpha=0.6, zorder=7)
        ax.plot(traj_b[:, 0], traj_b[:, 1], ls="--", lw=0.9, color=color_b, alpha=0.6, zorder=7)

        # Col 3 – angle change
        ax = axes[k, 3]
        ax.contourf(X1, X2, ang_deg, levels=14, cmap=CMAP_ANGLE, zorder=1)
        ax.plot(traj_a[:, 0], traj_a[:, 1], ls="--", lw=0.9, color=color_a, alpha=0.6, zorder=7)
        ax.plot(traj_b[:, 0], traj_b[:, 1], ls="--", lw=0.9, color=color_b, alpha=0.6, zorder=7)

        for j in range(4):
            _apply_ax_style(axes[k, j], bounds,
                            add_pc_labels=(k == K_phases - 1 and j == 0))

    _add_speed_colorbar(fig, axes[:, :2].ravel().tolist(), norm, CMAP_SPEED)

    fname = f"flow_compare_baseline_trials_{trial_a}_vs_{trial_b}"
    fig.savefig(f"{fname}.png", bbox_inches="tight")
    fig.savefig(f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)





if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = get_default_params(batch_size=16, test=True)
    hp_tanh = get_default_hp(2, 1, activation="tanh", test=True)

    model_tanh = Run_Model(hp_tanh, params, RNNLayer=RNNLayer).to(device)
    model_tanh.load_state_dict(
        torch.load("model_tanh_best.pt", map_location=device, weights_only=False)
    )

    X_test, Y_test = model_tanh.generate_trials(batch_size=16)

    START_PERT, END_PERT = 1150,1550
    X_test_pert = apply_perturbation(
        X_test, channel=0, start_idx=START_PERT, end_idx=END_PERT, magnitude=-0.4,
    )

    model_tanh.eval()
    with torch.no_grad():
        _, _, _, pred_tanh, hid      = model_tanh(X=X_test,      Y=Y_test)
        _, _, _, pred_pert, hid_pert = model_tanh(X=X_test_pert, Y=Y_test)
        plot_output_targets(pred_tanh, Y_test)
        plot_output_targets(pred_pert, Y_test)
        plt.show()



    edges_sorted = sorted({0, 1550, 1650, START_PERT, END_PERT, 2500})

    x_phase_base, phase_meta = phase_binned_inputs_time_locked(
        X_test, edges=edges_sorted, per_trial=True
    )
    x_phase_pert, _ = phase_binned_inputs_time_locked(
        X_test_pert, edges=edges_sorted, per_trial=True
    )

    print("Edges used:", edges_sorted)
    print("Phase bins:", phase_meta)

    pca_ref, bounds_ref = prepare_baseline_pca_with_union_bounds(
        _to_numpy(hid), _to_numpy(hid_pert), q_lo=0, q_hi=100
    )

    # Fixed-point projections
    fixed_pts  = np.load("fixed_points.npy")             # shape (H, B, nTfix) before transpose
    fixed      = np.transpose(fixed_pts, (2, 0, 1))       # → (nTfix, B, H)
    nTfix, B_fp, H_fp = fixed.shape
    fp_project = pca_ref.transform(fixed.reshape(-1, H_fp)).reshape(nTfix, B_fp, 2)

    trials_norm = np.linspace(0, 1, 16)
    K           = len(phase_meta)

    
    for trial_idx in [0, 15]:
        plot_baseline_vs_perturbed(
            trial_idx, model_tanh, pca_ref, bounds_ref,
            x_phase_base, x_phase_pert,
            hid, hid_pert, phase_meta,
            fp_project, nTfix, trials_norm, CMAP_TRIALS,
        )

    plot_trial_comparison(
        0, 15, model_tanh, pca_ref, bounds_ref,
        x_phase_base, hid, phase_meta,
        fp_project, nTfix, trials_norm, CMAP_TRIALS,)
