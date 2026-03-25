"""
analysis/flow_field.py
-----------------------
Velocity field computation, PCA utilities, phase binning, and
trajectory-based field analysis for RNN perturbation experiments.

All functions are pure computation — no plotting happens here.
See plotting/plot_flow_field.py for visualisation.

Conventions
-----------
- Hidden states have shape (T, B, H): time × batch × hidden units.
- PCA is always 2-component unless otherwise stated, and is fit on the
  baseline condition so the same axes are used for all comparisons.
- `device` is always passed explicitly; no module-level globals are used.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA


# General utilities


def to_numpy(x) -> np.ndarray:
    """Convert a torch tensor or array-like to a numpy array.

    Parameters
    ----------
    x : torch.Tensor or array-like

    Returns
    -------
    np.ndarray
    """
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def split_points(pts: np.ndarray):
    """Split an (N, 2) array into first, last, and middle sub-arrays.

    Parameters
    ----------
    pts : np.ndarray, shape (N, 2)

    Returns
    -------
    first : np.ndarray, shape (1, 2) or (0, 2)
    last  : np.ndarray, shape (1, 2) or (0, 2)
    middle : np.ndarray, shape (N-2, 2) or (0, 2)
    """
    n = len(pts)
    if n == 0:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2))
    if n == 1:
        return pts[0:1], np.empty((0, 2)), np.empty((0, 2))
    return pts[0:1], pts[-1:], pts[1:-1] if n > 2 else np.empty((0, 2))


def filter_points_near_trajectory(
    points: np.ndarray,
    trajectory: np.ndarray,
    threshold: float = 0.5,
):
    """Return a boolean mask for points within `threshold` of a trajectory.

    Parameters
    ----------
    points : np.ndarray, shape (N, 2)
    trajectory : np.ndarray, shape (T, 2)
    threshold : float
        Maximum distance to the nearest trajectory point.

    Returns
    -------
    mask : np.ndarray of bool, shape (N,)
    distances : np.ndarray of float, shape (N,)
    """
    distances, _ = cKDTree(trajectory).query(points)
    return distances <= threshold, distances


# RNN velocity

@torch.no_grad()
def compute_velocity(
    model,
    hidden_states: torch.Tensor,
    x_const: torch.Tensor,
    device: torch.device,
    use_continuous: bool = False,
) -> torch.Tensor:
    """Compute the hidden-state velocity dH/dt (or Δh) for a batch of points.

    For a discrete RNN:  velocity = (1 - decay) * (activation(r) - H)
    For a continuous RNN: velocity = (-H + activation(r)) / tau

    Parameters
    ----------
    model : Run_Model
        Trained model with a `.model.rnn.rnncell` attribute.
    hidden_states : torch.Tensor, shape (N, H)
        Batch of hidden states at which to evaluate the velocity.
    x_const : torch.Tensor, shape (input_dim,)
        Constant input vector held fixed during evaluation.
    device : torch.device
    use_continuous : bool
        If True, use the continuous-time formulation.

    Returns
    -------
    velocity : torch.Tensor, shape (N, H)
    """
    cell = model.model.rnn.rnncell
    r = x_const @ cell.weight_ih.T + hidden_states @ cell.weight_hh.T
    if getattr(cell, "bias", None) is not None:
        r = r + cell.bias
    a = cell.nonlinearity(r)

    if use_continuous:
        tau = 1.0 / (1.0 - cell.decay)
        return (-hidden_states + a) / tau
    return (1.0 - cell.decay) * (a - hidden_states)


# PCA and grid utilities

def fit_pca_with_union_bounds(
    hid_base: np.ndarray,
    hid_pert: np.ndarray,
    n_components: int = 2,
    q_lo: float = 1.0,
    q_hi: float = 99.0,
) -> tuple[PCA, tuple]:
    """Fit PCA on baseline hidden states; derive axis bounds from both conditions.

    Using the union of baseline and perturbed projections to set bounds
    ensures that perturbed trajectories remain within the plotted region.

    Parameters
    ----------
    hid_base : np.ndarray, shape (T, B, H)
        Baseline hidden states.
    hid_pert : np.ndarray, shape (T, B, H)
        Perturbed hidden states.
    n_components : int
    q_lo : float
        Lower percentile for axis bounds.
    q_hi : float
        Upper percentile for axis bounds.

    Returns
    -------
    pca : sklearn.decomposition.PCA
        Fitted PCA object (use for all subsequent projections).
    bounds : tuple of ((x1min, x1max), (x2min, x2max))
    """
    pca = PCA(n_components=n_components).fit(hid_base.reshape(-1, hid_base.shape[-1]))

    proj_base = pca.transform(hid_base.reshape(-1, hid_base.shape[-1]))
    proj_pert = pca.transform(hid_pert.reshape(-1, hid_pert.shape[-1]))
    proj_all  = np.vstack([proj_base, proj_pert])

    x1min, x1max = np.percentile(proj_all[:, 0], [q_lo, q_hi])
    x2min, x2max = np.percentile(proj_all[:, 1], [q_lo, q_hi])
    return pca, ((x1min, x1max), (x2min, x2max))


def _make_grid(bounds: tuple, n: int):
    """Build a regular 2-D grid over the given PC bounds.

    Parameters
    ----------
    bounds : ((x1min, x1max), (x2min, x2max))
    n : int
        Grid resolution along each axis.

    Returns
    -------
    X1, X2 : np.ndarray, shape (n, n)
    grid2d : np.ndarray, shape (n*n, 2)
    """
    (x1min, x1max), (x2min, x2max) = bounds
    X1, X2 = np.meshgrid(
        np.linspace(x1min, x1max, n),
        np.linspace(x2min, x2max, n),
    )
    return X1, X2, np.stack([X1.ravel(), X2.ravel()], axis=1)


@torch.no_grad()
def project_trajectory(
    pca: PCA,
    hid_seq,
    batch_idx: int = 0,
    t_slice: slice = None,
) -> np.ndarray:
    """Project the hidden states of one trial onto the 2-D PCA plane.

    Parameters
    ----------
    pca : sklearn.decomposition.PCA
    hid_seq : array-like or torch.Tensor, shape (T, B, H)
    batch_idx : int
        Which trial to project.
    t_slice : slice, optional
        Time range to extract before projecting.

    Returns
    -------
    np.ndarray, shape (T, 2)
    """
    H = to_numpy(hid_seq)[:, batch_idx, :]
    if t_slice is not None:
        H = H[t_slice]
    return pca.transform(H)


# Flow field computation

@torch.no_grad()
def compute_velocity_field(
    model,
    pca: PCA,
    bounds: tuple,
    x_const,
    device: torch.device,
    n: int = 81,
    smooth_sigma: float = 0.2,
    use_continuous: bool = False,
) -> tuple:
    """Compute the 2-D velocity field on a grid for a constant input.

    The full-dimensional velocity is projected onto the PCA plane and
    optionally smoothed with a Gaussian filter.

    Parameters
    ----------
    model : Run_Model
    pca : sklearn.decomposition.PCA
    bounds : tuple
        Axis bounds from `fit_pca_with_union_bounds`.
    x_const : array-like, shape (input_dim,)
        Constant input vector.
    device : torch.device
    n : int
        Grid resolution.
    smooth_sigma : float
        Gaussian smoothing sigma (0 = no smoothing).
    use_continuous : bool

    Returns
    -------
    X1, X2 : np.ndarray, shape (n, n) — grid coordinates
    U, V   : np.ndarray, shape (n, n) — velocity components in PC space
    S      : np.ndarray, shape (n, n) — speed (magnitude of (U, V))
    """
    X1, X2, grid2d = _make_grid(bounds, n)
    H = torch.tensor(pca.inverse_transform(grid2d), dtype=torch.float32, device=device)
    x = torch.as_tensor(x_const, dtype=torch.float32, device=device).view(-1)

    Vh = compute_velocity(model, H, x, device, use_continuous=use_continuous)
    comps = torch.tensor(pca.components_[:2], dtype=torch.float32, device=device)
    V2d = (Vh @ comps.T).cpu().numpy()

    U = V2d[:, 0].reshape(n, n)
    V = V2d[:, 1].reshape(n, n)
    if smooth_sigma:
        U = gaussian_filter(U, sigma=smooth_sigma)
        V = gaussian_filter(V, sigma=smooth_sigma)
    return X1, X2, U, V, np.hypot(U, V)


def compute_field_delta(
    model,
    pca: PCA,
    bounds: tuple,
    x_base,
    x_pert,
    device: torch.device,
    n: int = 81,
    smooth_sigma: float = 0.2,
    use_continuous: bool = False,
) -> tuple:
    """Compute baseline and perturbed fields, plus their differences.

    Parameters
    ----------
    model : Run_Model
    pca : sklearn.decomposition.PCA
    bounds : tuple
    x_base : array-like
        Baseline input vector.
    x_pert : array-like
        Perturbed input vector.
    device : torch.device
    n : int
    smooth_sigma : float
    use_continuous : bool

    Returns
    -------
    X1, X2     : grid coordinates
    U0, V0, S0 : baseline velocity field
    U1, V1, S1 : perturbed velocity field
    dU, dV, dS : component-wise and speed differences
    ang        : angle between baseline and perturbed vectors (radians)
    """
    X1, X2, U0, V0, S0 = compute_velocity_field(
        model, pca, bounds, x_base, device, n=n,
        smooth_sigma=smooth_sigma, use_continuous=use_continuous,
    )
    _, _, U1, V1, S1 = compute_velocity_field(
        model, pca, bounds, x_pert, device, n=n,
        smooth_sigma=smooth_sigma, use_continuous=use_continuous,
    )
    dU, dV, dS = U1 - U0, V1 - V0, S1 - S0

    eps = 1e-12
    cos = np.clip(
        (U0*U1 + V0*V1) / ((np.hypot(U0, V0) + eps) * (np.hypot(U1, V1) + eps)),
        -1.0, 1.0,
    )
    return X1, X2, U0, V0, S0, U1, V1, S1, dU, dV, dS, np.arccos(cos)


# Phase binning

def phase_binned_inputs(
    inputs,
    K: int = None,
    *,
    edges=None,
    bins=None,
    axis_time: int = 0,
    axis_batch: int = 1,
    per_trial: bool = False,
) -> tuple[np.ndarray, list]:
    """Average inputs within time bins.

    Exactly one of `K`, `edges`, or `bins` must be provided.

    Parameters
    ----------
    inputs : array-like or torch.Tensor, shape (T, B, Nin)
    K : int, optional
        Number of equal-width bins spanning the full time axis.
    edges : array-like of int, optional
        Bin edge indices of length M+1, defining M bins.
    bins : list of (t0, t1), optional
        Explicit bin start/end pairs.
    axis_time : int
        Axis corresponding to time in `inputs`.
    axis_batch : int
        Axis corresponding to batch in `inputs`.
    per_trial : bool
        If True, return shape (B, M, Nin); else (M, Nin) averaged over trials.

    Returns
    -------
    x_phase : np.ndarray
        Phase-averaged inputs.
    out_bins : list of (int, int)
        The bin boundaries actually used.
    """
    X = to_numpy(inputs)
    X = np.moveaxis(X, [axis_time, axis_batch], [0, 1])
    T, B, Nin = X.shape[0], X.shape[1], int(np.prod(X.shape[2:]))
    X = X.reshape(T, B, Nin)

    if bins is not None:
        out_bins = [(int(t0), int(t1)) for t0, t1 in bins]
    elif edges is not None:
        e = np.asarray(edges, dtype=int)
        out_bins = [(int(e[i]), int(e[i+1])) for i in range(len(e) - 1)]
    elif K is not None:
        e = np.linspace(0, T, int(K) + 1, dtype=int)
        out_bins = [(int(e[i]), int(e[i+1])) for i in range(len(e) - 1)]
    else:
        raise ValueError("Provide exactly one of `bins`, `edges`, or `K`.")

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


# Trajectory-experienced field change

def _build_interpolators(X1, X2, U, V):
    """Build 2-D interpolators for the (U, V) velocity components."""
    x, y = X1[0, :], X2[:, 0]
    fu = RegularGridInterpolator((y, x), U, bounds_error=False, fill_value=np.nan)
    fv = RegularGridInterpolator((y, x), V, bounds_error=False, fill_value=np.nan)
    return fu, fv


def _interpolate_vectors(fu, fv, pts2d: np.ndarray) -> np.ndarray:
    """Interpolate (U, V) at a set of 2-D points.

    Parameters
    ----------
    pts2d : np.ndarray, shape (T, 2) with columns [x, y]

    Returns
    -------
    np.ndarray, shape (T, 2)
    """
    return np.stack([fu(pts2d[:, [1, 0]]), fv(pts2d[:, [1, 0]])], axis=1)


def _trajectory_tangent_normal(traj2d: np.ndarray, eps: float = 1e-12):
    """Compute unit tangent and normal vectors along a 2-D trajectory.

    Parameters
    ----------
    traj2d : np.ndarray, shape (T, 2)
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    t_hat : np.ndarray, shape (T, 2) — unit tangent
    n_hat : np.ndarray, shape (T, 2) — unit normal (90° CCW from tangent)
    speed : np.ndarray, shape (T,)   — tangent vector magnitude
    """
    d     = np.gradient(traj2d, axis=0)
    speed = np.linalg.norm(d, axis=1, keepdims=True)
    t_hat = d / (speed + eps)
    n_hat = np.stack([-t_hat[:, 1], t_hat[:, 0]], axis=1)
    return t_hat, n_hat, speed.squeeze(-1)


def experienced_field_change(
    X1, X2,
    U0, V0,
    U1, V1,
    traj2d: np.ndarray,
    dt: float = 1.0,
    eps: float = 1e-12,
) -> dict:
    """Quantify how the perturbation-induced field change is experienced along a trajectory.

    Decomposes the change vector Δv = v1 - v0 into components parallel and
    perpendicular to the trajectory and integrates each over time.

    Parameters
    ----------
    X1, X2 : np.ndarray, shape (n, n)
        Grid coordinates from `compute_velocity_field`.
    U0, V0 : np.ndarray, shape (n, n)
        Baseline velocity field components.
    U1, V1 : np.ndarray, shape (n, n)
        Perturbed velocity field components.
    traj2d : np.ndarray, shape (T, 2)
        Trajectory in PC space.
    dt : float
        Time step size in ms.
    eps : float

    Returns
    -------
    dict with keys:
        I_par         : time-integrated parallel component (scalar)
        I_norm        : time-integrated perpendicular component (scalar)
        dtheta        : integrated angular deflection (radians)
        par_mean      : I_par normalised by duration
        norm_mean     : I_norm normalised by duration
        par_per_len   : I_par normalised by path length
        norm_per_len  : I_norm normalised by path length
        dv_par_series : parallel Δv over time (array)
        dv_norm_series: perpendicular Δv over time (array)
        time          : time axis used for integration (array)
    """
    v0 = _interpolate_vectors(*_build_interpolators(X1, X2, U0, V0), traj2d)
    v1 = _interpolate_vectors(*_build_interpolators(X1, X2, U1, V1), traj2d)
    dv = v1 - v0

    t_hat, n_hat, base_path_rate = _trajectory_tangent_normal(traj2d, eps=eps)
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
        par_mean=I_par / T,    norm_mean=I_norm / T,
        par_per_len=I_par / L, norm_per_len=I_norm / L,
        dv_par_series=dv_par,  dv_norm_series=dv_norm,
        time=time,
    )