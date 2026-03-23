"""
analysis/metrics.py
--------------------
Functions for decoding behavioural readouts from model outputs and computing
performance metrics.

Conventions
-----------
- `output` / `target`: np.ndarray of shape (T, B, D), where T = time steps,
  B = trials, D = output dimensions.
- `set_onset`: time index of the "set" signal, used as the reference point
  for computing relative produced/target times.
- `threshold`: scalar value that the output must reach or exceed to count
  as a response.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np


# Time decoding

def decode_time(
    sequence: np.ndarray,
    set_onset: int,
    threshold: float = 1.0,
    burn_in: int = 50,
) -> List[Optional[int]]:
    """Find the first time each trial's output crosses a threshold.

    Parameters
    ----------
    sequence : np.ndarray, shape (T, B, D)
        Model output or target sequence.
    set_onset : int
        Time index of the set signal. The returned times are expressed
        relative to this index.
    threshold : float
        Crossing threshold applied across all output dimensions.
    burn_in : int
        Number of initial timesteps to ignore (avoids spurious early crossings).

    Returns
    -------
    produced_times : list of int or None, length B
        Relative crossing time for each trial, or None if no crossing occurred.
    """
    produced_times: List[Optional[int]] = []

    for trial in range(sequence.shape[1]):
        pred = sequence[:, trial, :]                       # (T, D)
        crossings = np.nonzero(pred >= threshold)[0]       # timestep indices

        if crossings.size:
            t = int(crossings[0]) - set_onset - burn_in
            produced_times.append(t)
        else:
            produced_times.append(None)

    return produced_times


# Performance metrics

def performance_mape(
    output: np.ndarray,
    target: np.ndarray,
    set_onset: int,
    threshold: float = 1.0,
) -> float:
    """Scalar performance based on mean absolute percentage error (MAPE).

    Defined as max(0, 1 - MAPE), so that a perfect match gives 1.0 and
    large errors approach 0.0.

    Parameters
    ----------
    output : np.ndarray, shape (T, B, D)
    target : np.ndarray, shape (T, B, D)
    set_onset : int
    threshold : float

    Returns
    -------
    perf : float in [0, 1]
    """
    Tp = np.array([t if t is not None else -1
                   for t in decode_time(output, set_onset, threshold)])
    Ts = np.array([t if t is not None else -1
                   for t in decode_time(target, set_onset, threshold)])

    mask = (Tp >= 0) & (Ts > 0)
    if mask.sum() < 1:
        return 0.0

    ape = np.abs(Tp[mask] - Ts[mask]) / Ts[mask]
    return float(np.clip(1.0 - np.mean(ape), 0.0, 1.0))


def performance_nrmse(
    output: np.ndarray,
    target: np.ndarray,
    set_onset: int,
    threshold: float = 1.0,
    eps: float = 1e-8,
) -> float:
    """Scalar performance based on normalised RMSE (NRMSE).

    Defined as 1 - RMSE / range(Ts), clipped to [0, 1].
    A perfect match gives 1.0; RMSE >= range(Ts) gives 0.0.

    Parameters
    ----------
    output : np.ndarray, shape (T, B, D)
    target : np.ndarray, shape (T, B, D)
    set_onset : int
    threshold : float
    eps : float
        Small constant to avoid division by zero when all Ts are identical.

    Returns
    -------
    perf : float in [0, 1]
    """
    Tp = np.array([t if t is not None else -1
                   for t in decode_time(output, set_onset, threshold)])
    Ts = np.array([t if t is not None else -1
                   for t in decode_time(target, set_onset, threshold)])

    mask = (Ts > 0) & (Tp >= 0)
    if mask.sum() < 1:
        return 0.0

    Tp_v, Ts_v = Tp[mask], Ts[mask]
    rmse = np.sqrt(np.mean((Tp_v - Ts_v) ** 2))
    ts_range = Ts_v.max() - Ts_v.min()

    if ts_range < eps:
        # All target times are identical; normalise by the target magnitude.
        return float(np.clip(1.0 - rmse / (Ts_v.max() + eps), 0.0, 1.0))

    return float(np.clip(1.0 - rmse / (ts_range + eps), 0.0, 1.0))


def performance_ratio(
    produced: List[Optional[float]],
    targets: List[float],
) -> float:
    """Scalar performance as mean min/max ratio between produced and target times.

    Each trial contributes min(Tp, Ts) / max(Tp, Ts), which equals 1 for a
    perfect match and approaches 0 for large mismatches. Trials where
    `produced` is None score 0.

    Parameters
    ----------
    produced : list of float or None, length B
    targets : list of float, length B

    Returns
    -------
    perf : float in [0, 1]
    """
    if len(produced) != len(targets):
        raise ValueError("produced and targets must have the same length.")

    scores = []
    for p, t in zip(produced, targets):
        if t <= 0:
            raise ValueError(f"Target time must be > 0, got {t}.")
        scores.append(0.0 if p is None else min(p, t) / max(p, t))

    return sum(scores) / len(targets)


# Trajectory speed

def compute_trajectory_speed(
    pca_trajectories: np.ndarray,
    dt: float = 1.0,
) -> np.ndarray:
    """Compute the per-trial average speed along PCA trajectories.

    Speed is estimated via central differences of the trajectory coordinates,
    then averaged over time for each trial. The result is normalised to [0, 1].

    Parameters
    ----------
    pca_trajectories : np.ndarray, shape (T, B, K)
        PCA-projected hidden states (time × trials × components).
    dt : float
        Time step size in ms.

    Returns
    -------
    norm_avg_speed : np.ndarray, shape (B,)
        Normalised average speed per trial.
    """
    T, B, K = pca_trajectories.shape
    diffs = np.zeros_like(pca_trajectories)

    # Forward difference at t=0, backward at t=T-1, central elsewhere
    diffs[0]    = pca_trajectories[1] - pca_trajectories[0]
    diffs[-1]   = pca_trajectories[-1] - pca_trajectories[-2]
    diffs[1:-1] = (pca_trajectories[2:] - pca_trajectories[:-2]) / (2.0 * dt)

    speed = np.linalg.norm(diffs, axis=2)           # (T, B)
    avg_speed = speed.mean(axis=0)                  # (B,)

    speed_range = avg_speed.max() - avg_speed.min()
    if speed_range == 0:
        return np.zeros(B)
    return (avg_speed - avg_speed.min()) / speed_range