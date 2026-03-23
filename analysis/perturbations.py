"""
analysis/perturbation.py
-------------------------
Functions for applying input perturbations to a trained model and collecting
behavioural metrics (produced times, Ts/Tp slopes) across repeats.

Typical workflow
----------------
1. Call `generate_perturbation_results` to run the model under perturbation.
2. Save / load results with `save_results` / `load_results`.
3. Pass results to the plotting functions in plotting/plot_perturbation.py.

Result format
-------------
Single-window mode (multi_window=False):
    results[rep] = {
        'predictions' : {mag: np.ndarray(B,)},
        'mean_times'  : [float, ...],   # one per magnitude
        'slopes'      : [float, ...],
    }

Multi-window mode (multi_window=True):
    results[rep] = {
        start: {
            'predictions' : {mag: np.ndarray(B,)},
            'mean_times'  : [float, ...],
            'slopes'      : [float, ...],
        }
        for start in perturb_starts
    }
"""

import pickle

import numpy as np
import torch
from scipy import stats
from metrics import decode_time


def save_results(results, filename):
    with open(filename, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results to {filename!r}")


def load_results(filename):
    with open(filename, "rb") as f:
        results = pickle.load(f)
    print(f"Loaded results from {filename!r}")
    return results


# Perturbation helpers


def apply_perturbation(inputs, channel=0, start_idx=50, end_idx=350, magnitude=1.0):
    """Add a constant offset to one input channel over a time window.

    Parameters
    ----------
    inputs : torch.Tensor, shape (T, B, C)
        Model input tensor (time × batch × channels).
    channel : int
        Index of the channel to perturb.
    start_idx : int
        First timestep (inclusive) of the perturbation window.
    end_idx : int
        Last timestep (exclusive) of the perturbation window.
    magnitude : float
        Additive offset applied to the selected channel.

    Returns
    -------
    torch.Tensor
        Cloned input with the perturbation applied.
    """
    perturbed = inputs.clone()
    perturbed[start_idx:end_idx, :, channel] += magnitude
    return perturbed


def silence_channel(inputs, channel=0, t_start=50, t_end=350):
    """Set one input channel to zero over a time window.

    Parameters
    ----------
    inputs : torch.Tensor, shape (T, B, C)
        Model input tensor (time × batch × channels).
    channel : int
        Index of the channel to silence.
    t_start : int
        First timestep (inclusive) of the silencing window.
    t_end : int
        Last timestep (exclusive) of the silencing window.

    Returns
    -------
    torch.Tensor
        Cloned input with the selected channel zeroed.
    """
    silenced = inputs.clone()
    silenced[t_start:t_end, :, channel] = 0
    return silenced


# Main analysis function

def generate_perturbation_results(
    model,
    perturb_magnitudes,
    n_repeats,
    batch_size,
    threshold=1.0,
    set_onset=400,
    multi_window=False,
    perturb_starts=None,
    perturb_ends=None,
    start=50,
    window_length=300,
):
    """Run the model under additive input perturbations and collect behavioural metrics.

    A single fixed test batch is generated once and reused across all repeats
    and magnitudes, so variability across repeats reflects model stochasticity
    only (e.g. dropout), not trial sampling.

    Parameters
    ----------
    model : object
        Trained model with a `generate_trials(batch_size)` method and a
        callable interface `model(X, Y)` returning `(total, data, reg, pred, hid)`.
    perturb_magnitudes : list of float
        Additive offsets to apply to channel 0.
    n_repeats : int
        Number of independent forward passes per condition.
    batch_size : int
        Number of trials per test batch.
    threshold : float
        Threshold passed to `decode_time` for reading out produced times.
    set_onset : int
        Time of the set signal (ms), passed to `decode_time`.
    multi_window : bool
        If True, sweep over multiple perturbation windows defined by
        `perturb_starts` and `perturb_ends`.
        If False, use a single window defined by `start` and `window_length`.
    perturb_starts : list of int, optional
        Window onset times (ms). Required when multi_window=True.
    perturb_ends : list of int, optional
        Window offset times (ms). Required when multi_window=True.
        Must have the same length as `perturb_starts`.
    start : int
        Onset of the single perturbation window (ms). Used when multi_window=False.
    window_length : int
        Duration of the single perturbation window (ms). Used when multi_window=False.

    Returns
    -------
    results : list of dict
        Length n_repeats. Structure depends on `multi_window`; see module docstring.
    targ_times : np.ndarray, shape (B,)
        Target produced times decoded from the unperturbed labels.
    """
    X_test, Y_test = model.generate_trials(batch_size=batch_size)
    Y_np = Y_test.detach().cpu().numpy()
    targ_times = np.array(
        [t if t is not None else -1 for t in decode_time(Y_np, set_onset, threshold)]
    )

    if multi_window:
        results = _run_multi_window(
            model, X_test, Y_test, targ_times,
            perturb_magnitudes, perturb_starts, perturb_ends, n_repeats,
            set_onset, threshold,
        )
    else:
        results = _run_single_window(
            model, X_test, Y_test, targ_times,
            perturb_magnitudes, start, start + window_length, n_repeats,
            set_onset, threshold,
        )

    return results, targ_times


def compute_recovery_times(
    norms_full_arr,
    time_axis,
    perturb_start,
    perturb_end,
    perturb_magnitudes,
    threshold_frac=0.2,
):
    """Estimate how long the trajectory takes to recover after each perturbation.

    For each magnitude, the recovery time is defined as the first timepoint
    *after* the perturbation window at which the mean normalised displacement
    drops below baseline + threshold_frac * (peak - baseline).

    Parameters
    ----------
    norms_full_arr : np.ndarray, shape (M, T, B)
        Normalised displacement traces for M magnitudes, T timepoints, B trials.
        Typically the output of `plot_perturbation.plot_displacement_over_time`.
    time_axis : np.ndarray, shape (T,)
        Time labels (ms) corresponding to axis 1 of `norms_full_arr`.
    perturb_start : int
        Index into `time_axis` where the perturbation begins.
    perturb_end : int
        Index into `time_axis` where the perturbation ends.
    perturb_magnitudes : list of float
        Perturbation magnitudes corresponding to axis 0 of `norms_full_arr`.
    threshold_frac : float
        Recovery threshold as a fraction of peak displacement above baseline.

    Returns
    -------
    recovery_durations : np.ndarray, shape (M,)
        Elapsed time from perturbation offset to recovery (ms).
        np.nan if the signal never recovers within the recording.
    """
    M, T, B = norms_full_arr.shape
    t_end = time_axis[perturb_end]
    recovery_durations = []

    for m_idx in range(M):
        mean_curve = norms_full_arr[m_idx].mean(axis=1)  # (T,)

        baseline = mean_curve[:perturb_start].mean()
        peak     = mean_curve[perturb_start:perturb_end].max()
        thresh   = baseline + (peak - baseline) * threshold_frac

        post  = mean_curve[perturb_end:]
        below = np.where(post <= thresh)[0]

        elapsed = time_axis[perturb_end + below[0]] - t_end if below.size else np.nan
        recovery_durations.append(elapsed)

    return np.array(recovery_durations)


def _run_single_window(
    model, X_test, Y_test, targ_times,
    perturb_magnitudes, start, end, n_repeats,
    set_onset, threshold,
):
    """Inner loop for a single perturbation window."""
    overall = []
    for _ in range(n_repeats):
        rep_data = {
            "predictions": {mag: None for mag in perturb_magnitudes},
            "mean_times": [],
            "slopes": [],
        }
        for mag in perturb_magnitudes:
            perturbed = apply_perturbation(X_test, channel=0,
                                           start_idx=start, end_idx=end,
                                           magnitude=mag)
            with torch.no_grad():
                *_, pred, _ = model(X=perturbed, Y=Y_test)

            pred_times = np.array(
                [t if t is not None else -1
                 for t in decode_time(pred, set_onset, threshold)]
            )
            slope, *_ = stats.linregress(targ_times, pred_times)

            rep_data["predictions"][mag] = pred_times
            rep_data["mean_times"].append(pred_times.mean())
            rep_data["slopes"].append(slope)

        overall.append(rep_data)
    return overall


def _run_multi_window(
    model, X_test, Y_test, targ_times,
    perturb_magnitudes, perturb_starts, perturb_ends, n_repeats,
    set_onset, threshold,
):
    """Inner loop for multiple perturbation windows."""
    overall = []
    for _ in range(n_repeats):
        rep_data = {
            start: {
                "predictions": {mag: None for mag in perturb_magnitudes},
                "mean_times": [],
                "slopes": [],
            }
            for start in perturb_starts
        }
        for start, end in zip(perturb_starts, perturb_ends):
            for mag in perturb_magnitudes:
                perturbed = apply_perturbation(X_test, channel=0,
                                               start_idx=start, end_idx=end,
                                               magnitude=mag)
                with torch.no_grad():
                    *_, pred, _ = model(X=perturbed, Y=Y_test)

                pred_times = np.array(
                    [t if t is not None else -1
                     for t in decode_time(pred, set_onset, threshold)]
                )
                slope, *_ = stats.linregress(targ_times, pred_times)

                rep_data[start]["predictions"][mag] = pred_times
                rep_data[start]["mean_times"].append(pred_times.mean())
                rep_data[start]["slopes"].append(slope)

        overall.append(rep_data)
    return overall