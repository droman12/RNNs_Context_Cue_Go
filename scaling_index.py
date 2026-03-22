import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d
import torch
from model import *
from hp import *
from plotting_functions import *
import numpy as np
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def scaling_index(short_profile, long_profile, Tp_short, Tp_long, plot=True, demean=True, time_label="Time"):
    """
    Compare 'short' vs 'long' firing-rate profiles by time-compressing the long one
    so its duration matches Tp_short, then compute R^2.

    Parameters
    ----------
    short_profile : array-like
        Samples over the short interval (duration Tp_short).
    long_profile : array-like
        Samples over the long interval (duration Tp_long).
    Tp_short : float
        Duration of the short condition (use consistent units, e.g., ms or s).
    Tp_long : float
        Duration of the long condition (same units as Tp_short).
    plot : bool
        If True, show the profiles and the rescaled long trace.
    demean : bool
        If True, subtract each profile’s mean before comparison.
    time_label : str
        Label for the x-axis (e.g., "Time (ms)" or "Time (s)").
    """
    short_profile = np.asarray(short_profile)
    long_profile  = np.asarray(long_profile)

    if demean:
        short_profile = short_profile - short_profile.mean()
        long_profile  = long_profile  - long_profile.mean()

    # Time axes in real units, not normalized
    t_short = np.linspace(0, Tp_short, len(short_profile), endpoint=True)
    t_long  = np.linspace(0, Tp_long,  len(long_profile),  endpoint=True)

    # Compression factor to map long duration onto short duration
    scale = Tp_short / Tp_long

    # Rescale long profile in time so its domain becomes [0, Tp_short]
    t_long_scaled = t_long * scale

    # Interpolate rescaled long onto the short time grid
    f_long_interp = interp1d(
        t_long_scaled, long_profile, kind="linear", bounds_error=False, fill_value="extrapolate"
    )
    long_resampled = f_long_interp(t_short)

    # R^2 over the full short interval
    r2 = r2_score(short_profile, long_resampled)

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(t_short, short_profile, label=f"Short (Tp={Tp_short})", lw=2)
        plt.plot(t_long, long_profile, label=f"Long (Tp={Tp_long}) original", lw=2, alpha=0.6)
        plt.plot(t_short, long_resampled, '--', label="Long rescaled → short", lw=2)
        plt.xlabel(time_label)
        plt.ylabel("Firing Rate (a.u.)")
        plt.title(f"Scaling Index (R²) = {r2:.3f}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return r2
def amplitude_gain(short_profile, long_profile, Tp_short, Tp_long,
                   demean=True, fit_offset=False):
    """
    Compute amplitude scaling g (and optional offset b) after the SAME
    time alignment your scaling_index() uses (rescale long -> short).

    Returns: g, b, R2_amp  where R2_amp is the R^2 of y ≈ g*x + b
    """
    short = np.asarray(short_profile)
    long  = np.asarray(long_profile)

    if demean:
        short = short - short.mean()
        long  = long  - long.mean()

    # same real-time axes + rescale of the long trace as in scaling_index()
    t_short = np.linspace(0, Tp_short, len(short), endpoint=True)
    t_long  = np.linspace(0, Tp_long,  len(long),  endpoint=True)
    scale = Tp_short / Tp_long
    f_long = interp1d(t_long * scale, long, kind="linear",
                      bounds_error=False, fill_value="extrapolate")

    x = f_long(t_short)   # long aligned to short timeline
    y = short

    if fit_offset:
        A = np.column_stack([x, np.ones_like(x)])
        g, b = np.linalg.lstsq(A, y, rcond=None)[0]
    else:
        # gain-only fit (good default when demean=True)
        denom = float(np.dot(x, x)) + 1e-12
        g = float(np.dot(x, y) / denom)
        b = 0.0

    y_hat = g * x + b
    R2_amp = r2_score(y, y_hat)
    return g, b, R2_amp


# ---------- Utility: safe crop ----------
def safe_crop(x, start, length):
    """Return x[start : start+length] with bounds safely clamped."""
    n = len(x)
    s = max(0, int(round(start)))
    e = min(n, max(s+1, int(round(start + length))))  # at least 1 sample
    return x[s:e]

# ---------- Main: compute SI per unit ----------
def scaling_index_per_unit(
    hid,                   # shape: [time, trial, hid_dim]
    pred,                  # whatever decode_time() needs
    unit_list,             # iterable of unit indices into last dim
    set_onset=1600,
    set_end=1650,
    threshold=1.0,
    short_trials=slice(0,7),
    long_trials=slice(7,None),
    demean=False,   # function: decode_time(pred, set_onset, threshold) -> Tp per trial
):
    
    Tp = decode_time(pred, set_onset=set_onset, threshold=threshold)  # shape: [trials]
    Tp_short = float(np.mean(Tp[short_trials]))
    Tp_long  = float(np.mean(Tp[long_trials]))

    results = []
    T, _, H = hid.shape

    for un_idx in unit_list:
        if not (0 <= un_idx < H):
            continue

        # 2) mean across trials to get one profile per condition
        hid_short = np.mean(hid[:, short_trials, un_idx], axis=1)  # shape [time]
        hid_long  = np.mean(hid[:, long_trials,  un_idx], axis=1)  # shape [time]

        # 3) crop each profile from set_end for the duration Tp_*
        prof_short = safe_crop(hid_short, start=set_end, length=Tp_short)
        prof_long  = safe_crop(hid_long,  start=set_end, length=Tp_long)

        # Edge case: if either crop is too short, skip
        if len(prof_short) < 2 or len(prof_long) < 2:
            r2 = np.nan
        else:
            r2 = scaling_index(prof_short, prof_long, Tp_short, Tp_long, demean=demean, plot= False)
            g, b, r2_amp = amplitude_gain(prof_short, prof_long, Tp_short, Tp_long)

        results.append({
            'unit': un_idx,
            'Tp_short': Tp_short,
            'Tp_long': Tp_long,
            'r2': r2,
            'len_short': len(prof_short),
            'len_long': len(prof_long),
            'g_ampl': g,
            'gain_change': np.abs(g-1), 
            'bias': b, 
            'r2_improv': r2_amp-r2, 
            'r2_amp': r2_amp
        })

    return results


