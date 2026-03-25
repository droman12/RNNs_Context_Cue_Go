import numpy as np
from plotting.plot_trajectories import plot_current 

def subspace_decomposition(
    hid: np.ndarray,
    ip_idx: np.ndarray,
    n_cond: int = 16,
    burn_in: int = 100,
    var_thresh: float = 0.98,
    center_units: bool = True,
    subspace_fit: str = "cond_avg",  # "cond_avg" or "all"
):
    """
    Decompose RNN hidden activity into time-shared and stimulus-specific subspaces.

    Args:
        hid:          (T, C, n_units) — hidden states, C trials in interleaved
                      condition order (as produced by data_generation.get_inputs).
        ip_idx:       (C,)            — condition index per trial (interleaved).
        n_cond:       number of conditions.
        burn_in:      time steps to discard from the start.
        var_thresh:   cumulative variance threshold for selecting time-subspace dims.
        center_units: if True, subtract each unit's mean before SVD.
        subspace_fit: how to construct the matrix for SVD:
                      "cond_avg" — use condition-averaged activity (n_units, T') [default]
                      "all"      — use all conditions concatenated (n_units, T'*n_cond)

    Returns:
        R_time_mean : (n_units, T', 1)       shared mean across conditions
        R_time_res  : (n_units, T', n_cond)  condition residuals in time subspace
        R_time_full : (n_units, T', n_cond)  = R_time_mean + R_time_res
        R_stim_sub  : (n_units, T', n_cond)  activity in stimulus subspace
        E_mean      : (n_units,)  energy of mean (scaled so E_mean+E_res+E_stim=E_total)
        E_res       : (n_units,)  ||R_time_res||_F^2
        E_stim      : (n_units,)  ||R_stim_sub||_F^2
        P_t         : (n_units, n_units)  projector onto time subspace
        Q           : (n_units, n_units)  projector onto stimulus subspace
        k_t         : int  number of time-subspace dimensions kept

    Notes:
        Energy partition (exact):
            E_mean + E_res + E_stim == E_total
        where E_total = sum(R_us**2, axis=(1,2)).

        This holds because:
          1. P_t and Q are orthogonal complements =>
               ||R_us||^2 = ||R_time_full||^2 + ||R_stim_sub||^2
          2. R_time_mean and R_time_res are orthogonal (mean vs zero-mean) =>
               ||R_time_full||^2 = n_cond*||R_time_mean||^2 + ||R_time_res||^2
    """
    T, C, n_units = hid.shape

    if C % n_cond != 0:
        raise ValueError(f"C={C} is not divisible by n_cond={n_cond}.")
    if len(ip_idx) != C:
        raise ValueError(f"ip_idx length {len(ip_idx)} must equal C={C}.")
    if not (0 <= burn_in < T):
        raise ValueError(f"burn_in must be in [0, {T-1}], got {burn_in}.")
    if not (0 < var_thresh <= 1):
        raise ValueError(f"var_thresh must be in (0, 1], got {var_thresh}.")
    if subspace_fit not in ("cond_avg", "all"):
        raise ValueError("subspace_fit must be 'cond_avg' or 'all'.")

    n_reps = C // n_cond

    # --- Sort trials into block order so reshape assigns conditions correctly ---
    sort_idx   = np.argsort(ip_idx, kind='stable')
    hid_sorted = hid[:, sort_idx, :]

    # (T, n_cond, n_reps, n_units) -> drop burn-in -> average over reps
    R_speed = hid_sorted.reshape(T, n_cond, n_reps, n_units)[burn_in:].mean(axis=2)
    # (n_units, T', n_cond)
    R_us = np.transpose(R_speed, (2, 0, 1))

    T_prime  = R_us.shape[1]
    zeros_u  = np.zeros(n_units)
    eye_u    = np.eye(n_units)
    zeros_TC = np.zeros((n_units, T_prime, n_cond))

    if np.allclose(R_us, 0.0):
        return (
            np.zeros((n_units, T_prime, 1)),
            zeros_TC, zeros_TC, zeros_TC,
            zeros_u, zeros_u, zeros_u,
            eye_u, eye_u, 0,
        )

    if subspace_fit == "cond_avg":
        R_fit = R_us.mean(axis=2)           # (n_units, T')
    else:  # "all"
        R_fit = R_us.reshape(n_units, -1)   # (n_units, T'*n_cond)

    if center_units:
        R_fit = R_fit - R_fit.mean(axis=1, keepdims=True)

    U_t, S_t, _ = np.linalg.svd(R_fit, full_matrices=False)
    ss2 = np.sum(S_t ** 2)
    if ss2 > 0:
        cume = np.cumsum(S_t ** 2) / ss2
        k_t  = int(np.searchsorted(cume, var_thresh) + 1)
    else:
        k_t = 1
    k_t = max(1, min(k_t, U_t.shape[1]))

    L_t = U_t[:, :k_t]
    P_t = L_t @ L_t.T
    P_t = 0.5 * (P_t + P_t.T)              # symmetrize for numerical stability
    Q   = np.eye(n_units) - P_t
    Q   = 0.5 * (Q + Q.T)

    R_time_full = np.tensordot(P_t, R_us, axes=([1], [0]))   # (n_units, T', n_cond)
    R_stim_sub  = np.tensordot(Q,   R_us, axes=([1], [0]))   # (n_units, T', n_cond)

    R_time_mean = R_time_full.mean(axis=2, keepdims=True)    # (n_units, T', 1)
    R_time_res  = R_time_full - R_time_mean                  # (n_units, T', n_cond)

    E_mean = np.sum(R_time_mean ** 2, axis=(1, 2)) * n_cond  # (n_units,)
    E_res  = np.sum(R_time_res  ** 2, axis=(1, 2))           # (n_units,)
    E_stim = np.sum(R_stim_sub  ** 2, axis=(1, 2))           # (n_units,)

    E_total = np.sum(R_us ** 2, axis=(1, 2))
    assert np.allclose(E_mean + E_res + E_stim, E_total, rtol=1e-4), \
        "Energy partition violated — check projection logic."

    return R_time_mean, R_time_res, R_time_full, R_stim_sub, \
           E_mean, E_res, E_stim, P_t, Q, k_t


def current(
    model,
    hid: np.ndarray,
    ip_idx: np.ndarray,
    n_cond: int = 16,
    kind: str = "out",          # renamed from 'type' (shadows builtin)
    **decomp_kwargs,            # forwarded to subspace_decomposition
):
    """
    Project readout (kind='out') or input (kind='in') weights onto the
    time-mean, time-residual and stimulus subspaces, then plot.

    Args:
        model:         Run_Model instance.
        hid:           (T, C, n_units) hidden-state array.
        ip_idx:        (C,) condition index per trial.
        n_cond:        number of conditions.
        kind:          'out' — project readout weights (n_out, n_units)
                       'in'  — project input weights   (n_units, n_in)
        **decomp_kwargs: passed through to subspace_decomposition
                       (e.g. var_thresh, center_units, subspace_fit).

    Returns:
        I_time, I_res, I_stim
    """
    # Single decomposition — R_time_full and R_time_res are guaranteed consistent
    R_time_mean, R_time_res, R_time_full, R_stim_sub, \
        E_mean, E_res, E_stim, P_t, Q, k_t = subspace_decomposition(
            hid, ip_idx, n_cond=n_cond, **decomp_kwargs
        )

    if kind == "out":
        W = model.model.readout.W.detach().cpu().numpy()    # (n_out, n_units)
        # Contract over n_units: axis 1 of W with axis 0 of R_*
        I_time = np.tensordot(W, R_time_full, axes=([1], [0])).squeeze(0)  # (T', n_cond)
        I_stim = np.tensordot(W, R_stim_sub,  axes=([1], [0])).squeeze(0)
        I_res  = np.tensordot(W, R_time_res,  axes=([1], [0])).squeeze(0)
        plot_current(I_time, I_res, I_stim)

    elif kind == "in":
        W = model.model.rnn.rnncell.weight_ih.detach().cpu().numpy()  # (n_units, n_in)
        # Contract over n_units: axis 0 of W with axis 0 of R_*
        I_time = np.tensordot(W, R_time_full, axes=([0], [0]))  # (n_in, T', n_cond)
        I_stim = np.tensordot(W, R_stim_sub,  axes=([0], [0]))
        I_res  = np.tensordot(W, R_time_res,  axes=([0], [0]))
        for i in range(W.shape[1]):
            plot_current(I_time[i], I_res[i], I_stim[i])

    else:
        raise ValueError(f"kind must be 'out' or 'in', got '{kind}'.")

    return I_time, I_res, I_stim