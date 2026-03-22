
import numpy as np
import matplotlib.pyplot as plt
import torch
from plotting_functions import *
import pickle

from scipy.spatial.distance import pdist



start_color = (1, 0.549, 0)
end_color = (0.392, 0.584, 0.929)
custom_cmap = LinearSegmentedColormap.from_list('custom_gradient', [start_color, end_color])
def save_results(results, filename):
    """
    Serializes `results` (the output of generate_perturbation_results)
    to a file so you can reload it later without recomputing.
    """
    with open(filename, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results to {filename!r}")

def load_results(filename):
    """
    Loads back the results dicts from disk.
    """
    with open(filename, "rb") as f:
        results = pickle.load(f)
    print(f"Loaded results from {filename!r}")
    return results



def set_to_0(inputs, channel=0, t_start=50, t_end=350):
    """
    Applies a constant perturbation of given magnitude to a specified channel between start_idx and end_idx.
    """
    perturbed_inputs = inputs.clone()
    perturbed_inputs[t_start:t_end, :, channel] =0
    return perturbed_inputs


def apply_perturbation(inputs, channel=0, start_idx=50, end_idx=350, magnitude=1):
    """
    Applies a constant perturbation of given magnitude to a specified channel between start_idx and end_idx.
    """
    perturbed_inputs = inputs.clone()
    perturbed_inputs[start_idx:end_idx, :, channel] += magnitude
    return perturbed_inputs

def generate_perturbation_results_original(
    run_model, threshold, set_onset,
    perturb_starts, perturb_magnitudes,
    perturb_ends, n_repeats, batch_size
):
    """
    Runs multiple repeats of perturbations over specified starts and magnitudes.
    Returns a list (length n_repeats) of dicts mapping
      rep_data[start]['predictions'][mag] → pred_times (array)
      rep_data[start]['mean_times'] → list of means
      rep_data[start]['slopes']     → list of slopes
    """
    # 1) grab one fixed test‐batch
    X_test, Y_test = run_model.generate_trials(batch_size=batch_size)
    # move to numpy for decode_time
    Y_np = Y_test.detach().cpu().numpy()
    # decode the *clean* target times
    targ_times = decode_time(Y_np, set_onset, threshold)
    targ_times = np.array([t if t is not None else -1 for t in targ_times])

    overall = []
    for rep in range(n_repeats):
        # prepare storage
        rep_data = {
            start: {
                'predictions': {mag: None for mag in perturb_magnitudes},
                'mean_times': [],
                'slopes': [], 
                'performance': []
            }
            for start in perturb_starts
        }

        for start, end in zip(perturb_starts, perturb_ends):

            for mag in perturb_magnitudes:
                # 2) apply your perturbation
                perturbed = apply_perturbation(
                    X_test, channel=0,
                    start_idx=start, end_idx=end,
                    magnitude=mag
                )

                with torch.no_grad():
                    _, _, _, pred_np, _ = run_model(X=perturbed, Y=Y_test)

                # 4) decode & regress
                pred_times = decode_time(pred_np, set_onset, threshold)
                pred_times = np.array([t if t is not None else -1 for t in pred_times])
                #Ts = np.array([t if t is not None else -1 for t in Ts])
                slope, _, _, _, _ = stats.linregress(targ_times, pred_times)

                #perf = performance_nrmse(pred_np, Y_np, set_onset, threshold)

                # 5) stash results
                rep_data[start]['predictions'][mag] = pred_times
                rep_data[start]['mean_times'].append(pred_times.mean())
                rep_data[start]['slopes'].append(slope)
                #rep_data[start]['performance'].append(perf)

        overall.append(rep_data)

    return overall

def perturbation_results_new_task(model, perturb_magnitudes, n_repeats, batch_size, threshold=1.0, set_onset = 400, start = 50, window_length = 300):
    X_test, Y_test = model.generate_trials(batch_size=batch_size)
    Y_test_numpy = Y_test.detach().cpu().numpy()
    targ_times = decode_time(Y_test_numpy, set_onset, threshold)
    print(targ_times)

    overall = []
    end = start+ window_length
    for rep in range(n_repeats):
        rep_data = {'predictions' : {mag: [] for mag in perturb_magnitudes}, 
                    'mean_times': [], 'slopes': [], 'performance': []}
        for mag in perturb_magnitudes: 
            perturbed = apply_perturbation(X_test, channel = 0, start_idx = start, end_idx = end, magnitude = mag)
            with torch.no_grad():
                total, data, reg, pred, hid = model(X = perturbed, Y = Y_test)
            pred_times = decode_time(pred, set_onset, threshold)
            slope, _, _, _, _ = stats.linregress(targ_times, pred_times)
            # perf = performance_nrmse(pred, Y_test_numpy, set_onset, threshold)
            rep_data['predictions'][mag] = pred_times
            rep_data['mean_times'].append(np.mean(pred_times))
            rep_data['slopes'].append(slope)
            # rep_data['performance'].append(perf)
        overall.append(rep_data)
    return overall


def plot_mean_times_vs_magnitude_original(
    results,
    perturb_starts,
    perturb_magnitudes,
    cmap_name='BuGn'
):
    """
    Plot mean produced times ± std across repeats for each perturbation window,
    styled to match the single-window version.
    """
    N = len(perturb_starts)
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0.3, 1.0, N))
    mags = np.array(perturb_magnitudes)

    # create figure exactly like your second plot
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)

    for i, start in enumerate(perturb_starts):
        # gather mean-times across repeats
        all_means = np.array([r[start]['mean_times'] for r in results])
        mu = all_means.mean(axis=0)
        sd = all_means.std(axis=0, ddof=1)

        ax.plot(mags, mu, '-o', markersize = 10, color=colors[i], label=f'{start} - {start + 300} ms')
        ax.fill_between(mags, mu - sd, mu + sd, alpha=0.5, color=colors[i])

    # tick styling
    ticks = mags
    ax.tick_params(axis='both', which='major',
                   labelsize=14, length=8, width=2, direction='out')
    ax.tick_params(axis='both', which='minor',
                   labelsize=12, length=4, width=1.5, direction='out')

    # remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # labels
    ax.set_xlabel('Perturbation magnitude', fontsize=16)
    ax.set_ylabel('Mean $T_p$ (ms)', fontsize=16)
    ax.set_xticks(ticks)



    fig.tight_layout()
    plt.show()

def plot_mean_times_vs_magnitude(results, perturb_magnitudes, cmap='Greys'):
    """
    Plot mean produced times +/- std across repeats.
    """
    mags = np.array(perturb_magnitudes)

    fig, ax = plt.subplots(figsize=(4,4), dpi=150)

    all_means = np.array([r['mean_times'] for r in results])
    mu = all_means.mean(axis=0)
    sd = all_means.std(axis=0, ddof=1)

    # Grey connecting line
    ax.plot(mags, mu, '-', color='grey', zorder=1)

    # Scatter points colored by magnitude
    sc = ax.scatter(
        mags, mu,
        c=mags,
        cmap=cmap,
        vmin=mags.min(), vmax=mags.max(),
        s = 80,
        edgecolor='black',
        zorder=2
    )

    # Std shading with light grey
    ax.fill_between(mags, mu-sd, mu+sd, color='grey', alpha=0.3, zorder=0)

    ax.tick_params(axis='both', which='major', labelsize=14,
                   length=8, width=2, direction='out')
    ax.tick_params(axis='both', which='minor', labelsize=12,
                   length=4, width=1.5, direction='out')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('Perturbation magnitude', fontsize=16)
    ax.set_ylabel('Mean $T_p$ (ms)', fontsize=16)

    fig.tight_layout()
    plt.show()


def plot_mean_performance_vs_magnitude_original(results, perturb_starts, perturb_magnitudes, cmap_name='BuGn'):
    """
    Plot mean performance +/- std across repeats.
    """
    N = len(perturb_starts)
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0.3, 1.0, N))
    mags = np.array(perturb_magnitudes)
    

    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)

    for i, start in enumerate(perturb_starts):
        all_means = np.array([r[start]['performance'] for r in results])
        mu = all_means.mean(axis=0)
        sd = all_means.std(axis=0, ddof=1)
        ax.plot(mags, mu, '-o', markersize = 10, color=colors[i], label=f'{start} - {start + 300} ms')
        ax.fill_between(mags, mu - sd, mu + sd, alpha=0.5, color=colors[i])
    # tick styling
    ticks = mags
    ax.tick_params(axis='both', which='major',
                   labelsize=14, length=8, width=2, direction='out')
    ax.tick_params(axis='both', which='minor',
                   labelsize=12, length=4, width=1.5, direction='out')

    # remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # labels
    ax.set_xlabel('Perturbation magnitude', fontsize=16)
    ax.set_ylabel('Mean performance', fontsize=16)
    ax.set_xticks(ticks)


    fig.tight_layout()
    plt.show()



def plot_mean_performance_vs_magnitude(results, perturb_magnitudes, cmap='Greys'):
    """
    Plot mean performance +/- std across repeats.
    """
    base_cmap = plt.get_cmap(cmap)
    mag_colors = base_cmap(np.linspace(0.4,1.0,len(perturb_magnitudes)))
    mags = np.array(perturb_magnitudes)
    

    fig, ax = plt.subplots(figsize=(4,4), dpi = 150)
    all_means = np.array([r['performance'] for r in results])
    mu = all_means.mean(axis=0)
    sd = all_means.std(axis=0, ddof=1)
    ax.plot(mags, mu, '-o', color = 'indigo')
    ax.fill_between(mags, mu-sd, mu+sd, alpha=0.4, color = 'indigo')
    ax.tick_params(axis='both', which='major', labelsize=14,
                   length=8, width=2, direction='out')
    ax.tick_params(axis='both', which='minor', labelsize=12,
                   length=4, width=1.5, direction='out')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Perturbation magnitude', fontsize = 16)
    ax.set_ylabel('Mean performance', fontsize = 16)
    fig.tight_layout()
    plt.show()


def plot_slopes_vs_magnitude_original(
    results,
    perturb_starts,
    perturb_magnitudes,
    cmap_name='BuGn'
):
    """
    Plot regression slopes ±1 SD across repeats for each perturbation window,
    styled like the single-window version.
    """
    N = len(perturb_starts)
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0.3, 1.0, N))
    mags = np.array(perturb_magnitudes)

    # set up figure exactly like your other
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)

    for i, start in enumerate(perturb_starts):
        all_slopes = np.array([r[start]['slopes'] for r in results])
        mu = all_slopes.mean(axis=0)
        sd = all_slopes.std(axis=0, ddof=1)

        # mean ± SD band
        ax.plot(mags, mu, '-o', markersize = 10, color=colors[i], label=f'{start} - {start+300} ms')
        ax.fill_between(mags, mu - sd, mu + sd, alpha=0.4, color=colors[i])

    # axis labels
    ax.set_xlabel('Perturbation magnitude', fontsize=16)
    ax.set_ylabel('$T_s$/$T_p$ slope',         fontsize=16)

    # ticks
    ax.tick_params(axis='both', which='major',
                   labelsize=14, length=8, width=2, direction='out')
    ax.tick_params(axis='both', which='minor',
                   labelsize=12, length=4, width=1.5, direction='out')
    ticks = mags
    # remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(ticks)

    fig.tight_layout()
    plt.show()


def plot_slopes_vs_magnitude(results, perturb_magnitudes, cmap='Greys'):
    """
    Plot regression slopes +/- std across repeats for each perturbation window.
    """
    base_cmap = plt.get_cmap(cmap)
    mags = np.array(perturb_magnitudes)

    fig, ax = plt.subplots(figsize=(4,4), dpi=150)

    all_slopes = np.array([r['slopes'] for r in results])
    mu = all_slopes.mean(axis=0)
    sd = all_slopes.std(axis=0, ddof=1)

    # Plot neutral (grey) connecting line
    ax.plot(mags, mu, '-', color='grey', zorder=1)

    # Scatter points with facecolor by magnitude
    sc = ax.scatter(
        mags, mu, 
        c=mags, 
        cmap=cmap, 
        vmin=mags.min(), vmax=mags.max(),
        s = 80,
        edgecolor='black', 
        zorder=2
    )

    # Std shading with light grey
    ax.fill_between(mags, mu-sd, mu+sd, color='grey', alpha=0.3, zorder=0)

    ax.set_xlabel('Perturbation magnitude', fontsize=16)
    ax.set_ylabel('$T_s$/$T_p$ slope', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14,
                   length=8, width=2, direction='out')
    ax.tick_params(axis='both', which='minor', labelsize=12,
                   length=4, width=1.5, direction='out')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    plt.show()



def plot_scatter_produced_vs_target_original(
    results,
    perturb_starts,
    perturb_magnitudes,
    targ_times,
    custom_cmap = custom_cmap,
    cmap='Greys'
):
    """
    For each perturbation window in `perturb_starts`, plot:
      • scatter: mean produced vs target (colored by trial duration)
      • error‐bands: ±1 SD across repeats
      • one identity line
    Styled to match the single‐window version.
    """

    # colormap for different magnitudes
    base_cmap = plt.get_cmap(cmap)
    mag_colors = base_cmap(np.linspace(0.4, 1.0, len(perturb_magnitudes)))

    for start in perturb_starts:
        fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
        n_trials = len(targ_times)
        trial_pos = np.linspace(0, 1, n_trials)
        colors = custom_cmap(trial_pos)

        # plot each magnitude
        for i, mag in enumerate(perturb_magnitudes):
            # stack predictions across repeats
            stack   = np.vstack([r[start]['predictions'][mag] for r in results])
            mean_p  = stack.mean(axis=0)
            sd_p    = stack.std(axis=0, ddof=1)

            # mean line + error band
            ax.plot(targ_times, mean_p,
                    '-', lw=3.0, color=mag_colors[i], zorder = 1)
            ax.fill_between(targ_times,
                            mean_p - sd_p, mean_p + sd_p,
                            color=mag_colors[i], alpha=0.4)

            # scatter individual trials
            for j, t in enumerate(targ_times):

                ax.scatter(t, mean_p[j],
                           s=80,
                           c=colors[j],
                           edgecolors='k',
                           linewidth=0.4, zorder = 2)

        # identity
        mn, mx = np.min(targ_times), np.max(targ_times)
        ax.plot([mn, mx], [mn, mx],
                'k--', lw=1)

        ax.set_xlabel('Target Time $T_s$ (ms)', fontsize=16)
        ax.set_ylabel('Produced Time $T_p$ (ms)', fontsize=16)


        # tick styling
        ax.tick_params(axis='both', which='major',
                       labelsize=14, length=8, width=2, direction='out')
        ax.tick_params(axis='both', which='minor',
                       labelsize=12, length=4, width=1.5, direction='out')
        # remove top/right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ticks = np.array([450, 550, 720, 880])

        # apply to both axes:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        # style ticks:
        ax.tick_params(which='major', length=8, width=2)
        ax.tick_params(which='minor', length=4, width=1)

        fig.tight_layout()
        plt.show()

def plot_scatter_produced_vs_target(results, perturb_magnitudes,
                                    targ_times, custom_cmap = custom_cmap, cmap='Greys',
                                    show_colorbar=True):
    """
    For each perturbation window (magnitude), scatter produced vs target times
    colored by trial duration; neutral grey mean line with +/- SD band.
    """
    targ_times = np.asarray(targ_times)
    mn, mx = np.min(targ_times), np.max(targ_times)

    fig, ax = plt.subplots(figsize=(4,4), dpi=150)

    # Plot per-magnitude curves (neutral line + grey band) and colored scatters by trial duration
    for mag in perturb_magnitudes:
        stack = np.vstack([r['predictions'][mag] for r in results])  # (repeats x trials)
        mean_p = stack.mean(axis=0)
        sd_p   = stack.std(axis=0, ddof=1)

        # Grey mean line and SD band
        ax.plot(targ_times, mean_p, '-', lw=1.0, color='grey', zorder = 0)
        ax.fill_between(targ_times, mean_p-sd_p, mean_p+sd_p,
                        alpha=0.25, color='grey', zorder=1)

        # Scatter colored by trial duration (lighter = shorter)
        ax.scatter(targ_times, mean_p,
                   c=targ_times, cmap=custom_cmap,
                   vmin=mn, vmax=mx,
                   s=80, edgecolors='k', linewidth=0.5, zorder=2)

    # Identity line
    ax.plot([mn, mx], [mn, mx], 'k--', lw=0.6)

    # Axes & styling
    ax.set_xlabel('Target Time $T_s$ (ms)', fontsize=16)
    ax.set_ylabel('Produced Time $T_p$ (ms)', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14,
                   length=8, width=2, direction='out')
    ax.tick_params(axis='both', which='minor', labelsize=12,
                   length=4, width=1.5, direction='out')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Example ticks; adjust as needed
    ticks = np.array([450, 550, 720, 880])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    fig.tight_layout()
    plt.show()



def plot_normalized_displacement_over_time(
    hid_clean,            # np.ndarray, shape (T, B, D)
    hid_perturbed,        # np.ndarray, same shape
    pcs,                  # np.ndarray, shape (D, >=n_components)
    n_components=3,
    time_axis=None,
    figsize=(6, 4),
    dpi=300,
    color_full='black',
    color_pca='gray'
):
    """
    Compute and plot the normalized Euclidean displacement over time,
    in both the full D-dim neural space and the first n_components of PCA space.
    """
    T, B, D = hid_clean.shape
    if time_axis is None:
        time_axis = np.arange(T)

    # 1) Raw displacement in full space
    delta    = hid_clean - hid_perturbed             # (T, B, D)
    dist_full = np.linalg.norm(delta, axis=2)        # (T, B)

    # 2) Displacement in PCA space
    delta_flat = delta.reshape(-1, D)                                # (T*B, D)
    delta_pca  = delta_flat @ pcs[:, :n_components]                  # (T*B, n_components)
    dist_pca   = np.linalg.norm(delta_pca, axis=1).reshape(T, B)     # (T, B)

    # 3) Overall task diameter (max pairwise dist per trial → global max)
    #    Full space:
    diam_full = [
        pdist(hid_clean[:, b, :], metric='euclidean').max()
        for b in range(B)
    ]
    overall_diameter_full = max(diam_full)

    #    PCA space:
    baseline_pca = (
        hid_clean.reshape(-1, D) @ pcs[:, :n_components]
    ).reshape(T, B, n_components)
    diam_pca = [
        pdist(baseline_pca[:, b, :], metric='euclidean').max()
        for b in range(B)
    ]
    overall_diameter_pca = max(diam_pca)

    # 4) Normalize
    norm_full = dist_full / overall_diameter_full    # (T, B)
    norm_pca  = dist_pca  / overall_diameter_pca     # (T, B)

    # 5) Mean over trials, per timepoint
    mean_full = norm_full.mean(axis=1)               # (T,)
    mean_pca  = norm_pca.mean(axis=1)                # (T,)

    # 6) Plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # light traces for each trial
    ax.plot(time_axis, norm_full, lw=3, color=color_full, alpha=0.2)
    ax.plot(time_axis, norm_pca,  lw=3, color=color_pca,  alpha=0.2)

    # bold mean traces
    ax.plot(time_axis, mean_full, '-', lw=2.0, color=color_full, label='Full space')
    ax.plot(time_axis, mean_pca,  '-', lw=2.0, color=color_pca,  label='PCA space')

    # styling
    ax.set_xlabel("Time (ms)",               fontsize=14)
    ax.set_ylabel("Normalized displacement", fontsize=14)

    ax.tick_params(axis='both', which='major',
                   labelsize=12, length=6, width=1.5, direction='out')
    ax.tick_params(axis='both', which='minor',
                   labelsize=10, length=3, width=1.0, direction='out')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()
    return norm_full






def compute_recovery_times(
    norms_full_arr,         # array of shape (M, T, B)
    time_axis,              # array length T
    perturb_start,          # index where perturbation starts
    perturb_end,            # index where perturbation ends
    perturb_magnitudes,
    threshold_frac=0.2
):
    """
    For each perturbation magnitude:
      • computes the mean response over trials,
      • defines threshold at baseline + threshold_frac*(peak−baseline),
      • finds first time *after* perturb_end at which the signal 
        falls below that threshold,
      • returns the *elapsed* time (i.e. recovery_duration = t_rec − t_end).
    """
    M, T, B = norms_full_arr.shape
    recovery_durations = []

    # cache the absolute end time
    t_end = time_axis[perturb_end]

    for m_idx in range(M):
        mean_curve = norms_full_arr[m_idx].mean(axis=1)  # shape (T,)

        # baseline & peak inside the perturbation window
        baseline = mean_curve[:perturb_start].mean()
        peak     = mean_curve[perturb_start:perturb_end].max()
        thresh   = baseline + (peak - baseline) * threshold_frac

        # look only *after* the perturbation ends
        post  = mean_curve[perturb_end:]
        below = np.where(post <= thresh)[0]

        if below.size:
            # absolute recovery time
            t_rec_abs = time_axis[perturb_end + below[0]]
            elapsed   = t_rec_abs - t_end
        else:
            elapsed = np.nan

        recovery_durations.append(elapsed)

    recovery_durations = np.array(recovery_durations)

    # Plot
    ticks = [np.min(recovery_durations), np.max(recovery_durations)]
    ticks = np.array(ticks)
    fig, ax = plt.subplots(figsize=(4,4), dpi=300)
    ax.plot(perturb_magnitudes, recovery_durations, 'o-', linewidth=1, color='black')
    ax.set_xlabel('Perturbation magnitude', fontsize=16)
    ax.set_ylabel(f'Recovery time to {int(threshold_frac*100)}% of peak (ms)', fontsize=14)

    ax.tick_params(axis="both", which="major", labelsize=14, length=8, width=2, direction="out")
    ax.tick_params(axis="both", which="minor", length=4, width=1.5, direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks(ticks)
    ax.set_xticks(np.array(perturb_magnitudes))
    fig.tight_layout()
    plt.show()

    return recovery_durations
