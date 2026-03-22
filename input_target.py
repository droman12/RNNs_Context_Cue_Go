import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker


# custom two‐color gradient for plotting
start_color = (1.0, 0.549, 0.0)
end_color   = (0.392, 0.584, 0.929)
cmap        = LinearSegmentedColormap.from_list("custom_gradient", [start_color, end_color])

def get_inputs(param, single_pulse_time=None, seed=None, dt=1, num_inputs = 2):
    """
    Build inputs/targets binned at dt ms per step.
    Returns:
      inputs:  (n_bins, ntrials, 2)
      target:  (n_bins, ntrials, 1)
      set_onset: (ntrials,)  # in bins
      ip_idx:    (ntrials,)  # condition index per trial
    """
    # RNG
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    # unpack & convert time params (ms → bins)
    intervals_ms = np.array(param["intervals"], dtype=float)
    intervals    = np.round(intervals_ms / dt).astype(int)

    burn_len     = int(round(param["burn_length"]   / dt))
    pulse        = int(round(param["pulse"]         / dt))  # e.g. 100 ms → 10 bins
    ntimes       = int(round(param["ntimes"]        / dt))

    # flash‐onset sample (in bins)
    lo_bin = param["setonset"]  / dt
    hi_bin = param["setoffset"] / dt
    set_onset = rng.uniform(lo_bin, hi_bin, param["ntrials"])
    set_onset = np.round(set_onset).astype(int)
    if single_pulse_time is not None:
        set_onset = np.full(param["ntrials"],
                            int(round(single_pulse_time / dt)),
                            dtype=int)

    # build ip_idx to tile through your interval conditions
    n_conds = len(param["intervals"])
    ip_idx  = np.tile(np.arange(n_conds), int(np.ceil(param["ntrials"]/n_conds)))
    ip_idx  = ip_idx[: param["ntrials"]]

    # allocate
    # inputs = np.zeros((ntimes, param["ntrials"], 2), dtype=np.float32)
    inputs = np.zeros((ntimes, param["ntrials"], 2), dtype=np.float32)
    target = np.full((ntimes, param["ntrials"], 1), np.nan, dtype=np.float32)

    # constants for ramp
    eta, A = 3.3, 2.85
    offsets = np.array(param["offsets"], dtype=float)

    for i in range(param["ntrials"]):
        cond   = ip_idx[i]
        onset  = set_onset[i]
        length = intervals[cond]
        mag    = offsets[cond]

        # build ramp (length bins)
        vect = np.arange(1, length + 1)
        cf   = A * (np.exp(vect / (eta * length)) - 1)
        cf  /= cf.max()

        

        # channel 1 = flash for 'pulse' bins, and blank context
        start2 = burn_len + onset  # 1550
        end2   = start2 + pulse # 1650


        inputs[burn_len:, i, 0] = mag #50 - 1500
        # inputs[start2:end2, i, 0] = 0.0
        # inputs[end2:, i, 0] = mag
        
        inputs[start2:end2, i, 1] = param["pulseheight"]
        

        target[burn_len:end2, i, 0] = 0.0

        # then ramp for 'length' bins
        resp_end = min(end2 + length, ntimes)
        target[end2:resp_end, i, 0] = cf[: resp_end - end2]

    return inputs, target, set_onset, ip_idx


def plot_inputs_targets(param, seed=None, single_pulse_time=None, custom_cmap=cmap, dt=1):
    inputs, target, set_onset, ip_idx = get_inputs(
        param, single_pulse_time=single_pulse_time, seed=seed, dt=dt
    )
    intervals       = np.array(param["intervals"], dtype=int)
    trial_durations = intervals[ip_idx]
    norm_durations  = (trial_durations - trial_durations.min()) / (
                        trial_durations.max() - trial_durations.min()
                      )

    # timeaxis in ms
    n_bins   = inputs.shape[0]
    timeaxis = np.arange(n_bins) * dt

    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True, dpi=150)
    for i in range(param["ntrials"]):
        color = custom_cmap(norm_durations[i])
        axes[0].plot(timeaxis, inputs[:, i, 0], color=color, linewidth=1.0, alpha=0.9)
        axes[1].plot(timeaxis, inputs[:, i, 1], color=color, linewidth=1.0, alpha=0.9)
        axes[2].plot(timeaxis, target[:, i, :], color=color, linewidth=1.0, alpha=0.9)

    axes[0].set_ylabel("Input 1 (ctx)", fontsize=14)
    axes[1].set_ylabel("Input 2 (flash)", fontsize=14)
    axes[2].set_ylabel("Target",       fontsize=14)
    axes[2].set_xlabel("Time (ms)",    fontsize=14)

    for ax in axes:
        ax.tick_params(axis="both", which="major",
                       labelsize=14, length=8, width=2, direction="out")
        ax.tick_params(axis="both", which="minor",
                       length=4, width=1.5, direction="out")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()
    plt.close(fig)

    return inputs, target, set_onset, ip_idx


def plot_inputs_targets_with_noise(
    param,
    seed=None,
    single_pulse_time=None,
    custom_cmap=cmap,
    noise_std=0.01,
    dt=1
):
    # get clean
    inputs, target, set_onset, ip_idx = get_inputs(
        param, single_pulse_time=single_pulse_time, seed=seed, dt=dt
    )
    nt, B, nch = inputs.shape

    # simulate noise
    rng   = np.random.RandomState(seed)
    noise = rng.randn(nt, B, nch) * noise_std
    if nch == 3:
        noise[..., 0] *= 5.0
    noisy_inputs = inputs + noise

    # timeaxis in ms
    timeaxis = np.arange(nt) * dt

    # plotting
    fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True, dpi=400)
    intervals       = np.array(param["intervals"], dtype=int)
    trial_durations = intervals[ip_idx]
    norm_durations  = (trial_durations - trial_durations.min()) / (
                        trial_durations.max() - trial_durations.min()
                      )

    for i in range(B):
        color = custom_cmap(norm_durations[i])

        # channel 0 noisy & clean
        axes[0].plot(timeaxis, inputs[:, i, 0] + noise[:, i, 0],
                     color=color, linewidth=3.0, alpha=0.1, zorder=1)
        axes[0].plot(timeaxis, inputs[:, i, 0],
                     color=color, linewidth=1.0, alpha=0.9, zorder=2)

        # channel 1 noisy & clean
        axes[1].plot(timeaxis, noisy_inputs[:, i, 1],
                     color=color, linewidth=3.0, alpha=0.1, zorder=1)
        axes[1].plot(timeaxis, inputs[:, i, 1],
                     color=color, linewidth=1.0, alpha=0.9, zorder=2)
        
        axes[2].plot(timeaxis, target[:, i, :], color = color, linewidth = 1.0)

    axes[0].set_ylabel("Input 1 (ctx)", fontsize=16)
    axes[1].set_ylabel("Input 2 (flash)", fontsize=16)
    axes[2].set_ylabel("Target", fontsize = 16)
    axes[2].set_xlabel("Time (ms)",      fontsize=16)

    for ax in axes:
        ax.tick_params(axis="both", which="major",
                       labelsize=14, length=8, width=2, direction="out")
        ax.tick_params(axis="both", which="minor",
                       length=4, width=1.5, direction="out")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.align_ylabels(axes)
    plt.tight_layout()
    fig.align_ylabels(axes)

    plt.savefig("inputs_targets.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    plt.show()

    return inputs, noisy_inputs, target, set_onset, ip_idx

if __name__ == "__main__":
    intervals1 = np.linspace(475, 525, 10, endpoint=True)
    intervals2 = np.linspace(760, 840, 10, endpoint=True)
    new_targets = np.concatenate((intervals1, intervals2))

    new_inputs = np.interp(
        new_targets,
        [500, 800],  # old target‐range
        [0.2, 0.5],   # old mag‐range
    )

    params = {
        "intervals": new_targets,
        "ntrials": 20,
        "ntimes": 3500,          # total timesteps per trial
        "burn_length": 50,
        "input_duration": 3450,
        "pulse": 100,
        "pulseheight": 0.25,
        "offsets": new_inputs,
        "setonset": 1000,
        "setoffset": 2000,
    }
    inputs, target, set_onset, ip_idx = plot_inputs_targets(param=params)
    plot_inputs_targets_with_noise(
    params,
    seed=None,
    single_pulse_time=1500,
    custom_cmap=cmap,
    noise_std=0.01,
    dt=1)
    print(set_onset)