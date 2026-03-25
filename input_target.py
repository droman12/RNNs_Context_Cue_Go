import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker


# custom two-color gradient for plotting
start_color = (1.0, 0.549, 0.0)
end_color   = (0.392, 0.584, 0.929)
cmap        = LinearSegmentedColormap.from_list("custom_gradient", [start_color, end_color])


def get_inputs(param, single_pulse_time=None, seed=None, dt=1, num_inputs=2):
    """
    Build inputs/targets binned at dt ms per step.

    Args:
        param:             dict with keys: intervals, ntrials, ntimes, burn_length,
                           pulse, pulseheight, offsets, setonset, setoffset.
        single_pulse_time: if set, all trials use this fixed onset (ms).
        seed:              RNG seed.
        dt:                bin size in ms.
        num_inputs:        2 or 3.
            2-channel layout:
                ch0 = magnitude cue (constant 'mag' from burn_len onward)
                ch1 = pulse (pulseheight for pulse bins at flash onset)
            3-channel layout:
                ch0 = tonic context cue (0.3 from burn_len onward)
                ch1 = magnitude cue (mag from burn_len to start2-50, then 0)
                ch2 = pulse (pulseheight for pulse bins at flash onset)

    Returns:
        inputs:    (ntimes, ntrials, num_inputs)
        target:    (ntimes, ntrials, 1)        — NaN outside response window
        set_onset: (ntrials,)                  — flash onset in bins
        ip_idx:    (ntrials,)                  — condition index per trial
    """
    if num_inputs not in (2, 3):
        raise ValueError(f"num_inputs must be 2 or 3, got {num_inputs}")

    # RNG
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    # unpack & convert time params (ms → bins)
    intervals_ms = np.array(param["intervals"], dtype=float)
    intervals    = np.round(intervals_ms / dt).astype(int)

    burn_len = int(round(param["burn_length"] / dt))
    pulse    = int(round(param["pulse"]       / dt))
    ntimes   = int(round(param["ntimes"]      / dt))

    # flash-onset sample (bins)
    lo_bin    = param["setonset"]  / dt
    hi_bin    = param["setoffset"] / dt
    set_onset = np.round(rng.uniform(lo_bin, hi_bin, param["ntrials"])).astype(int)
    if single_pulse_time is not None:
        set_onset = np.full(param["ntrials"],
                            int(round(single_pulse_time / dt)),
                            dtype=int)

    # condition index (tile through interval conditions)
    n_conds = len(param["intervals"])
    ip_idx  = np.tile(np.arange(n_conds), int(np.ceil(param["ntrials"] / n_conds)))
    ip_idx  = ip_idx[: param["ntrials"]]

    # allocate
    inputs = np.zeros((ntimes, param["ntrials"], num_inputs), dtype=np.float32)
    target = np.full( (ntimes, param["ntrials"], 1),          np.nan, dtype=np.float32)

    # ramp constants
    eta, A  = 3.3, 2.85
    offsets = np.array(param["offsets"], dtype=float)

    for i in range(param["ntrials"]):
        cond   = ip_idx[i]
        onset  = set_onset[i]
        length = intervals[cond]
        mag    = offsets[cond]

        # exponential ramp (length bins)
        vect = np.arange(1, length + 1)
        cf   = A * (np.exp(vect / (eta * length)) - 1)
        cf  /= cf.max()

        start2 = burn_len + onset       # flash start (bins)
        end2   = start2 + pulse         # flash end   (bins)

        if num_inputs == 2:
            # ch0: constant magnitude cue from burn_len onward
            inputs[burn_len:, i, 0] = mag
            # ch1: pulse at flash
            inputs[start2:end2, i, 1] = param["pulseheight"]

        else:  # num_inputs == 3
            # ch0: tonic context cue
            inputs[burn_len:, i, 0] = 0.3
            # ch1: magnitude cue, turned off 50 bins before flash
            inputs[burn_len:start2 - 50, i, 1] = mag
            inputs[start2 - 50:,         i, 1] = 0.0
            # ch2: pulse at flash
            inputs[start2:end2, i, 2] = param["pulseheight"]

        # target: 0 during burn + pre-flash, ramp after flash
        target[burn_len:end2, i, 0] = 0.0
        resp_end = min(end2 + length, ntimes)
        target[end2:resp_end, i, 0] = cf[: resp_end - end2]

    return inputs, target, set_onset, ip_idx


def plot_inputs_targets(param, seed=None, single_pulse_time=None,
                        custom_cmap=cmap, dt=1, num_inputs=2):
    inputs, target, set_onset, ip_idx = get_inputs(
        param, single_pulse_time=single_pulse_time,
        seed=seed, dt=dt, num_inputs=num_inputs
    )
    intervals       = np.array(param["intervals"], dtype=int)
    trial_durations = intervals[ip_idx]
    norm_durations  = (trial_durations - trial_durations.min()) / (
                       trial_durations.max() - trial_durations.min())

    timeaxis = np.arange(inputs.shape[0]) * dt
    n_panels = num_inputs + 1               # one panel per input channel + target
    labels   = ([f"Input {k}" for k in range(num_inputs)] + ["Target"])

    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 2 * n_panels),
                             sharex=True, dpi=150)
    for i in range(param["ntrials"]):
        color = custom_cmap(norm_durations[i])
        for k in range(num_inputs):
            axes[k].plot(timeaxis, inputs[:, i, k],
                         color=color, linewidth=1.0, alpha=0.9)
        axes[-1].plot(timeaxis, target[:, i, 0],
                      color=color, linewidth=1.0, alpha=0.9)

    for ax, label in zip(axes, labels):
        ax.set_ylabel(label, fontsize=14)
        ax.tick_params(axis="both", which="major",
                       labelsize=14, length=8, width=2, direction="out")
        ax.tick_params(axis="both", which="minor",
                       length=4, width=1.5, direction="out")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time (ms)", fontsize=14)
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
    dt=1,
    num_inputs=2):
    inputs, target, set_onset, ip_idx = get_inputs(
        param, single_pulse_time=single_pulse_time,
        seed=seed, dt=dt, num_inputs=num_inputs
    )
    nt, B, nch = inputs.shape

    # simulate noise (ch0 gets 5x noise in 3-channel mode, matching original)
    rng   = np.random.RandomState(seed)
    noise = rng.randn(nt, B, nch) * noise_std
    if num_inputs == 3:
        noise[..., 0] *= 5.0
    noisy_inputs = inputs + noise

    timeaxis        = np.arange(nt) * dt
    intervals       = np.array(param["intervals"], dtype=int)
    trial_durations = intervals[ip_idx]
    norm_durations  = (trial_durations - trial_durations.min()) / (
                       trial_durations.max() - trial_durations.min())

    # n_panels = input channels + target
    n_panels = num_inputs + 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(6, 2 * n_panels),
                             sharex=True, dpi=300)

    for i in range(B):
        color = custom_cmap(norm_durations[i])
        for k in range(num_inputs):
            # noisy (thick, transparent) then clean (thin, opaque)
            axes[k].plot(timeaxis, noisy_inputs[:, i, k],
                         color=color, linewidth=3.0, alpha=0.1, zorder=1)
            axes[k].plot(timeaxis, inputs[:, i, k],
                         color=color, linewidth=1.0, alpha=0.9, zorder=2)
        axes[-1].plot(timeaxis, target[:, i, 0],
                      color=color, linewidth=1.0, alpha=0.9)

    labels = [f"Input {k}" for k in range(num_inputs)] + ["Target"]
    for ax, label in zip(axes, labels):
        ax.set_ylabel(label, fontsize=16)
        ax.tick_params(axis="both", which="major",
                       labelsize=14, length=8, width=2, direction="out")
        ax.tick_params(axis="both", which="minor",
                       length=4, width=1.5, direction="out")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time (ms)", fontsize=16)
    plt.tight_layout()
    fig.align_ylabels(axes)
    plt.savefig("inputs_targets.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return inputs, noisy_inputs, target, set_onset, ip_idx


if __name__ == "__main__":
    intervals1  = np.linspace(475, 525, 10, endpoint=True)
    intervals2  = np.linspace(760, 840, 10, endpoint=True)
    new_targets = np.concatenate((intervals1, intervals2))
    new_inputs  = np.interp(new_targets, [500, 800], [0.2, 0.5])

    params = {
        "intervals":      new_targets,
        "ntrials":        20,
        "ntimes":         3500,
        "burn_length":    50,
        "input_duration": 3450,
        "pulse":          100,
        "pulseheight":    0.25,
        "offsets":        new_inputs,
        "setonset":       1000,
        "setoffset":      2000,
    }

    # 2-channel
    plot_inputs_targets(param=params, num_inputs=2)
    plot_inputs_targets_with_noise(params, seed=None,
                                   single_pulse_time=1500,
                                   noise_std=0.01, num_inputs=2)

    # 3-channel
    plot_inputs_targets(param=params, num_inputs=3)
    plot_inputs_targets_with_noise(params, seed=None,
                                   single_pulse_time=1500,
                                   noise_std=0.01, num_inputs=3)