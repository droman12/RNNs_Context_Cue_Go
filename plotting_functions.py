import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as ticker
from scipy import stats
import matplotlib as mpl


start_color = (1, 0.549, 0)
end_color = (0.392, 0.584, 0.929)
cmap = LinearSegmentedColormap.from_list('custom_gradient', [start_color, end_color])

mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['xtick.major.size']  = 6
mpl.rcParams['ytick.major.size']  = 6

def plot_output_targets(output, target, set_onset = 1650, cmap=cmap):
    if hasattr(output, 'detach'):
        output = output.detach().cpu().numpy()
    if hasattr(target, 'detach'):
        target = target.detach().cpu().numpy()
    output = output[1000:2700, :, :]
    target = target[1000:2700, :, :]
    set_onset = 650
    T, B, D = output.shape
    assert D == 1, "1-dimensional outputs and targets"
    trial_col = np.linspace(0,1, B)
    fig, ax = plt.subplots(figsize=(6,4), dpi=300)
    for i in range(B):
        col = cmap(trial_col[i])
        ax.plot(np.arange(T)-set_onset, output[:, i, 0], color=col, lw=1, alpha=1)
        ax.plot(np.arange(T)-set_onset, target[:, i, 0], color=col, lw=3, alpha = 0.3)

    ax.hlines(1.0, xmin = -set_onset, xmax = T-set_onset,colors='black', lw =0.5, linestyles='dashed')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    times = [0, 500, 800]
    ax.plot([0, 0], [-0.1, 1.0], lw=0.5, color='black', linestyle='--')
    ax.plot([500, 500], [-0.1, 1.0], lw=0.5, color='black', linestyle='--')
    ax.plot([800, 800], [-0.1, 1.0], lw=0.5, color='black', linestyle='--')
    ticks = np.array(times)
    ax.set_ylim(bottom = -0.1, top = 1.1) 

    ax.set_xticks(ticks)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
     
    ax.xaxis.label.set_color('grey')        #setting up X-axis label color to yellow
    ax.yaxis.label.set_color('grey')      
    ax.tick_params(axis='x', colors='grey')    #setting up X-axis tick color to red
    ax.tick_params(axis='y', colors='grey')  #
    ax.spines['left'].set_color('grey')        # setting up Y-axis tick color to red
    ax.spines['bottom'].set_color('grey')         #setting up above X-axis tick color to red
    ax.set_xlabel("Time(ms)", fontsize=16)

    fig.tight_layout()
    return fig

def plot_hidden_units(hidden_acts, unit_list, set_onset =1650, hidden_acts_perturbed = None, cmap=None):
    """
    hidden_acts: np.ndarray of shape (T, B, H)
    unit_list: list of hidden‐unit indices to plot
    cmap:     name of matplotlib colormap
    """
    T, B, H = hidden_acts.shape
    # get a colormap instance
    trials = np.linspace(0,1,B)
    # if len(Tp) == 20:
    #     Tp_short = Tp[:10]
    #     Tp_long  = Tp[10:]
    # else:
    #     Tp_short = Tp[:7]
    #     Tp_long = Tp[7:] 
    # Tp_avg = [np.median (Tp_short), np.median(Tp_long)]
    for idx in unit_list:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
        # hidden_acts[:, trial, idx] is the time course of unit idx on trial
        for trial in range(B):
            color = cmap(trials[trial])  # maps trial→color
            if hidden_acts_perturbed is None:
                ax.plot(
                    np.arange(T)-set_onset,
                    hidden_acts[:, trial, idx],
                    color=color,
                    lw=0.75,
                    alpha=1.0,
                )

                # ax.axvline(x = Tp[trial], color = color, lw =3, alpha = 0.0)
                
            else:
                ax.plot(
                    np.arange(T)-set_onset,
                    hidden_acts[:, trial, idx],
                    color='gray',
                    lw=2,
                    alpha=0.2,
                    label=f"trial {trial}" if idx == unit_list[0] else None
                )
                ax.plot(
                np.arange(T)-set_onset,
                hidden_acts_perturbed[:, trial, idx],
                color=color,
                lw=1,
                alpha=1,
                label=f"trial {trial}" if idx == unit_list[0] else None)
                
        times = [0, 500, 800]
        ticks = np.array(times)
        ax.axvline(x = 0, lw = 0.5, color = 'black', label = "Set onset", linestyle = '--')
        ax.axvline(x = 500, lw = 0.5, color = 'black', label = "Dark", linestyle = '--')
        ax.axvline(x = 800, lw = 0.5, color = 'black', label = "Light", linestyle = '--')
        # ax.axvspan(0, 10, color='grey', alpha=0.3)
        
        ax.set_xlabel("Time (ms)", fontsize = 16)
        ax.set_ylabel("Activity (a.u.)", fontsize = 16)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks(ticks)
        
        
        ax.xaxis.label.set_color('grey')        #setting up X-axis label color to yellow
        ax.yaxis.label.set_color('grey')      
        ax.tick_params(axis='x', colors='grey')    #setting up X-axis tick color to red
        ax.tick_params(axis='y', colors='grey')  #
        ax.spines['left'].set_color('grey')        # setting up Y-axis tick color to red
        ax.spines['bottom'].set_color('grey')         #setting up above X-axis tick color to red
        fig.tight_layout()
       
        # fig.savefig(f"unit_{idx}_trials.png", dpi=150)
        plt.show()


def plot_pca_hidden(hidden_seq, num_components=3, x_star=None, x_star_2 = None, set_onset=None, colors=None):
    """
    Perform PCA on hidden_seq and plot 3D trajectories.
    hidden_seq: array [time, trials, neurons]
    num_components: number of PCs to plot (<=3)
    x_star: optional fixed points array [n_fp, neurons]
    set_onset: optional index or list of onset times
    colors: optional colors for each trajectory point (shape [time*trials])
    """
    seq_length, trials, neurons = hidden_seq.shape
    trial_pos = np.linspace(0, 1, trials)
    trial_rgba = cmap(trial_pos)            # shape (n_trials, 4)
    colors = np.tile(trial_rgba, (seq_length, 1))

    flat = hidden_seq.reshape(-1, neurons)
    mean_hidden = flat.mean(axis=0)
    centered = flat - mean_hidden
    # SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pcs = Vt[:num_components].T      # shape [neurons, num_components]
    proj = centered @ pcs            # shape [time*trials, num_components]
    # reshape to [time, trials, num_components]
    proj_reshaped = proj.reshape(seq_length, trials, num_components)

    # prepare figure
    fig = plt.figure(figsize=(8, 8), dpi = 300)
    ax = fig.add_subplot(111, projection='3d')
    

    # flatten for scatter
    x_vals = proj[:, 0]
    y_vals = proj[:, 1] if num_components > 1 else np.zeros_like(x_vals)
    z_vals = proj[:, 2] if num_components > 2 else np.zeros_like(x_vals)
    sc = ax.scatter(x_vals, y_vals, z_vals,
                    c=colors if colors is not None else None,
                    alpha=0.6, s=1, linewidths=0.05)
    if set_onset is not None:
        # ensure list
        onsets = set_onset if hasattr(set_onset, '__iter__') else [set_onset]
        for onset in onsets:
            if 0 <= onset < seq_length:
                pts = proj_reshaped[onset, :, :]
                pts_after_set = proj_reshaped[onset+200, : , :]
                # ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                #            marker='o', s=50, label=f'Set onset @ t={onset}')
                ax.scatter(pts_after_set[:, 0], pts_after_set[:,1], pts_after_set[:,2], 
                           marker ='^', s = 50, color = 'gray')

    if x_star is not None:
        # center x_star and project
        centered_fp = x_star - mean_hidden
        fp_proj = centered_fp @ pcs  # shape [n_fp, num_components]
        ax.scatter(fp_proj[:, 0], fp_proj[:, 1], fp_proj[:, 2],
                   marker='X', s=100, color='k', label='Fixed points')
    if x_star_2 is not None:
        # center x_star and project
        centered_fp = x_star_2 - mean_hidden
        fp_proj = centered_fp @ pcs  # shape [n_fp, num_components]
        ax.scatter(fp_proj[:, 0], fp_proj[:, 1], fp_proj[:, 2],
                   marker='X', s=100, color='k', label='Fixed points')
        
    ax.tick_params(
    which='both',      # affect both major and minor ticks
    length=0,          # no tick "lines"
    labelbottom=False, # x‐axis labels off
    labelleft=False,   # y‐axis labels off
    labelright=False,  # z‐axis labels off
    labeltop=False
    )
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo['grid']['color'] = (1, 1, 1, 0)
    ax.view_init(elev=-30, azim=75)
    ax.set_xlabel('PC1', fontsize = 18)
    ax.set_ylabel('PC2', fontsize = 18)
    ax.set_zlabel('PC3', fontsize = 18)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # 2) (optional) also remove any tick‐label text
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    plt.tight_layout()
    
                      # [1, 2, …, k]

    var_explained = S**2
    var_ratio = var_explained / np.sum(var_explained)      # shape = [n_components]
    cum_var_ratio = np.cumsum(var_ratio)
    n_pcs = 3                                   # total # of PCs
    pc_idx = np.arange(1, n_pcs+1)         
    plt.figure(figsize=(3,3), dpi = 300)
    plt.scatter(pc_idx, cum_var_ratio[:n_pcs], s=50, color = 'black')
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    
    plt.ylim(0,1.02)
    plt.title('Cumulative variance explained')
    plt.tight_layout()
    plt.show()
    

    return proj_reshaped, proj, pcs



def plot_pca_hidden_variable_lengths(
    trial_seqs,
    num_components=3,
    x_star=None,
    x_star_2=None,
    set_onset=None,
    cmap=cmap,
    alpha=0.6,
    s=1
):
    """
    Perform PCA on a collection of trial sequences with variable lengths and plot 3D trajectories.

    trial_seqs: list of arrays, each of shape [time_i, neurons]
    num_components: number of principal components to project (<=3)
    x_star: optional fixed points array [n_fp, neurons]
    x_star_2: optional second set of fixed points [n_fp, neurons]
    set_onset: optional onset index or list of onset indices relative to each trial
    cmap: matplotlib colormap for coloring trials
    alpha: transparency for scatter points
    s: marker size for scatter

    Returns:
      proj_by_trial: list of arrays, each of shape [time_i, num_components]
      pcs: array [neurons, num_components]
    """
    # Number of trials
    n_trials = len(trial_seqs)
    # Collect lengths and stack all data
    lengths = [seq.shape[0] for seq in trial_seqs]
    neurons = trial_seqs[0].shape[1]
    # Stack for PCA
    flat = np.vstack(trial_seqs)           # shape [sum(lengths), neurons]
    mean_hidden = flat.mean(axis=0)
    centered = flat - mean_hidden

    # SVD for PCA
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pcs = Vt[:num_components].T          # shape [neurons, num_components]
    proj_flat = centered @ pcs           # shape [sum(lengths), num_components]

    # Split projections back into trials
    splits = np.cumsum(lengths)[:-1]
    proj_by_trial = np.split(proj_flat, splits, axis=0)

    # Colors for each trial
    trial_pos = np.linspace(0, 1, n_trials)
    trial_rgba = cmap(trial_pos)

    # Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1.5, 1, 1])

    for i, traj in enumerate(proj_by_trial):
        xs = traj[:, 0]
        ys = traj[:, 1] if num_components > 1 else np.zeros_like(xs)
        zs = traj[:, 2] if num_components > 2 else np.zeros_like(xs)
        ax.plot(xs, ys, zs, color=trial_rgba[i], alpha=alpha, linewidth=1)
        ax.scatter(xs, ys, zs, color=trial_rgba[i], s=s, alpha=alpha)

    # Plot onsets if provided
    if set_onset is not None:
        onsets = set_onset if hasattr(set_onset, '__iter__') else [set_onset]
        for onset in onsets:
            for i, traj in enumerate(proj_by_trial):
                if 0 <= onset < traj.shape[0]:
                    pt = traj[onset]
                    # marker after some offset (e.g. onset + 500 if exists)
                    ax.scatter(pt[0], pt[1] if num_components>1 else 0, pt[2] if num_components>2 else 0,
                               marker='^', s=50, color='gray')

    # Plot fixed points
    for fp_set, marker, label in [(x_star, 'X', 'Fixed points'), (x_star_2, 'X', 'Fixed points 2')]:
        if fp_set is not None:
            centered_fp = fp_set - mean_hidden
            fp_proj = centered_fp @ pcs
            ax.scatter(fp_proj[:, 0], fp_proj[:, 1] if num_components>1 else 0,
                       fp_proj[:, 2] if num_components>2 else 0,
                       marker=marker, s=100, color='k', label=label)

    ax.tick_params(colors='black', labelsize=12, width=1, length=4)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo['grid']['color'] = (1, 1, 1, 0)
    ax.view_init(elev=25, azim=90)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    
    plt.tight_layout()
    plt.show()

    return proj_by_trial, pcs




def plot_overlayed_pca(baseline_hidden,
                       perturbed_hidden,
                       num_components=3,
                       cmap=cmap):
    """
    baseline_hidden, perturbed_hidden: arrays of shape (T, trials, neurons)
    num_components: how many PCs to keep (<=3)
    cmap: a matplotlib Colormap for the perturbed trajectories
    """
    # check shapes
    assert baseline_hidden.shape == perturbed_hidden.shape, \
        "baseline and perturbed must match shapes"
    T, n_trials, n_neurons = baseline_hidden.shape

    # flatten to (T*n_trials, neurons)
    flat_base = baseline_hidden.reshape(-1, n_neurons)
    flat_pert = perturbed_hidden.reshape(-1, n_neurons)

    # center by baseline mean
    mean_base = flat_base.mean(axis=0)
    centered_base = flat_base - mean_base
    centered_pert = flat_pert - mean_base

    # PCA via SVD on baseline
    _, _, Vt = np.linalg.svd(centered_base, full_matrices=False)
    pcs = Vt[:num_components].T   # shape (neurons, num_components)

    # project both datasets
    proj_base = centered_base @ pcs   # (T*n_trials, num_components)
    proj_pert = centered_pert @ pcs

    # reshape back to (T, trials, comps)
    PB = proj_base.reshape(T, n_trials, num_components)
    PP = proj_pert.reshape(T, n_trials, num_components)

    # prepare colors for perturbed
    trial_pos = np.linspace(0, 1, n_trials)
    pert_colors = cmap(trial_pos)           # (n_trials,4)
    pert_colors = np.tile(pert_colors, (T, 1))  # (T*n_trials,4)

    # flatten for plotting
    xb = PB[:, :, 0].reshape(-1)
    yb = PB[:, :, 1].reshape(-1) if num_components>1 else np.zeros_like(xb)
    zb = PB[:, :, 2].reshape(-1) if num_components>2 else np.zeros_like(xb)

    xp = PP[:, :, 0].reshape(-1)
    yp = PP[:, :, 1].reshape(-1) if num_components>1 else np.zeros_like(xp)
    zp = PP[:, :, 2].reshape(-1) if num_components>2 else np.zeros_like(xp)

    # plot
    fig = plt.figure(figsize=(10,7), dpi = 300)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])

    # baseline in gray
    ax.scatter(xb, yb, zb,
               color='gray',
               alpha=0.1,
               s=1,
               zorder = 1,
               label='Baseline')

    # perturbed in gradient
    ax.scatter(xp, yp, zp,
               c=pert_colors,
               alpha=0.8,
               s=2,
               zorder = 2, 
               label='Perturbed')

    # styling
    ax.tick_params(
    which='both',      # affect both major and minor ticks
    length=0,          # no tick "lines"
    labelbottom=False, # x‐axis labels off
    labelleft=False,   # y‐axis labels off
    labelright=False,  # z‐axis labels off
    labeltop=False
    )
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo['grid']['color'] = (1, 1, 1, 0)
    ax.view_init(elev=160, azim=-90)
    ax.set_xlabel('PC1', fontsize = 18)
    ax.set_ylabel('PC2', fontsize = 18)
    ax.set_zlabel('PC3', fontsize = 18)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # 2) (optional) also remove any tick‐label text
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    plt.tight_layout()
    plt.show()

    return PB, PP


# def plot_individual(trajectory_original, trajectory_perturbed, pc_channel, cmap = cmap):
#     fig, ax = plt.subplot(figsize = (5,4), dpi = 150)
#     for idx in range(trajectory_original.shape[1]):
#         col = trial_col[idx]
#         ax.plot(trajectory_original[100:, idx, pc_channel], color = col)
#         ax.plot(trajectory_perturbed[])
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)



def eigenvalue_spectra(eigs, cmap=cmap):
    num_fps = len(eigs)
    trial_pos = np.linspace(0, 1, num_fps)
 
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    for idx, eigvals in enumerate(eigs):
        # map idx → color
        color = cmap(trial_pos[idx])
        ax.scatter(eigvals.real,
                    eigvals.imag,
                    color=color,
                    alpha=0.7,
                    s=10,
                    label=f'FP {idx+1}' if idx==0 else None)
    ax.set_xlabel("Real", fontsize = 14)
    ax.set_ylabel("Imaginary", fontsize = 14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', which='major', labelsize=14,
                   length=8, width=2, direction='out')
    ax.tick_params(axis='both', which='minor', labelsize=12,
                   length=4, width=1.5, direction='out')
    fig.tight_layout()
    plt.axis('square')
    plt.show()

# function that takes as input target and output and plots Tp (produdced time = time the otuput first reaches a threshold) vs Ts
# function that returns performance metric (normalized from 0 to 1) based on nhow well the produced time approaches the target time. 

def decode_time(
    sequence: np.ndarray,
    set_onset: int,
    threshold: float = 1.0
):
    """
    For each trial in `sequence` (shape [time, n_trials, ...]),
    find the first time index where any unit crosses `threshold`.
    Subtracts `set_onset` and a fixed burn-in to give relative times.
    Returns: list of length n_trials, with None for no crossing.
    """
    produced_times: List[Optional[int]] = []
    burn_length = 50

    # iterate over trials
    for trial in range(sequence.shape[1]):
        pred = sequence[:, trial, :]                  # [time, n_units]
        row_inds = np.nonzero(pred >= threshold)[0]   # indices where any unit ≥ threshold

        if row_inds.size:
            # first crossing, adjust by onset and burn
            t = int(row_inds[0]) - set_onset - burn_length
            produced_times.append(t)
        else:
            # no crossing → record None
            produced_times.append(None)

    return produced_times

def Tp_Ts_plot(output, target, set_onset, threshold=0.95, cmap=cmap):
    """
    Scatter Tp vs Ts (each trial colored along cmap), draw best‐fit line and identity.
    Returns:
      Tp, Ts  arrays of length n_trials
    """
    # decode produced and target times per trial
    Tp = decode_time(output, set_onset, threshold)
    Ts = decode_time(target, set_onset, threshold)
    Tp = np.array([t if t is not None else -1 for t in Tp])
    Ts = np.array([t if t is not None else -1 for t in Ts])
    #perf =  performance_nrmse(output, target, set_onset, threshold)

    # only keep trials where both crossed
    mask = (Tp >= 0) & (Ts >= 0)
    if mask.sum() < 2:
        raise ValueError("Need at least two valid trials for regression.")

    # build per‐trial colors
    n_trials = len(Tp)
    cmap = cmap
    trial_pos = np.linspace(0, 1, n_trials)
    colors = cmap(trial_pos)   # shape (n_trials, 4)

    # plot
    fig, ax = plt.subplots(figsize=(4,4), dpi = 300)

    # scatter each trial
    plt.scatter(Ts, Tp, c=colors, alpha=1, s=80, edgecolor='k', linewidth=0.5, label='trials', zorder = 2)

    # if mask.sum() >= 2:
    #     slope, intercept, _, _, _ = stats.linregress(Ts[mask], Tp[mask])
    #     x_fit = np.array([Ts[mask].min(), Ts[mask].max()])
    #     y_fit = slope * x_fit + intercept
    #     plt.plot(x_fit, y_fit, linestyle='--', lw=1, color='black', zorder = 1)
    # else:
    #     print("Warning: Not enough valid trials for regression. Showing scatter only.")
   
    all_min = min(Ts.min(), Ts.min())
    all_max = max(Ts.max(), Ts.max())

    #ax.set_xlim(all_min, all_max)
    #ax.set_ylim(all_min, all_max)
    #identity line
    identity_x = np.array([all_min, all_max])
    identity_y = identity_x.copy()
    ax.plot(identity_x, identity_y,
            lw=0.5,
            color='black',
            zorder=1,
            label='Identity: $T_p=T_s$')

    ax.set_xlabel('Target time $T_s$ (ms)', fontsize = 16)
    ax.set_ylabel('Produced time $T_p$ (ms)', fontsize = 16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    #plt.title(f'Performance = {perf:.2f}', fontsize = 18)
    # define your “fixed” ticks:
    times = [450, 550, 720, 880]
    ticks = np.array(times)

    # apply to both axes:
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    plt.tight_layout()
    

    plt.show()
    return Tp, Ts 


def performance(output, target, set_onset, threshold=1):
    """
    Returns a scalar in [0,1] measuring how close Tp is to Ts,
    defined as 1 minus the mean absolute percentage error:
      perf = max(0, 1 - mean_i |Tp_i - Ts_i| / Ts_i ).
    A perfect match Tp=Ts gives perf=1 exactly.
    """
    Tp = decode_time(output, set_onset, threshold)
    Ts = decode_time(target, set_onset, threshold)
    mask = (Tp >= 0) & (Ts > 0)   # only valid, positive Ts
    if mask.sum() < 1:
        return 0.0
    # absolute percentage errors
    ape = np.abs(Tp[mask] - Ts[mask]) / Ts[mask]
    mape = np.mean(ape)
    perf = 1.0 - mape
    # clip into [0,1]
    return float(np.clip(perf, 0.0, 1.0))

def performance_nrmse(output, target, set_onset, threshold=1, eps=1e-8):
    """
    Returns a scalar in [0,1] using normalized RMSE:
      perf = 1 - (RMSE / (max(Ts)-min(Ts)))  (clipped at 0)

    A perfect match Tp=Ts gives perf=1.  If RMSE >= range(Ts), perf=0.
    """
    Tp = decode_time(output, set_onset, threshold)
    Ts = decode_time(target, set_onset, threshold)

    mask = (Ts > 0) & (Tp >= 0)   # only valid, positive Ts
    if mask.sum() < 1:
        return 0.0

    Tp_valid = Tp[mask]
    Ts_valid = Ts[mask]

    errors = Tp_valid - Ts_valid
    rmse = np.sqrt(np.mean(errors**2))

    # Normalize by (max-min) of Ts_valid
    ts_max = np.max(Ts_valid)
    ts_min = np.min(Ts_valid)
    rng = ts_max - ts_min

    if rng < eps:
        # All Ts are (nearly) identical; in that case, perf = 1 if rmse is 0,
        # otherwise degrade.  For instance, you could do:
        return float(np.clip(1.0 - (rmse / (np.abs(ts_max) + eps)), 0.0, 1.0))

    nrmse = rmse / (rng + eps)
    perf = 1.0 - np.minimum(nrmse, 1.0)
    return float(perf)


from typing import List, Optional

def compute_performance(
    produced: List[Optional[float]], 
    targets:  List[float]
) -> float:

    if len(produced) != len(targets):
        raise ValueError("produced and targets must be the same length")

    scores = []
    for p, t in zip(produced, targets):
        if p is None:
            scores.append(0.0)
        else:
            # protect against zero or negative targets if needed
            if t <= 0:
                raise ValueError(f"Target must be >0, got {t}")
            scores.append(min(p, t) / max(p, t))

    return sum(scores) / len(targets)


def speed(pca_trajectories, produced_time, dt = 1, cmap = cmap):
    central_diff = np.zeros_like(pca_trajectories)
    central_diff[0, :, :] = pca_trajectories[1, :, :] - pca_trajectories[0, :, :]
    central_diff[-1, :, :] = pca_trajectories[-1, :, :] - pca_trajectories[-2, :, :]
    central_diff[1:-1, :, :] = (pca_trajectories[2:, :, :] - pca_trajectories[:-2, :, :]) / (2.0 * dt)
    speed_pca_central = np.linalg.norm(central_diff, axis=2)
    avg_speed_per_trial = np.mean(speed_pca_central, axis  = 0)
    norm_avg_speed = (avg_speed_per_trial - avg_speed_per_trial.min()) / (avg_speed_per_trial.max() - avg_speed_per_trial.min())
    fig, ax = plt.subplots(figsize = (4,4), dpi = 300)
    for i in range(pca_trajectories.shape[1]):
        color = cmap(i/pca_trajectories.shape[1]) 
        ax.scatter(produced_time[i], norm_avg_speed[i], s=80, alpha = 1, edgecolor='k', color = color, zorder=2)
    coefficients = np.polyfit(produced_time, norm_avg_speed, 1)
    slope, intercept = coefficients

    x_fit = np.linspace(produced_time.min(), produced_time.max(), 100)
    y_fit = slope * x_fit + intercept

    ax.plot(x_fit, y_fit, color='black', linewidth=0.5, zorder = 1)

    ax.set_xlabel("Produced Interval", fontsize=16)
    ax.set_ylabel("Normalized Average Speed", fontsize=16)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # define your “fixed” ticks:
    times = [450, 550, 720, 880]
    ticks = np.array(times)

    # apply to both axes:
    ax.set_xticks(ticks)


    # style ticks:
    ax.tick_params(which='major', length=8, width=2)
    ax.tick_params(which='minor', length=4, width=1)

    fig.tight_layout()
    plt.show()



def plot_compressed_alpha_heatmap(leading, cmap='PRGn', alpha_label=r'$\lambda$'):
    """
    Plots a heatmap of the leading eigenvalue α across trials,
    compressing time by removing redundant columns (timepoints)
    where no changes occur in any trial.
    
    Parameters:
    - leading: np.ndarray of shape (n_trials, n_times)
    - cmap: colormap for the heatmap
    - alpha_label: label for the colorbar
    """
    n_trials, n_times = leading.shape

    # Identify timepoints where there is any change across trials
    # We'll include the first column and any column where change occurs in any trial
    tol = 0.005
    # 1) Compute significant changes between each pair of adjacent timepoints
    diffs   = np.abs(np.diff(leading, axis=1))     # shape = (n_trials, n_times-1)
    changes = np.any(diffs > tol, axis=0)          # shape = (n_times-1,)

    # 2) Make the boolean mask (always keep first and last)
    keep_cols       = np.insert(changes, 0, True)  # now length = n_times
    keep_cols[-1]   = True                         # ensure last timepoint is kept

    # 3) Compress both arrays
    leading_compressed    = leading[:, :]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8), dpi = 300)
    im = ax.imshow(leading_compressed, aspect='auto', origin = 'upper', cmap=cmap)
    ax.set_yticks(np.arange(-.5, n_trials, 1), minor=True)
    ax.grid(which='minor', axis='y', color='white', linestyle='-', linewidth=0.8)
    ax.tick_params(which='minor', length=0)
    ax.set_xlabel('time step')
    ax.set_ylabel('trial index')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(alpha_label, rotation=270, labelpad=15)
    
    ax.set_title("Leading eigenvalue α (compressed in time)")
    plt.tight_layout()
    plt.show()
    return keep_cols


def delta_lambda_heatmap(x, x_pert, times=None, t_mark=None, flip_threshold=1.0,
                         vclip=99.0, figsize=(8, 4.5)):
    """
    x, x_pert: arrays of shape (n_trials, n_times), may contain NaNs.
    times:     1D array of length n_times (for x-axis); else uses indices.
    t_mark:    scalar time index/value to draw a vertical line (e.g., perturb onset).
    flip_threshold: λ=1 boundary to detect stability flips.
    vclip:     percentile for symmetric color-limit clipping (robust scaling).
    """

    assert x.shape == x_pert.shape, "x and x_pert must have the same shape"
    Δ = x_pert - x

    # symmetric, robust color limits around 0
    finite = np.isfinite(Δ)
    if not np.any(finite):
        raise ValueError("Δλ contains no finite values.")
    vmax = np.percentile(np.abs(Δ[finite]), vclip)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    n_trials, n_times = Δ.shape
    if times is None:
        times = np.arange(n_times)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.2], hspace=0.35)

    # Panel A — Δλ heatmap
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(Δ, aspect='auto', interpolation='nearest',
                    origin='upper', norm=norm)
    ax0.set_ylabel("trial #")
    ax0.set_xticks(np.linspace(0, n_times-1, min(10, n_times)).astype(int))
    ax0.set_xticklabels([f"{times[i]:g}" for i in ax0.get_xticks().astype(int)])
    ax0.set_title("Δλ (perturbed − original)")
    cbar = fig.colorbar(im, ax=ax0)
    cbar.set_label("Δλ (pert − orig)")

    # optional vertical marker
    if t_mark is not None:
        # interpret t_mark as index if array-like times not monotonic floats
        if np.isscalar(t_mark):
            # try to map to nearest index in times
            idx = int(np.argmin(np.abs(times - t_mark))) if times is not None else int(t_mark)
        else:
            idx = int(t_mark)
        ax0.axvline(idx, linestyle='--', linewidth=1)

    # Compute mean ± SEM vs time (ignoring NaNs)
    mean_Δ = np.nanmean(Δ, axis=0)
    sem_Δ  = np.nanstd(Δ, axis=0) / np.sqrt(np.sum(np.isfinite(Δ), axis=0))

    # Panel B — mean Δλ ± SEM
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(times, mean_Δ, linewidth=2)
    ax1.fill_between(times, mean_Δ - sem_Δ, mean_Δ + sem_Δ, alpha=0.25, linewidth=0)
    ax1.axhline(0, linewidth=1)
    if t_mark is not None:
        ax1.axvline(times[idx], linestyle='--', linewidth=1)
    ax1.set_xlabel("time")
    ax1.set_ylabel("mean Δλ")

    # (Optional) report per-time “stability flips”
    # flip if λ crosses 1 after perturbation (orig <1 & pert >1) or opposite
    with np.errstate(invalid='ignore'):
        orig_stable = x < flip_threshold
        pert_stable = x_pert < flip_threshold
        flips = (orig_stable != pert_stable) & np.isfinite(x) & np.isfinite(x_pert)
    flip_rate = flips.sum(axis=0) / np.maximum(1, np.isfinite(Δ).sum(axis=0))
    ax1_2 = ax1.twinx()
    ax1_2.plot(times, 100*flip_rate, linestyle=':', linewidth=1.5)
    ax1_2.set_ylabel("% flips", rotation=270, labelpad=15)

    fig.tight_layout()
    return fig, (Δ, mean_Δ, sem_Δ, flip_rate)
