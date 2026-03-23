import numpy as np



def get_default_hp(n_inputs, n_outputs, activation = 'tanh', test = False):
    hp = {
        # model architecture
        'n_input'         : n_inputs,         # input dimension
        'n_rnn'           : 200,              # size of recurrent hidden layer
        'n_output'        : n_outputs,        # output dimension

        'dropout_hidden'   : 0.1 ,  # fraction of hidden units to lesion in the recurrence
        'dropout_readout'  : 0.1 , # fraction of hidden units to lesion before readout

        'dropcon_hh'       : 0.0,

        'gamma_manifold' : 0.0,
        
        # RNN dynamics
        'activation'      : activation,           # 'relu' | 'tanh' | 'softplus'
        'decay'           : 0.9,              # α in h_next = α h + (1-α) a_next
        'decay_zeta'      : 0.5,              # ζ decay for recurrent noise
        'bias'            : True,             # include bias term in the cell
        
        # noise levels
        'sigma_rec'       : 0.025,              # σ_rec, recurrent (zeta) noise std
        'sigma_x'         : 0.01,              # σ_x, input noise std
        
        # weight initialization
        'w_rec_init'      : 'randgauss',      # 'diag' | 'randortho' | 'randgauss'
        'spectral_radius' : 1.0,              # g
        
        # loss / training
        'loss_type'       : 'lsq',            # 'lsq' (MSE) or anything else→CrossEntropy
        'l1_h'            : 0.0,              # L1 on hidden activity
        'l2_h'            : 1e-4,              # L2 on hidden activity
        'l1_weight'       : 0.0,              # L1 on all weights
        'l2_weight'       : 0.0,             # L2 on all weights
        
        # data / reproducibility
        'seed'            : 42,               # random seed for get_inputs
        'single_pulse_time': None,             # passed to get_inputs()

        # training / optimization
        'batch_size'      : 80,
        'batch_size_val'  : 20,
        'learning_rate'   : 1e-4,
        'l2_weight'       : 0.0,            # weight‐decay in Adam
        'lr_factor'       : 0.95,           # ReduceLROnPlateau factor
        'lr_patience'     : 20,             # patience for LR scheduler
        'n_epochs'        : 10000,
        'grad_clip'       : 1.0,            # max‐norm for gradient clipping
        'target_perf'     : 0.5,           # early‐stop threshold on val loss

        # checkpointing / logging
        'save_name'       : 'model_tanh_less_noise',        # base filename for best‐model .pt
        'log_dir'         : 'runs_robust',         # parent dir for TensorBoard
    }



    if test:
        hp.update({
            'seed'             : None,
            'single_pulse_time': 1500,
            'batch_size': 16, 
            'sigma_rec' : 0.0, 
            'sigma_x': 0.0

        })

    return hp
 

# def get_default_params(batch_size, seed = 123, test = False):
#     rng = np.random.RandomState(seed=seed)

#     intervals1 = rng.uniform(450, 550, 1)
#     intervals2 = rng.uniform(720, 880, 1)

#     new_targets = np.concatenate((intervals1, intervals2))

#     # same encoding rule as training
#     new_inputs = np.interp(
#         new_targets,
#         [450, 880],
#         [0.2, 0.5],
#     )


#     params = {
#         "intervals": new_targets,
#         "ntrials": batch_size,
#         "ntimes": 3500,          # total timesteps per trial
#         "burn_length": 50,
#         "input_duration": 3450,
#         "pulse": 100,
#         "pulseheight": 0.25,
#         "offsets": new_inputs,
#         "setonset": 1000,
#         "setoffset": 2000,
#     }
#     return params


def get_default_params(batch_size, test = False):
    n1, n2 = (10, 10) if test else (7, 9)
    
    # linearly spaced target intervals
    intervals1 = np.linspace(450, 550, n1, endpoint=True)
    intervals2 = np.linspace(720, 880, n2, endpoint=True)
    new_targets = np.concatenate((intervals1, intervals2))
    new_inputs = np.interp(
        new_targets,
        [450, 880],  # old target‐range
        [0.2, 0.5],   # old mag‐range
    )



    params = {
        "intervals": new_targets,
        "ntrials": batch_size,
        "ntimes": 3500,          # total timesteps per trial
        "burn_length": 50,
        "input_duration": 3450,
        "pulse": 100,
        "pulseheight": 0.25,
        "offsets": new_inputs,
        "setonset": 1000,
        "setoffset": 2000,
    }
    return params


    

# def get_extrapolation_params(batch_size, n_points=50):
#     """
#     Extrapolation in BOTH:
#       - target intervals (outside [450, 880])
#       - input magnitudes (outside [0.2, 0.5])

#     Low side:  300–430 ms  with inputs below 0.2
#     High side: 900–1100 ms with inputs above 0.5
#     """

#     # Split points between low and high extrapolation regions
#     n_low = n_points // 2
#     n_high = n_points - n_low

#     # Intervals outside the training range
#     low_targets = np.linspace(300, 430, n_low, endpoint=True)
#     high_targets = np.linspace(900, 1100, n_high, endpoint=True)
#     new_targets = np.concatenate((low_targets, high_targets))

#     # Piecewise-linear mapping that:
#     #  - preserves the original mapping on [450, 880] -> [0.2, 0.5]
#     #  - extends it below and above that range
#     #    300 -> 0.10
#     #    450 -> 0.20  (same as training)
#     #    880 -> 0.50  (same as training)
#     #    1100 -> 0.60
#     break_intervals = np.array([350, 450, 880, 1050])
#     break_inputs    = np.array([0.15, 0.2, 0.5, 0.55])

#     new_inputs = np.interp(new_targets, break_intervals, break_inputs)

#     params = {
#         "intervals": new_targets,
#         "ntrials": batch_size,
#         "ntimes": 3500,
#         "burn_length": 50,
#         "input_duration": 3450,
#         "pulse": 100,
#         "pulseheight": 0.25,
#         "offsets": new_inputs,
#         "setonset": 1000,
#         "setoffset": 2000,
#     }
#     return params

# def get_default_params(batch_size, test=False):
#     n_points = 100 if test else 50  # adjust for test mode if needed

#     # single interpolated target interval
#     new_targets = np.linspace(450, 880, n_points, endpoint=True)

#     # interpolate input magnitude between 0.2 and 0.5 over this time range
#     new_inputs = np.interp(
#         new_targets,
#         [450, 880],  # time range
#         [0.2, 0.5],  # input magnitude range
#     )

#     params = {
#         "intervals": new_targets,
#         "ntrials": batch_size,
#         "ntimes": 3500,          # total timesteps per trial
#         "burn_length": 50,
#         "input_duration": 3450,
#         "pulse": 100,
#         "pulseheight": 0.25,
#         "offsets": new_inputs,
#         "setonset": 1000,
#         "setoffset": 2000,
#     }
#     return params

# def get_default_params(batch_size, test=False):
#     n_points = 100 if test else 50  # adjust for test mode if needed

#     # single interpolated target interval
#     new_targets = np.linspace(450, 880, n_points, endpoint=True)

#     # interpolate input magnitude between 0.2 and 0.5 over this time range
#     new_inputs = np.interp(
#         new_targets,
#         [450, 880],  # time range
#         [0.2, 0.5],  # input magnitude range
#     )

#     params = {
#         "intervals": new_targets,
#         "ntrials": batch_size,
#         "ntimes": 3500,          # total timesteps per trial
#         "burn_length": 50,
#         "input_duration": 3450,
#         "pulse": 100,
#         "pulseheight": 0.25,
#         "offsets": new_inputs,
#         "setonset": 1000,
#         "setoffset": 2000,
#     }
#     return params
# def get_default_params(batch_size, test = False):
#     n1, n2 = (7, 9) if test else (10, 10)
#     intervals1 = np.linspace(475, 525, n1, endpoint=True)
#     intervals2 = np.linspace(760, 840, n2, endpoint=True)
#     new_targets = np.concatenate((intervals1, intervals2))

#     new_inputs = np.interp(
#         new_targets,
#         [500, 800],  # old target‐range
#         [0.2, 0.5],   # old mag‐range
#     )

#     params = {
#         "intervals": new_targets,
#         "ntrials": batch_size,
#         "ntimes": 2200,          # total timesteps per trial
#         "burn_length": 50,
#         "input_duration": 2200 - 50,
#         "pulse": 10,
#         "pulseheight": 0.5,
#         "offsets": new_inputs,
#         "setonset": 500,
#         "setoffset": 1000,
#     }
#     return params