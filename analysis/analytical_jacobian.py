import numpy as np
import torch.nn as nn


def _activation_derivative(nonlinearity: nn.Module, r: np.ndarray) -> np.ndarray:
    """
    Elementwise derivative of the activation function at pre-activation r.
      tanh:      1 - tanh(r)^2
      relu:      1 where r > 0, else 0
      softplus:  sigmoid(beta * r)  [with beta from the module, default 1]
    """
    if isinstance(nonlinearity, nn.Tanh):
        return 1.0 - np.tanh(r) ** 2
    elif isinstance(nonlinearity, nn.ReLU):
        return (r > 0).astype(np.float64)
    elif isinstance(nonlinearity, nn.Softplus):
        beta = float(nonlinearity.beta)
        return 1.0 / (1.0 + np.exp(-beta * r))
    else:
        raise NotImplementedError(
            f"Analytic derivative not implemented for {type(nonlinearity).__name__}. "
            "Add it to _activation_derivative or use autograd-based Jacobians."
        )


def trajectory_jacobians_analytic(model, hid_seq, x_input, trial_idx=0):
    """
    Analytically compute A = dh_{t+1}/dh_t and B = dh_{t+1}/dx_t
    along a stored trajectory.

    The RNN update rule is:
        r_t     = h_t @ Whh.T + x_t @ Wih.T + bias
        h_{t+1} = decay * h_t + (1-decay) * f(r_t)

    So:
        A_t = dh_{t+1}/dh_t = decay * I + (1-decay) * diag(f'(r_t)) @ Whh
        B_t = dh_{t+1}/dx_t =             (1-decay) * diag(f'(r_t)) @ Wih

    Parameters
    ----------
    model     : Run_Model instance.
    hid_seq   : (T, B, hid_dim)  stored hidden states (post-update).
    x_input   : (T, B, n_inputs) input sequence fed to the model.
    trial_idx : int or iterable of ints selecting trials from the batch dim.

    Returns
    -------
    Single trial (trial_idx is int):
        A_traj : (T, hid_dim, hid_dim)
        B_traj : (T, hid_dim, n_inputs)
    Multiple trials (trial_idx is iterable of K ints):
        A_traj : (T, K, hid_dim, hid_dim)
        B_traj : (T, K, hid_dim, n_inputs)
    """
    cell   = model.model.rnn.rnncell
    Whh    = cell.weight_hh.detach().cpu().numpy()      # (hid_dim, hid_dim)
    Wih    = cell.weight_ih.detach().cpu().numpy()      # (hid_dim, n_inputs)
    bias   = cell.bias.detach().cpu().numpy() if cell.bias is not None else 0.0
    decay  = float(cell.decay)
    alpha  = 1.0 - decay
    nonlin = cell.nonlinearity

    H_all = hid_seq if isinstance(hid_seq, np.ndarray) else hid_seq.detach().cpu().numpy()
    X_all = x_input if isinstance(x_input, np.ndarray) else x_input.detach().cpu().numpy()

    T, B, hid_dim = H_all.shape
    n_inputs      = X_all.shape[2]
    I             = np.eye(hid_dim)

    if isinstance(trial_idx, (int, np.integer)):
        trial_indices, return_single = [int(trial_idx)], True
    else:
        trial_indices, return_single = [int(i) for i in trial_idx], False

    K      = len(trial_indices)
    A_traj = np.zeros((T, K, hid_dim, hid_dim), dtype=np.float64)
    B_traj = np.zeros((T, K, hid_dim, n_inputs), dtype=np.float64)

    for k, tr in enumerate(trial_indices):
        H = H_all[:, tr, :]   # (T, hid_dim)  — post-update states h_1 … h_T
        X = X_all[:, tr, :]   # (T, n_inputs)

        for t in range(T):
            # h_t is the pre-update state for step t:
            #   t=0 -> initial hidden state (zeros)
            #   t>0 -> hid_seq[t-1]  (post-update from previous step)
            h_t = np.zeros(hid_dim) if t == 0 else H[t - 1]

            r     = h_t @ Whh.T + X[t] @ Wih.T + bias   # (hid_dim,)
            fprime = _activation_derivative(nonlin, r)    # (hid_dim,)

            D = fprime[:, None]                          # (hid_dim, 1) for broadcasting
            A_traj[t, k] = decay * I + alpha * (D * Whh)
            B_traj[t, k] =             alpha * (D * Wih)

    if return_single:
        return A_traj[:, 0], B_traj[:, 0]
    return A_traj, B_traj