"""
analysis/fixed_point_finder.py
-------------------------------
Gradient-based fixed-point finder for RNN cells of the form h' = F(h, u).

Fixed points satisfy F(h*, u) = h* for a constant input u. They are found by
minimising the per-state loss q = ||F(h, u) - h||^2 with Adam and a
ReduceLROnPlateau scheduler.

Typical workflow
----------------
1. Instantiate `CustomFixedPointFinder` with a trained RNN cell.
2. Call `find_fixed_points` with seed states and their corresponding inputs.
3. Optionally call `compute_jacobian` or `compute_input_jacobian` on the
   returned fixed points to characterise local dynamics.
"""

import numpy as np
import torch
from torch.autograd.functional import jacobian


class CustomFixedPointFinder:
    """Gradient-based fixed-point finder for an RNN cell h' = F(h, u).

    Parameters
    ----------
    rnn : torch.nn.Module
        Recurrent module whose forward signature is
        `(inputs, h0) -> (h_seq, h_final)`, called with a single time step.
    tol_q : float
        Convergence threshold on the per-state loss q = ||F(h) - h||^2.
    tol_dq : float
        Convergence threshold on the change in q between successive steps.
    max_iters : int
        Maximum gradient steps per initial state.
    lr_init : float
        Initial learning rate for the hidden-state optimiser.
    lr_patience : int
        Number of epochs with no improvement before reducing the learning rate.
    lr_factor : float
        Factor by which the learning rate is reduced on plateau.
    lr_cooldown : int
        Epochs to wait before resuming normal operation after a lr reduction.
    device : torch.device, optional
        Computation device. Defaults to the device of the RNN parameters.
    """

    def __init__(
        self,
        rnn: torch.nn.Module,
        tol_q: float = 1e-8,
        tol_dq: float = 1e-8,
        max_iters: int = 5000,
        lr_init: float = 1e-2,
        lr_patience: int = 5,
        lr_factor: float = 0.5,
        lr_cooldown: int = 0,
        device: torch.device = None,
    ):
        self.rnn = rnn
        self.tol_q = tol_q
        self.tol_dq = tol_dq
        self.max_iters = max_iters
        self.device = device if device is not None else next(rnn.parameters()).device
        self.lr_init = lr_init
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_cooldown = lr_cooldown

    # Fixed-point search

    def find_fixed_points(self, initial_states: np.ndarray, inputs: np.ndarray):
        """Find fixed points by minimising ||F(h, u) - h||^2.

        RNN weights are frozen during the optimisation; only the hidden
        states are updated.

        Parameters
        ----------
        initial_states : np.ndarray, shape (N, hidden_size)
            Seed hidden states.
        inputs : np.ndarray, shape (N, input_dim)
            Constant input held fixed during each solve.

        Returns
        -------
        x_star : np.ndarray, shape (N, hidden_size)
            Converged hidden states (fixed-point candidates).
        F_xstar : np.ndarray, shape (N, hidden_size)
            One-step RNN output evaluated at x_star: F(h*, u).
        """
        for p in self.rnn.parameters():
            p.requires_grad = False
        self.rnn.eval()

        N, _ = initial_states.shape

        # inputs shaped as (seq_len=1, batch=N, input_dim)
        u_t = torch.from_numpy(inputs).float().to(self.device).unsqueeze(0)
        # hidden state is the leaf variable being optimised
        h = torch.from_numpy(initial_states).float().to(self.device)
        h.requires_grad_(True)

        optimizer = torch.optim.Adam([h], lr=self.lr_init)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.lr_factor,
            patience=self.lr_patience,
            cooldown=self.lr_cooldown,
        )

        q_prev = torch.full((N,), float("inf"), device=self.device)

        for it in range(1, self.max_iters + 1):
            optimizer.zero_grad()

            h_seq, _ = self.rnn(u_t, h)   # h_seq: (1, N, H)
            h1 = h_seq[0]                 # (N, H)

            q_b = 0.5 * (h1 - h).pow(2).sum(dim=1)  # per-state loss (N,)
            q_mean = q_b.mean()

            with torch.no_grad():
                dq = (q_b - q_prev).abs()
                if it > 1 and torch.all((dq < self.tol_dq) | (q_b < self.tol_q)):
                    print(f"Converged at iteration {it}.")
                    break

            q_mean.backward()
            optimizer.step()
            scheduler.step(q_mean.item())
            q_prev = q_b.detach()

        x_star = h.detach().cpu().numpy()      # (N, H)
        F_xstar = h1.detach().cpu().numpy()    # (N, H)
        return x_star, F_xstar

    # Jacobian analysis

    def compute_jacobian(self, x_star: np.ndarray, inputs: np.ndarray):
        """Compute the state Jacobian dF/dh at each fixed point.

        Parameters
        ----------
        x_star : np.ndarray, shape (N, hidden_size)
            Fixed-point hidden states, e.g. from `find_fixed_points`.
        inputs : np.ndarray, shape (N, input_dim)
            Constant inputs corresponding to each fixed point.

        Returns
        -------
        Js : list of np.ndarray, each shape (H, H)
            Jacobian matrices dF/dh evaluated at each fixed point.
        eigs : list of np.ndarray, each shape (H,)
            Eigenvalues of each Jacobian (may be complex).
        """
        u_t = torch.from_numpy(inputs).float().to(self.device).unsqueeze(0)  # (1, N, D)
        x_t = torch.from_numpy(x_star).float().to(self.device)               # (N, H)

        Js, eigs = [], []
        for i in range(x_t.size(0)):
            hi = x_t[i]           # (H,)
            ui = u_t[:, i:i+1, :] # (1, 1, D)

            def F_map(h_vec):
                # h_vec: (H,) → (1, 1, H) for the RNN call
                h0 = h_vec.unsqueeze(0).unsqueeze(0)
                _, h_final = self.rnn(ui, h0)
                return h_final.squeeze(0).squeeze(0)  # (H,)

            J = jacobian(F_map, hi)  # (H, H)
            J_np = J.detach().cpu().numpy()
            Js.append(J_np)
            eigs.append(np.linalg.eigvals(J_np))

        return Js, eigs

    def compute_input_jacobian(self, x_star: np.ndarray, inputs: np.ndarray):
        """Compute the input Jacobian dF/du at each fixed point.

        Parameters
        ----------
        x_star : np.ndarray, shape (N, hidden_size)
            Fixed-point hidden states, e.g. from `find_fixed_points`.
        inputs : np.ndarray, shape (N, input_dim)
            Constant inputs corresponding to each fixed point.

        Returns
        -------
        J_us : list of np.ndarray, each shape (H, D)
            Input Jacobian matrices dF/du at each fixed point.
        s_us : list of np.ndarray
            Singular values of each input Jacobian.
        """
        u_t = torch.from_numpy(inputs).float().to(self.device).unsqueeze(0)  # (1, N, D)
        x_t = torch.from_numpy(x_star).float().to(self.device)               # (N, H)

        J_us, s_us = [], []
        for i in range(x_t.size(0)):
            hi = x_t[i:i+1]                 # (1, H)
            ui = u_t[:, i:i+1, :].clone()   # (1, 1, D)
            ui.requires_grad_(True)

            def F_wrt_u(u_seq):
                # u_seq: (1, 1, D), hi is held constant
                _, h_next = self.rnn(u_seq, hi)
                return h_next  # (1, H)

            J_full = jacobian(F_wrt_u, ui)   # (1, H, 1, 1, D)
            J_u_np = J_full.squeeze().detach().cpu().numpy()  # (H, D)
            J_us.append(J_u_np)
            s_us.append(np.linalg.svd(J_u_np, compute_uv=False))

        return J_us, s_us


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_state_and_input_at_time(hidden_sequence, inputs, time_step):
    """Extract hidden state and input at a given time step.
    """
    return hidden_sequence[time_step, :, :], inputs[time_step, :, :]