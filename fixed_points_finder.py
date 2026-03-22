import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd.functional import jacobian

class CustomFixedPointFinder:
    def __init__(self,
                 rnn: torch.nn.Module,
                 tol_q: float = 1e-8,
                 tol_dq: float = 1e-8,
                 max_iters: int = 5000,
                 lr_init: float = 1e-2,
                 lr_patience: int = 5,
                 lr_factor: float = 0.5,
                 lr_cooldown: int = 0,
                 device: torch.device = None):
        """
        A simple fixed‐point finder for RNN cells of the form
          h_new = F(h, u)
        by minimizing ||F(h,u) − h||^2 via gradient‐based optimization.

        Args:
          rnn:        the recurrent module, whose forward(inputs, h0)
                      returns (h_seq, h_final).  We call it with
                      a single time step.
          tol_q:      convergence threshold on per‐state loss q = ||F(h)−h||^2
          tol_dq:     convergence threshold on change in q between steps
          max_iters:  maximum gradient‐steps per initial state
          lr_init:    initial learning rate for the hidden‐state optimiser
          lr_patience, lr_factor, lr_cooldown:
                      scheduler parameters for ReduceLROnPlateau
          device:     torch.device (defaults to rnn parameters’ device)
        """
        self.rnn       = rnn
        self.tol_q     = tol_q
        self.tol_dq    = tol_dq
        self.max_iters = max_iters

        self.device    = device if device is not None else next(rnn.parameters()).device

        # optimizer hyperparams
        self.lr_init      = lr_init
        self.lr_patience  = lr_patience
        self.lr_factor    = lr_factor
        self.lr_cooldown  = lr_cooldown

    def find_fixed_points(self,
                          initial_states: np.ndarray,
                          inputs: np.ndarray
                         ):
        """
        Args:
          initial_states: array (N, hidden_size) of seed states
          inputs:         array (N, input_dim) of *constant* inputs
                          to hold during the fixed‐point solve

        Returns:
          x_star:   array (N, hidden_size) of converged hidden‐states
          F_xstar:  array (N, hidden_size) of one‐step rnn(h*,u)
        """
        # freeze RNN weights
        for p in self.rnn.parameters():
            p.requires_grad = False
        self.rnn.eval()

        # prepare tensors
        N, H = initial_states.shape
        _, B, _ = 1, N, inputs.shape[1]  # seq_len=1, batch=N, input_dim

        # inputs: make (seq_len=1, batch=N, input_dim)
        u_t = torch.from_numpy(inputs).float().to(self.device).unsqueeze(0)
        # hidden: leaf tensor we optimize
        h = torch.from_numpy(initial_states).float().to(self.device)
        h.requires_grad_(True)

        # optimizer & scheduler
        optimizer = torch.optim.Adam([h], lr=self.lr_init)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.lr_factor,
            patience=self.lr_patience,
            cooldown=self.lr_cooldown
        )

        q_prev = torch.full((N,), float('inf'), device=self.device)
        for it in range(1, self.max_iters+1):
            optimizer.zero_grad()

            # one‐step through RNN
            # forward expects (seq_len, batch, input_dim), h0=(batch,hidden)
            self.rnn = self.rnn
            h_seq, _ = self.rnn(u_t, h)        # h_seq: (1, N, H)
            h1 = h_seq[0]                      # (N, H)

            # loss per batch‐element
            diff   = h1 - h                    # (N, H)
            q_b    = 0.5 * diff.pow(2).sum(dim=1)  # (N,)
            q_mean = q_b.mean()                # scalar
            
            # check termination
            with torch.no_grad():
                dq = (q_b - q_prev).abs()
                if it>1 and torch.all((dq < self.tol_dq) | (q_b < self.tol_q)):
                    print("Converged at iteration", it+1)
                    break

            # backward + step
            q_mean.backward()
            optimizer.step()
            scheduler.step(q_mean.item())

            q_prev = q_b.detach()

        # gather results
        x_star = h.detach().cpu().numpy()      # (N, H)
        F_xstar = h1.detach().cpu().numpy()   # (N, H)
        return x_star, F_xstar
    
    def compute_jacobian(self, x_star, inputs):
        device = self.device
        u_t = torch.from_numpy(inputs).float().to(device).unsqueeze(0)   # (1, N, D)
        x_t = torch.from_numpy(x_star).float().to(device)                # (N, H)

        Js, eigs = [], []
        for i in range(x_t.size(0)):
            # --- fix shapes here ---
            hi = x_t[i]                 # (H,)
            ui = u_t[:, i:i+1, :]       # (1, 1, D)

            # F_map: R^H → R^H, holding ui fixed
            def F_map(h_vec):
                # h_vec: (H,) → wrap into (1, 1, H)
                h0 = h_vec.unsqueeze(0).unsqueeze(0)  # (1, 1, H)
                h_seq, h_final = self.rnn(ui, h0)
                # h_final: (1, 1, H) → squeeze back to (H,)
                return h_final.squeeze(0).squeeze(0)

            # Now jacobian(F_map) yields a (H, H) tensor
            J = jacobian(F_map, hi)          # shape (H, H)
            J_np = J.detach().cpu().numpy()
            Js.append(J_np)

            eigvals = np.linalg.eigvals(J_np)
            eigs.append(eigvals)

        return Js, eigs

    
    def compute_input_jacobian(self, x_star, inputs):
        u_t = torch.from_numpy(inputs).float().to(self.device).unsqueeze(0)  # (1, N, D)
        x_t = torch.from_numpy(x_star).float().to(self.device)              # (N, H)

        J_us, s_us = [], []
        for i in range(x_t.size(0)):
            # grab the fixed state and its corresponding input
            hi = x_t[i:i+1]                # (1, H)
            ui = u_t[:, i:i+1, :].clone()  # (1, 1, D)
            ui.requires_grad_(True)

            # map from input u to next hidden state F(h*=hi, u)
            def F_wrt_u(u_seq):
                # u_seq: (1,1,D), hi closed over as constant
                h_seq, h_next = self.rnn(u_seq, hi)
                return h_next              # (1, H)

            # compute dF/du at this point: result shape (1, H, 1, 1, D)
            J_full = jacobian(F_wrt_u, ui)

            # squeeze out all singleton dims → (H, D)
            J_u = J_full.squeeze()
            
            J_u_np = J_u.detach().cpu().numpy()
            J_us.append(J_u_np)

            s = np.linalg.svd(J_u_np, compute_uv=False)
            s_us.append(s)

        return J_us, s_us

def get_paired(hidden_sequence, inputs, time_step):
    hidden = hidden_sequence[time_step, :, :]
    inputs = inputs[time_step, :, :]
    return hidden, inputs


def analytical_jacobian():
    return 