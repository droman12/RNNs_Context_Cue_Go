import math
from typing import Optional, Literal
# from mossy_cereb import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from input_target import *
from cerebellum import *


@torch.no_grad()
def compute_spectral_radius(W: torch.Tensor) -> float:
    '''Compute the spectral radius (max abs eigenvalue) of W.'''
    eigvals = torch.linalg.eigvals(W)
    return eigvals.abs().max().item()

class RNNCellDale(nn.Module):
    """
    Dale-constrained recurrent cell (columns are presynaptic neurons):
      - Columns flagged excitatory are clamped >= 0; inhibitory columns <= 0.
      - Optional spectral-radius rescaling on W_hh.
      - Optional structural nonrecurrent mode (W_hh is zero and grads off).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        nonlinearity: nn.Module,
        bias: bool = True,
        noise_std: float = 0.0,
        input_noise_std: float = 0.0,
        w_rec_init: Optional[
            Literal["dale", "uniform", "gaussian", "randgauss", "zero", "diag", "randortho"]
        ] = "dale",
        spectral_radius: Optional[float] = None,
        exc_frac: float = 0.1,
        randomize_ei: bool = True,
        seed: Optional[int] = None,
        nonrecurrent: bool = False,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.nonlinearity = nonlinearity
        self.noise_std = float(noise_std)
        self.input_noise_std = float(input_noise_std)
        self.w_rec_init = (w_rec_init.lower() if isinstance(w_rec_init, str) else w_rec_init)
        self.spectral_radius = None if spectral_radius is None else float(spectral_radius)
        self.exc_frac = float(max(0.0, min(1.0, exc_frac)))
        self.randomize_ei = bool(randomize_ei)
        self._seed = seed
        self.nonrecurrent = bool(nonrecurrent)

        # Parameters
        self.weight_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(hidden_size))
        else:
            self.register_parameter("bias", None)

        # E/I assignment (columns = presynaptic neurons)
        N = self.hidden_size
        n_exc = int(round(self.exc_frac * N))
        n_exc = max(0, min(N, n_exc))
        if self.randomize_ei:
            g = torch.Generator()
            if seed is not None:
                g.manual_seed(int(seed))
            perm = torch.randperm(N, generator=g)
        else:
            perm = torch.arange(N)
        exc_cols = perm[:n_exc]
        exc_mask = torch.zeros(N, dtype=torch.bool)
        exc_mask[exc_cols] = True
        inh_mask = ~exc_mask
        self.register_buffer("exc_cols_mask", exc_mask, persistent=True)
        self.register_buffer("inh_cols_mask", inh_mask, persistent=True)

        # Structural mask
        rec_mask = torch.ones(N, N, dtype=self.weight_hh.dtype, device=self.weight_hh.device)
        if self.nonrecurrent:
            rec_mask.zero_()
        self.register_buffer("rec_mask", rec_mask, persistent=True)

        self.reset_parameters()

        # If nonrecurrent, keep W_hh frozen for a tiny speed win
        if self.nonrecurrent:
            self.weight_hh.requires_grad_(False)

    def _dale_reinit_weight_hh(self) -> None:
        with torch.no_grad():
            N = self.hidden_size
            u = 1.0 / math.sqrt(max(1, N))
            self.weight_hh.zero_()

            if not self.nonrecurrent:
                init_key = self.w_rec_init
                if init_key in (None, "dale", "uniform"):
                    # Positive on excit columns, negative on inhib columns
                    self.weight_hh[:, self.exc_cols_mask].uniform_(0.0, u)
                    self.weight_hh[:, self.inh_cols_mask].uniform_(-u, 0.0)

                elif init_key in ("gaussian", "randgauss"):
                    self.weight_hh.normal_(0.0, u / 2.0)
                    self.weight_hh[:, self.exc_cols_mask].clamp_(min=0.0)
                    self.weight_hh[:, self.inh_cols_mask].clamp_(max=0.0)

                elif init_key == "diag":
                    sign = torch.where(self.exc_cols_mask, 1.0, -1.0).to(self.weight_hh.dtype)
                    self.weight_hh.copy_(torch.diag(sign.to(self.weight_hh.device)))

                elif init_key == "randortho":
                    nn.init.orthogonal_(self.weight_hh)
                    self.weight_hh[:, self.exc_cols_mask].clamp_(min=0.0)
                    self.weight_hh[:, self.inh_cols_mask].clamp_(max=0.0)

                elif init_key == "zero":
                    pass

                else:
                    raise ValueError(f"Unknown w_rec_init: {self.w_rec_init}")

                # Spectral radius scaling (only if matrix nonzero)
                if self.spectral_radius is not None:
                    rho = compute_spectral_radius(self.weight_hh)
                    if rho > 0.0:
                        self.weight_hh.mul_(float(self.spectral_radius) / (rho + 1e-12))

            # Apply structural mask last
            self.weight_hh.mul_(self.rec_mask)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_ih, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_ih)
            bound = 1.0 / math.sqrt(max(1, fan_in))
            nn.init.uniform_(self.bias, -bound, bound)
        self._dale_reinit_weight_hh()

    @torch.no_grad()
    def enforce_dales_law(self, rescale_to_spectral_radius: bool = True) -> None:
        """
        Project W_hh back onto Dale constraints and (optionally) re-scale.
        """
        if self.nonrecurrent:
            self.weight_hh.zero_()
            return

        self.weight_hh[:, self.exc_cols_mask].clamp_(min=0.0)
        self.weight_hh[:, self.inh_cols_mask].clamp_(max=0.0)
        self.weight_hh.mul_(self.rec_mask)

        if rescale_to_spectral_radius and (self.spectral_radius is not None):
            rho = compute_spectral_radius(self.weight_hh)
            if rho > 0.0:
                self.weight_hh.mul_(float(self.spectral_radius) / (rho + 1e-12))

class RNNCell(RNNCellDale):
    def __init__(self, *args, decay: float = 0.9, decay_zeta: float = 0.5, passthrough_input: bool = False, **kwargs):
        super().__init__(*args, **kwargs)  # <-- fixed **kwargs
        self.decay = decay
        self.decay_zeta = decay_zeta
        self.passthrough_input = bool(passthrough_input)

        if self.passthrough_input:
            # Keep bias as tonic state bias; drop weight_ih
            self.register_parameter("weight_ih", None)

    def forward(self, x, h, zeta_prev):
        if self.input_noise_std > 0.0:
            x = x + self.input_noise_std * torch.randn_like(x)  # <-- randn_like

        if self.noise_std > 0.0:
            g_noise = torch.randn_like(zeta_prev)               # <-- randn_like
            alpha = 1.0 - self.decay_zeta
            coeff = math.sqrt(max(0.0, 1.0 - alpha * alpha)) * self.noise_std
            zeta_t = alpha * zeta_prev + coeff * g_noise
        else:
            zeta_t = torch.zeros_like(zeta_prev)

        r = x if (self.weight_ih is None) else (x @ self.weight_ih.t())

        if not self.nonrecurrent:
            r = r + h @ self.weight_hh.t()

        if self.bias is not None:
            r = r + self.bias
        r = r + zeta_t

        a_next = self.nonlinearity(r)
        h_next = self.decay * h + (1.0 - self.decay) * a_next
        return h_next, zeta_t


class RNNLayer(nn.Module):
    def __init__(self, *cell_args, **cell_kwargs):
        super().__init__()                                   # <-- added
        self.rnncell = RNNCell(*cell_args, **cell_kwargs)

    def forward(self, inputs, hidden0):
        # inputs: (T, B, I); hidden0: (1, B, H) or (B, H)
        T, B, _ = inputs.shape
        h = hidden0.squeeze(0) if hidden0.dim() == 3 else hidden0

        zeta = torch.zeros_like(h)
        outputs = []

        for t in range(T):
            x_t = inputs[t]
            h, zeta = self.rnncell(x_t, h, zeta)
            outputs.append(h)                               

        return torch.stack(outputs, dim=0), h.unsqueeze(0)
    
class ExcitatoryReadout(nn.Module):
    def __init__(self, n_rnn, n_out, exc_mask):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_out, n_rnn))
        self.b = nn.Parameter(torch.zeros(n_out))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.register_buffer("exc_mask", exc_mask.float())  # shape (n_rnn,)

    def forward(self, h):
        W_eff = F.softplus(self.W) * self.exc_mask
        if h.dim() == 3:  # (T,B,H)
            z = torch.einsum('tbh,oh->tbo', h, W_eff)
            return z + self.b.view(1,1,-1)
        return h @ W_eff.T + self.b
    

class Cortex(nn.Module): 
    def __init__(self, hp, RNNLayer):
        super().__init__()
        n_in, n_rnn, n_out = hp['n_input'], hp['n_rnn'], hp['n_output']
        act_name           = hp['activation'].lower()
        decay              = hp['decay']
        decay_zeta         = hp['decay_zeta']
        bias               = hp['bias']
        noise_std          = hp['sigma_rec']
        input_noise_std    = hp['sigma_x']
        w_rec_init         = hp['w_rec_init']
        spectral_radius    = hp['spectral_radius']

        if act_name == 'relu':
            nonlin = nn.ReLU()
        elif act_name == 'tanh':
            nonlin = nn.Tanh()
        elif act_name == 'softplus':
            nonlin = nn.Softplus()
        else:
            raise ValueError('Unknown activation {}'.format(hp['activation']))
        
        self.rnn = RNNLayer(
            n_in, n_rnn,
            nonlinearity     = nonlin,
            decay            = decay,
            decay_zeta       = decay_zeta,
            bias             = bias,
            noise_std        = noise_std,
            input_noise_std  = input_noise_std,
            w_rec_init       = w_rec_init,
            spectral_radius  = spectral_radius,
            exc_frac = 0.7,
            randomize_ei = True, 
            nonrecurrent = False, 
            passthrough_input = False
        )

        exc_mask = self.rnn.rnncell.exc_cols_mask  # (H,)
        self.n_rnn   = n_rnn
        self.readout = ExcitatoryReadout(n_rnn, n_out, exc_mask)
        
    
    def forward(self, x):
        device  = x.device
        batch   = x.size(1)
        hidden0 = torch.zeros(1, batch, self.n_rnn, device=device)
        hid_seq, last_h= self.rnn(x, hidden0)
        # compute outputs
        z = self.readout(hid_seq)
        return z, hid_seq, last_h




class Run_Model(nn.Module):
    def __init__(self, hp, params, RNNLayer):
        super().__init__()
        self.hp     = hp
        self.params = params
        self.model  = Cortex(hp, RNNLayer)

        if hp['loss_type'] == 'lsq':
            self.loss_fnc = nn.MSELoss(reduction='none')
        else:
            self.loss_fnc = nn.CrossEntropyLoss(reduction='none')

    def generate_trials(self, batch_size=None):
        p = self.params.copy()
        if batch_size is not None:
            p['ntrials'] = batch_size
        seed             = self.hp['seed']
        single_pulse_time= self.hp['single_pulse_time']
        X, Y, _, _       = get_inputs(p, seed=seed, single_pulse_time=single_pulse_time)
        X_t = torch.from_numpy(X).float()
        Y_t = torch.from_numpy(Y).float()
        device = next(self.model.parameters()).device
        return X_t.to(device), Y_t.to(device)

    def calculate_loss(self, pred, hid_seq, target):
        mask         = ~torch.isnan(target)
        target_clean = torch.where(mask, target, torch.zeros_like(target))

        # handle regression vs. classification
        if isinstance(self.loss_fnc, nn.CrossEntropyLoss):
            T, B, C       = pred.shape
            pred_flat     = pred.view(T * B, C)
            target_flat   = target_clean.view(T * B).long()
            mask_flat     = mask.view(T * B)
            losses_flat   = self.loss_fnc(pred_flat, target_flat) * mask_flat.float()
            data_loss     = losses_flat.sum()
        else:
            raw        = self.loss_fnc(pred, target_clean) * mask.float()
            data_loss  = raw.sum()

        # hidden‐activity regularization
        reg_h = torch.tensor(0.0, device=pred.device)
        if self.hp.get('l1_h', 0.0) > 0.0:
            reg_h = hid_seq.abs().mean() * self.hp['l1_h']
        if self.hp.get('l2_h', 0.0) > 0.0:
            reg_h = reg_h + (hid_seq ** 2).mean() * self.hp['l2_h']

        # weight regularization
        reg_w = torch.tensor(0.0, device=pred.device)
        for param in self.model.parameters():
            if self.hp.get('l1_weight', 0.0) > 0.0:
                reg_w = reg_w + param.abs().sum() * self.hp['l1_weight']
            if self.hp.get('l2_weight', 0.0) > 0.0:
                reg_w = reg_w + (param ** 2).sum() * self.hp['l2_weight']
        # manifold orthogonality penalty: ||cov(H) - I||_F^2
        ortho_pen = torch.tensor(0.0, device=pred.device)
        gamma     = self.hp.get('gamma_manifold', 0.0)
        if gamma > 0.0:
            T, B, H = hid_seq.shape
            Hmat    = hid_seq.view(T * B, H)                           # [T*B, H]
            cov     = (Hmat.T @ Hmat) / (T * B)                        # [H, H]
            diff    = cov - torch.eye(H, device=cov.device, dtype=cov.dtype)
            ortho_pen = gamma * (diff.norm(p='fro') ** 2)
        total_reg = reg_h + reg_w + ortho_pen
        return data_loss, total_reg

    def forward(self, batch_size=None, X=None, Y=None):
        if X is None or Y is None:
            X, Y = self.generate_trials(batch_size)
        pred, hid_seq, _ = self.model(X)
        loss_data, loss_reg = self.calculate_loss(pred, hid_seq, Y)
        total_loss = loss_data + loss_reg
        return total_loss, loss_data, loss_reg, pred.detach().cpu().numpy(), hid_seq.detach().cpu().numpy()




class CerebCortex(nn.Module):
    def __init__(self, cerebellum: Cerebellum, cortex: Cortex, use_concat: bool = False, layernorm_dcn: bool = True):
        super().__init__()
        self.cereb = cerebellum 
        self.cortex = cortex
        self.use_concat = use_concat 
        self.ln = nn.LayerNorm(self.cereb.dcn.b_dcn.shape[0]) if layernorm_dcn else nn.Identity()

    def forward(self, mf_seq):
        dcn, g, pc = self.cereb(mf_seq)
        dcn  = self.ln(dcn)
        if self.use_concat: 
            x = torch.cat([mf_seq, dcn], dim=-1)
        else: 
            x = dcn
        z, hseq, last_h = self.cortex(x)
        return z, hseq, last_h, (dcn, g, pc)




# def compute_spectral_radius(W: torch.Tensor, iters: int = 50) -> float:
#     """Power iteration spectral radius estimate (||W|| largest |eig|)."""
#     if W.numel() == 0:
#         return 0.0
#     with torch.no_grad():
#         v = torch.randn(W.shape[1], device=W.device, dtype=W.dtype)
#         v = v / (v.norm() + 1e-12)
#         for _ in range(iters):
#             v = W @ v
#             n = v.norm()
#             if n < 1e-20:
#                 return 0.0
#             v = v / n
#         # Rayleigh quotient for eigenvalue magnitude estimate
#         lam = torch.dot(v, (W @ v))
#         return float(lam.abs().item())
