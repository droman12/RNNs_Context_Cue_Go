import math
import torch
from torch import nn
from input_target import get_inputs


def compute_spectral_radius(W: torch.Tensor) -> float:
    '''Compute the spectral radius (max abs eigenvalue) of W.'''
    eigvals = torch.linalg.eigvals(W)
    return eigvals.abs().max().item()


class RNNCell_base(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        nonlinearity,
        bias,
        noise_std,
        input_noise_std,
        w_rec_init,
        spectral_radius, 
    ):
        super().__init__()
        self.input_size       = input_size
        self.hidden_size      = hidden_size
        self.nonlinearity     = nonlinearity
        self.noise_std        = noise_std
        self.input_noise_std  = input_noise_std
        self.w_rec_init       = w_rec_init
        self.spectral_radius  = spectral_radius

        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # --- Input-to-hidden init ---
        # if isinstance(self.nonlinearity, nn.Tanh):
        #     gain = nn.init.calculate_gain('tanh')
        #     nn.init.xavier_uniform_(self.weight_ih, gain=gain)
        # else:
        nn.init.kaiming_uniform_(self.weight_ih, a=math.sqrt(5))
        #nn.init.uniform_(self.weight_ih, -1.0, 1.0)

        # --- Hidden-to-hidden init ---
        with torch.no_grad():
            N = self.hidden_size
            # if no spectral_radius was given, default g→1.0
            g = self.spectral_radius if self.spectral_radius is not None else 1.0

            if self.w_rec_init == 'diag':
                # exactly W = g * I
                I = torch.eye(N, device=self.weight_hh.device)
                self.weight_hh.copy_(I * g)

            elif self.w_rec_init == 'randortho':
                nn.init.orthogonal_(self.weight_hh)
                if self.spectral_radius is not None:
                    rho = compute_spectral_radius(self.weight_hh)
                    self.weight_hh.mul_(g / (rho + 1e-12))

            elif self.w_rec_init == 'randgauss':
                # W_ij ~ N(0, (g/√N)^2)
                std = g / math.sqrt(N)
                nn.init.normal_(self.weight_hh, mean=0.0, std=std)

            else:
                # fallback: Kaiming
                if isinstance(self.nonlinearity, nn.Tanh):
                    gain_h = nn.init.calculate_gain('tanh')
                    nn.init.xavier_uniform_(self.weight_hh, gain=gain_h)
                else:
                    nn.init.kaiming_uniform_(self.weight_hh, a=math.sqrt(5))
                if self.spectral_radius is not None:
                    rho = compute_spectral_radius(self.weight_hh)
                    self.weight_hh.mul_(g / (rho + 1e-12))

        # --- Bias init (uniform in [−1/√fan_in, +1/√fan_in]) ---
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_ih)
            bound     = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

class RNNCell(RNNCell_base):
    def __init__(self, *args, decay: float = 0.9, decay_zeta: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.decay       = decay
        self.decay_zeta  = decay_zeta

    def forward(self, x, h, zeta_prev):
        # recurrent noise
        if self.noise_std > 0.0:
            g_noise = torch.randn_like(zeta_prev)
            coeff   = math.sqrt(2.0 * self.decay_zeta * (self.noise_std ** 2))
            zeta_t  = (1.0 - self.decay_zeta) * zeta_prev + coeff * g_noise
        else:
            zeta_t = torch.zeros_like(zeta_prev)

        r = x @ self.weight_ih.t() + h @ self.weight_hh.t()
        if self.bias is not None:
            r = r + self.bias
        r = r + zeta_t

        a_next = self.nonlinearity(r)
        h_next = self.decay * h + (1.0 - self.decay) * a_next
        return h_next, zeta_t


class RNNLayer(nn.Module):
    def __init__(self, *cell_args, dropcon_hh = 0.0 ,**cell_kwargs):
        super().__init__()
        self.dropcon_hh = dropcon_hh
        self.rnncell = RNNCell(*cell_args, **cell_kwargs)

    def forward(self, inputs, hidden0):
        
        T, B, _ = inputs.shape
        H       = self.rnncell.hidden_size
        h        = hidden0.squeeze(0)

        zeta     = torch.zeros_like(h)
        outputs  = []
        sigma_in = self.rnncell.input_noise_std

        for t in range(T):
            x_t   = inputs[t]
            noise = torch.randn_like(x_t) * sigma_in
            # if you have exactly 3 input channels, amplify the first
            if x_t.shape[1] == 3:
                noise[:, 0] *= 5.0
            x_noisy = x_t + noise

            h, zeta = self.rnncell(x_noisy, h, zeta)
            outputs.append(h)

        return torch.stack(outputs, dim=0), h.unsqueeze(0)


class Model(nn.Module):
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
            nonlin = nn.Softplus(beta=1.0, threshold=0.5)
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
        )

        self.readout = nn.Linear(n_rnn, n_out, bias=False)
        self.n_rnn   = n_rnn

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
        self.model  = Model(hp, RNNLayer)

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
