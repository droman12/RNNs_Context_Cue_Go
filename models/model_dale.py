import torch
from torch import nn
import torch.nn.functional as F
from input_target import get_inputs
import math


def compute_spectral_radius(W: torch.Tensor) -> float:
    W_cpu = W.detach().cpu()
    eigvals = torch.linalg.eigvals(W_cpu)
    return eigvals.abs().max().item()


class DaleRNNCell_base(nn.Module):
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
        dropout_hidden: float = 0.0,
        frac_excit: float = 0.7,               # fraction of excitatory units
    ):
        super().__init__()
        self.input_size       = input_size
        self.hidden_size      = hidden_size
        self.nonlinearity     = nonlinearity
        self.noise_std        = noise_std
        self.input_noise_std  = input_noise_std
        self.w_rec_init       = w_rec_init
        self.spectral_radius  = spectral_radius

        # raw parameters
        # raw recurrent weights
        self.weight_hh_raw = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # input and bias
        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias', None)


        # dropout on recurrent outputs
        self.dropout_hh = nn.Dropout(p=dropout_hidden)

        # Dale's law: create sign mask for hidden units
        nE = int(frac_excit * hidden_size)
        signs = torch.cat([torch.ones(nE), -torch.ones(hidden_size - nE)])
        # optional shuffle:
        perm = torch.randperm(hidden_size)
        signs = signs[perm]
        self.register_buffer('sign_rec', signs)  # shape (H,)

        self.reset_parameters()

    def reset_parameters(self):
        # input-to-hidden
        nn.init.kaiming_uniform_(self.weight_ih, a=math.sqrt(5))

        # hidden-to-hidden
        with torch.no_grad():
            H = self.weight_hh_raw.size(0)
            if self.w_rec_init == 'randortho':
                nn.init.orthogonal_(self.weight_hh_raw)
            elif self.w_rec_init == 'randgauss':
                nn.init.normal_(self.weight_hh_raw, 0.0, 1.0/math.sqrt(H))
            else:
                gain = (nn.init.calculate_gain('tanh')
                        if isinstance(self.nonlinearity, nn.Tanh)
                        else 1.0)
                nn.init.xavier_uniform_(self.weight_hh_raw, gain=gain)

            if self.spectral_radius is not None:
                W_signed = self._signed_W_hh(self.weight_hh_raw)
                rho = compute_spectral_radius(W_signed)
                self.weight_hh_raw.mul_(self.spectral_radius / (rho+1e-12))


        # bias
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_ih)
            bound     = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    def _signed_W_hh(self, W_raw: torch.Tensor) -> torch.Tensor:
        """
        Apply Dale mask on columns: each column j is
        all + if neuron j excitatory, all – if inhibitory.
        """
        # W_raw: (H_out, H_in)
        # We want columns = presyn outputs, so:
        return torch.abs(W_raw) * self.sign_rec.view(1, -1)



class RNNCell(DaleRNNCell_base):
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

        # lesion and sign-constrain hidden outputs
        h_lesioned = self.dropout_hh(h)
        # enforce Dale's sign on recurrent weights
        W_hh_signed = self._signed_W_hh(self.weight_hh_raw)
        r = x @ self.weight_ih.t() + h_lesioned @ W_hh_signed.t() + self.bias
        if self.bias is not None:
            r = r + self.bias
        r = r + zeta_t

        a_next = self.nonlinearity(r)
        h_next = self.decay * h + (1.0 - self.decay) * a_next
        return h_next, zeta_t


class RNNLayer(nn.Module):
    def __init__(self, *cell_args, **cell_kwargs):
        super().__init__()
        self.rnncell = RNNCell(*cell_args, **cell_kwargs)

    def forward(self, inputs, hidden0):
        T, B, _ = inputs.shape
        h        = hidden0.squeeze(0)
        zeta     = torch.zeros_like(h)
        outputs  = []
        sigma_in = self.rnncell.input_noise_std

        for t in range(T):
            x_t   = inputs[t]
            noise = torch.randn_like(x_t) * sigma_in
            if x_t.shape[1] == 3:
                noise[:, 0] *= 5.0
            x_noisy = x_t + noise

            h, zeta = self.rnncell(x_noisy, h, zeta)
            outputs.append(h)

        return torch.stack(outputs, dim=0), h.unsqueeze(0)


class Model(nn.Module):
    def __init__(self, hp, RNNLayer):
        super().__init__()
        n_in, n_rnn, n_out    = hp['n_input'], hp['n_rnn'], hp['n_output']
        act_name              = hp['activation'].lower()
        decay, decay_zeta     = hp['decay'], hp['decay_zeta']
        bias                  = hp['bias']
        noise_std             = hp['sigma_rec']
        input_noise_std       = hp['sigma_x']
        w_rec_init            = hp['w_rec_init']
        spectral_radius       = hp['spectral_radius']
        dropout_hidden        = hp.get('dropout_hidden', 0.0)
        dropout_readout       = hp.get('dropout_readout', 0.0)
        frac_excit            = hp.get('frac_excit', 0.7)

        # choose nonlinearity
        if act_name == 'relu':
            nonlin = nn.ReLU()
        elif act_name == 'tanh':
            nonlin = nn.Tanh()
        elif act_name == 'softplus':
            nonlin = nn.Softplus(beta=1.0, threshold=0.5)
        else:
            raise ValueError(f"Unknown activation {act_name}")

        # build RNN with Dale's law
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
            dropout_hidden   = dropout_hidden,
            frac_excit       = frac_excit,
        )

        # dropout and readout
        self.dropout_readout = nn.Dropout(p=dropout_readout)
        self.readout         = nn.Linear(n_rnn, n_out, bias=False)
        self.n_rnn           = n_rnn

    def forward(self, x):
        device  = x.device
        batch   = x.size(1)
        hidden0 = torch.zeros(1, batch, self.n_rnn, device=device)

        hid_seq, last_h = self.rnn(x, hidden0)

        # apply dropout on hidden->readout
        hid_d = self.dropout_readout(hid_seq)
        # enforce Dale on readout weights
        sign_rec = self.rnn.rnncell.sign_rec
        W_ro_signed = sign_rec.unsqueeze(0) * torch.abs(self.readout.weight)
        z = torch.matmul(hid_d, W_ro_signed.t())
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

        # regularization as before
        reg_h = torch.tensor(0.0, device=pred.device)
        if self.hp.get('l1_h', 0.0) > 0.0:
            reg_h = hid_seq.abs().mean() * self.hp['l1_h']
        if self.hp.get('l2_h', 0.0) > 0.0:
            reg_h = reg_h + (hid_seq ** 2).mean() * self.hp['l2_h']

        reg_w = torch.tensor(0.0, device=pred.device)
        for param in self.model.parameters():
            if self.hp.get('l1_weight', 0.0) > 0.0:
                reg_w = reg_w + param.abs().sum() * self.hp['l1_weight']
            if self.hp.get('l2_weight', 0.0) > 0.0:
                reg_w = reg_w + (param ** 2).sum() * self.hp['l2_weight']

        return data_loss, (reg_h + reg_w)

    def forward(self, batch_size=None, X=None, Y=None):
        if X is None or Y is None:
            X, Y = self.generate_trials(batch_size)
        pred, hid_seq, _ = self.model(X)
        loss_data, loss_reg = self.calculate_loss(pred, hid_seq, Y)
        total_loss = loss_data + loss_reg
        return total_loss, loss_data, loss_reg, pred.detach().cpu().numpy(), hid_seq.detach().cpu().numpy()

