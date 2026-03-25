import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from input_target import *


class MossyGranuleLayer(nn.Module):
    def __init__(
        self,
        n_mf: int,
        n_gc: int,
        nsyn: int = 4,
        threshold_mode: str = "k_of_n",
        k_active: Optional[int] = None,
        theta_abs: float = 0.0,
        golgi_beta: float = 0.0,
        seed: Optional[int] = 0,
        weight_scale: float = 1.0,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        assert 1 <= nsyn <= n_mf, "nsyn must be in [1, n_mf]"
        self.n_mf, self.n_gc, self.nsyn = n_mf, n_gc, nsyn
        self.threshold_mode = threshold_mode
        self.k_active = k_active if (k_active is not None) else max(1, int(round(0.75 * nsyn)))
        self.theta_abs = float(theta_abs)
        self.golgi_beta = float(golgi_beta)
        self.weight_scale = float(weight_scale)

        g = torch.Generator()
        if seed is not None:
            g.manual_seed(int(seed))
        idx = torch.stack(
            [torch.randperm(n_mf, generator=g)[:nsyn] for _ in range(n_gc)],
            dim=0,
        )
        self.register_buffer("idx", idx, persistent=True)

        W_raw = torch.rand(n_gc, nsyn, generator=g, dtype=dtype) + 1e-3
        W_raw = W_raw / (W_raw.sum(dim=1, keepdim=True) + 1e-12)
        self.register_buffer("W_conn", W_raw * self.weight_scale, persistent=True)

        if threshold_mode == "learned":
            theta0 = float(self.k_active) / float(self.nsyn) * self.weight_scale
            self.theta_param = nn.Parameter(torch.full((n_gc,), theta0, dtype=dtype))
        else:
            self.register_parameter("theta_param", None)

    def _compute_theta(self, x_slice: torch.Tensor) -> torch.Tensor:
        if self.threshold_mode == "k_of_n":
            base = (float(self.k_active) / float(self.nsyn)) * self.weight_scale
            theta_gc = torch.full((self.n_gc,), base, device=x_slice.device, dtype=x_slice.dtype)
        elif self.threshold_mode == "absolute":
            theta_gc = torch.full((self.n_gc,), float(self.theta_abs), device=x_slice.device, dtype=x_slice.dtype)
        elif self.threshold_mode == "learned":
            theta_gc = self.theta_param.to(device=x_slice.device, dtype=x_slice.dtype)
        else:
            raise ValueError("threshold_mode must be one of {'k_of_n','absolute','learned'}")

        if self.golgi_beta != 0.0:
            if x_slice.dim() == 1:
                mean_mf = x_slice.mean()
                theta_gc = theta_gc + self.golgi_beta * mean_mf
            else:
                mean_mf = x_slice.mean(dim=1, keepdim=True)
                theta_gc = theta_gc.unsqueeze(0) + self.golgi_beta * mean_mf
        return theta_gc

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        x_sel = x[:, self.idx]
        w = self.W_conn.unsqueeze(0)
        return (x_sel * w).sum(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2
        theta = self._compute_theta(x)
        pre = self._project(x)
        g = F.relu(pre - theta)
        return g


class PurkinjeLayer(nn.Module):
    def __init__(self, n_gc: int, n_pc: int, pf_fanin: int = 256, seed: Optional[int] = 0):
        super().__init__()
        self.n_gc, self.n_pc = n_gc, n_pc
        self.pf_fanin = int(min(max(1, pf_fanin), n_gc))

        g = torch.Generator()
        if seed is not None:
            g.manual_seed(int(seed))
        mask = torch.zeros(n_pc, n_gc)
        for p in range(n_pc):
            idx = torch.randperm(n_gc, generator=g)[: self.pf_fanin]
            mask[p, idx] = 1.0
        self.register_buffer("mask", mask, persistent=True)

        self.W_pf_pc_raw = nn.Parameter(torch.empty(n_pc, n_gc))
        nn.init.kaiming_uniform_(self.W_pf_pc_raw, a=math.sqrt(5))
        # b_pc starts at zero; PCs will be silent until driven by GCs
        self.b_pc = nn.Parameter(torch.zeros(n_pc))

    def effective_weights(self) -> torch.Tensor:
        # GC->PC connections are excitatory (softplus enforces positivity)
        return F.softplus(self.W_pf_pc_raw) * self.mask

    def forward(self, g_in: torch.Tensor) -> torch.Tensor:
        W = self.effective_weights()
        pc = g_in @ W.T + self.b_pc
        pc = F.relu(pc)
        return pc


class DCNLayer(nn.Module):
    def __init__(self, n_mf: int, n_pc: int, n_dcn: int):
        super().__init__()
        self.n_mf, self.n_pc, self.n_dcn = n_mf, n_pc, n_dcn
        self.W_mf_dcn_raw = nn.Parameter(torch.empty(n_dcn, n_mf))
        nn.init.kaiming_uniform_(self.W_mf_dcn_raw, a=math.sqrt(5))
        self.W_pc_dcn_raw = nn.Parameter(torch.empty(n_dcn, n_pc))
        nn.init.kaiming_uniform_(self.W_pc_dcn_raw, a=math.sqrt(5))
        self.b_dcn = nn.Parameter(torch.zeros(n_dcn))

    def preact(self, m_in: torch.Tensor, pc: torch.Tensor) -> torch.Tensor:
        Wm = F.softplus(self.W_mf_dcn_raw)   # excitatory MF->DCN
        Wp = F.softplus(self.W_pc_dcn_raw)   # inhibitory PC->DCN
        exc = m_in @ Wm.T
        inh = pc @ Wp.T
        return exc - inh + self.b_dcn

    def forward(self, m_in: torch.Tensor, pc: torch.Tensor) -> torch.Tensor:
        return F.relu(self.preact(m_in, pc))


class Cerebellum(nn.Module):
    def __init__(self, hp, MossyGranuleLayerClass=MossyGranuleLayer):
        super().__init__()
        n_mf  = hp['n_input']
        n_gc  = hp['n_gc']
        nsyn  = hp.get('n_syn', 4)
        n_pc  = hp['n_pc']
        n_dcn = hp['n_dcn']

        self.dt      = float(hp.get('dt', 1))
        self.tau_gc  = float(hp.get('tau_gc', 10))
        self.tau_pc  = float(hp.get('tau_pc', 20))
        self.tau_dcn = float(hp.get('tau_dcn', 30))

        pf_fanin = int(hp.get('pf_fanin', 256))

        self.gc = MossyGranuleLayerClass(
            n_mf, n_gc, nsyn=nsyn,
            threshold_mode=hp.get('gc_threshold_mode', 'k_of_n'),
            k_active=hp.get('gc_k_active', None),
            theta_abs=hp.get('gc_theta_abs', 0.0),
            golgi_beta=hp.get('gc_golgi_beta', 0.0),
            seed=hp.get('seed', 42),
            weight_scale=hp.get('gc_weight_scale', 1.0),
            dtype=torch.float32,
        )
        self.pc  = PurkinjeLayer(n_gc=self.gc.n_gc, n_pc=n_pc, pf_fanin=pf_fanin, seed=hp.get('seed', 42))
        self.dcn = DCNLayer(n_mf=n_mf, n_pc=n_pc, n_dcn=n_dcn)

    def _leaky(self, y_prev: torch.Tensor, drive: torch.Tensor, tau: float) -> torch.Tensor:
        return y_prev + (self.dt / tau) * (drive - y_prev)

    def forward(self, x: torch.Tensor, batch_first: bool = False) -> tuple:
        """
        Args:
            x:           Input tensor. Shape (T, B, M) by default,
                         or (B, T, M) if batch_first=True.
                         A 2-D tensor (B, M) is treated as a single time step.
            batch_first: If True, input/output are (B, T, ...) instead of (T, B, ...).

        Returns:
            dcn_seq, gc_seq, pc_seq  — same leading-dim convention as input.

        Notes on temporal ordering:
            There is a deliberate one-step lag through the hierarchy:
              - GC at time t is driven by the *current* input x[t].
              - PC at time t is driven by the *previous* GC state (gc_state[t-1]).
              - DCN at time t is driven by the *previous* PC state (pc_state[t-1]).
            This avoids instantaneous signal propagation across layers.
            At t=0, PC and DCN receive zero input (states initialised to zero).
        """
        n_mf = self.gc.n_mf
        if x.size(-1) != n_mf:
            raise ValueError(f"Cerebellum expects last dim == n_input={n_mf}, got {tuple(x.shape)}")

        squeeze_out = False
        if x.dim() == 2:
            # (B, M) -> (1, B, M)
            x = x.unsqueeze(0)
            squeeze_out = True
        elif x.dim() == 3:
            if batch_first:
                x = x.permute(1, 0, 2).contiguous()  # (B,T,M) -> (T,B,M)
        else:
            raise ValueError("x must have shape (B,M), (T,B,M), or (B,T,M)")

        T, B, M = x.shape
        if T == 0:
            raise ValueError(f"Empty time dimension (T=0) for input {tuple(x.shape)}.")

        G = self.gc.n_gc
        P = self.pc.n_pc
        O = self.dcn.n_dcn

        gc_state  = torch.zeros(B, G, device=x.device, dtype=x.dtype)
        pc_state  = torch.zeros(B, P, device=x.device, dtype=x.dtype)
        dcn_state = torch.zeros(B, O, device=x.device, dtype=x.dtype)

        gc_seq, pc_seq, dcn_seq = [], [], []

        for t in range(T):
            xt = x[t]

            # --- Instantaneous drives (see lag note in docstring) ---
            # GC: driven by current input
            g_inst = self.gc(xt)
            # PC: driven by previous GC state
            pc_inst = self.pc(gc_state)
            # DCN: pre-activation uses previous PC state; rectify *before* filtering
            # so the membrane variable stays non-negative.
            dcn_inst = self.dcn(m_in=xt, pc=pc_state)   # FIX: relu on drive, not on filtered state

            # --- Leaky integration ---
            gc_state  = self._leaky(gc_state,  g_inst,   self.tau_gc)
            pc_state  = self._leaky(pc_state,  pc_inst,  self.tau_pc)
            dcn_state = self._leaky(dcn_state, dcn_inst, self.tau_dcn)
            # dcn_state is now always >= 0 because dcn_inst >= 0 and the
            # convex combination of non-negatives stays non-negative.

            gc_seq.append(gc_state)
            pc_seq.append(pc_state)
            dcn_seq.append(dcn_state)

        gc_seq  = torch.stack(gc_seq,  dim=0)  # (T, B, G)
        pc_seq  = torch.stack(pc_seq,  dim=0)  # (T, B, P)
        dcn_seq = torch.stack(dcn_seq, dim=0)  # (T, B, O)

        if squeeze_out:
            return dcn_seq[0], gc_seq[0], pc_seq[0]

        if batch_first:
            return dcn_seq.permute(1,0,2), gc_seq.permute(1,0,2), pc_seq.permute(1,0,2)

        return dcn_seq, gc_seq, pc_seq


class Run_Model(nn.Module):
    def __init__(self, hp, params, MossyGranuleLayerClass=MossyGranuleLayer):
        super().__init__()
        self.hp     = hp
        self.params = params
        # FIX: pass through MossyGranuleLayerClass instead of hardcoding the default
        self.model  = Cerebellum(hp, MossyGranuleLayerClass=MossyGranuleLayerClass)

        if hp['loss_type'] == 'lsq':
            self.loss_fnc = nn.MSELoss(reduction='none')
        else:
            self.loss_fnc = nn.CrossEntropyLoss(reduction='none')
        self.n_out   = self.hp.get('n_output', self.hp['n_dcn'])
        self.readout = nn.Linear(self.hp['n_dcn'], self.n_out, bias=True)

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

    def calculate_loss(self, pred, target):
        mask         = ~torch.isnan(target)
        target_clean = torch.where(mask, target, torch.zeros_like(target))

        if isinstance(self.loss_fnc, nn.CrossEntropyLoss):
            T, B, C     = pred.shape
            pred_flat   = pred.view(T * B, C)
            target_flat = target_clean.view(T * B).long()
            mask_flat   = mask.view(T * B)
            losses_flat = self.loss_fnc(pred_flat, target_flat) * mask_flat.float()
            data_loss   = losses_flat.sum()
        else:
            raw       = self.loss_fnc(pred, target_clean) * mask.float()
            data_loss = raw.sum()

        return data_loss

    def forward(self, batch_size=None, X=None, Y=None):
        if X is None or Y is None:
            X, Y = self.generate_trials(batch_size)
        dcn, gc, pc = self.model(X)
        T, B, O = dcn.shape
        pred      = self.readout(dcn.view(T * B, O)).view(T, B, self.n_out)
        loss_data = self.calculate_loss(pred, Y)
        return (
            loss_data,
            pred.detach().cpu().numpy(),
            dcn.detach().cpu().numpy(),
            gc.detach().cpu().numpy(),
            pc.detach().cpu().numpy(),
        )