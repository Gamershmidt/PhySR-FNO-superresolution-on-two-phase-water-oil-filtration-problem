from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from physr_fno.models.common import ChannelNorm, ResBlock, spatial_interp, temporal_interp


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=kernel_size // 2)

    def init_state(self, b, h, w, device, dtype):
        zeros = torch.zeros(b, self.hidden_dim, h, w, device=device, dtype=dtype)
        return zeros, zeros

    def forward(self, x, state):
        h_prev, c_prev = state
        i, f, g, o = torch.chunk(self.gates(torch.cat([x, h_prev], dim=1)), 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c = f * c_prev + i * torch.tanh(g)
        return o * torch.tanh(c), c


class ResidualPhySRConvLSTM(nn.Module):
    def __init__(self, mean, std, cfg, n_feats: int = 48, n_blocks: int = 4, hidden_dim: int = 48):
        super().__init__()
        self.cfg = cfg
        self.norm = ChannelNorm(mean, std)
        self.head = nn.Conv2d(2, n_feats, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(n_feats) for _ in range(n_blocks)])
        self.convlstm = ConvLSTMCell(n_feats, hidden_dim)
        self.tail = nn.Sequential(nn.Conv2d(hidden_dim, n_feats, 3, padding=1), nn.GELU(), nn.Conv2d(n_feats, 2, 3, padding=1))

    def forward(self, lr, inj_mask=None, prod_mask=None):
        base = spatial_interp(temporal_interp(lr, self.cfg.nt_hr), (self.cfg.ny_hr, self.cfg.nx_hr))
        z = self.norm.encode(base)
        b, t, _, h, w = z.shape
        state = self.convlstm.init_state(b, h, w, z.device, z.dtype)
        outs = []
        for k in range(t):
            xk = self.body(self.head(z[:, k]))
            hk, ck = self.convlstm(xk, state)
            state = (hk, ck)
            outs.append(self.tail(hk))
        delta = torch.stack(outs, dim=1)
        P = base[:, :, 0:1] + self.cfg.alpha_p * torch.tanh(delta[:, :, 0:1])
        S = base[:, :, 1:2] + self.cfg.alpha_s * torch.tanh(delta[:, :, 1:2])
        S = torch.clamp(S, self.cfg.Swc + 1e-4, self.cfg.Smax)
        return torch.cat([P, S], dim=2), base
