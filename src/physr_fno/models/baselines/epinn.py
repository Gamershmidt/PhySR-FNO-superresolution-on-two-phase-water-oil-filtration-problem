from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveTanh(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.alpha * x + self.beta)


class GatedResBlock(nn.Module):
    def __init__(self, hidden_dim: int, anchor_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act1 = AdaptiveTanh(hidden_dim)
        self.act2 = AdaptiveTanh(hidden_dim)
        self.gate = nn.Linear(hidden_dim + anchor_dim, hidden_dim)

    def forward(self, h: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        u = self.fc2(self.act2(self.act1(self.fc1(h))))
        g = torch.sigmoid(self.gate(torch.cat([h, anchor], dim=-1)))
        return h + g * u


class EPINN(nn.Module):
    def __init__(self, in_dim: int = 9, hidden: int = 128, depth: int = 6, out_dim: int = 1):
        super().__init__()
        if out_dim != 1:
            raise ValueError("EPINN baseline predicts scalar pressure correction")
        anchor_dim = 3 * in_dim
        self.in_fc = nn.Linear(anchor_dim, hidden)
        self.in_act = AdaptiveTanh(hidden)
        self.blocks = nn.ModuleList([GatedResBlock(hidden, anchor_dim) for _ in range(depth)])
        self.out_fc = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out_fc.weight)
        nn.init.zeros_(self.out_fc.bias)

    def forward(self, X, A, bc_mask, bc_value, p_anchor, corr_scale: float = 1.0):
        Xn = A @ X
        anchor = torch.cat([X, Xn, X - Xn], dim=-1)
        h = self.in_act(self.in_fc(anchor))
        for block in self.blocks:
            h = block(h, anchor)
        dp = float(corr_scale) * torch.tanh(self.out_fc(h).squeeze(-1))
        p_anchor = p_anchor.reshape(-1)
        bc_mask = bc_mask.reshape(-1)
        bc_value = bc_value.reshape(-1)
        p = p_anchor + dp
        p = p * (1.0 - bc_mask) + bc_value * bc_mask
        return p.reshape(-1, 1)
