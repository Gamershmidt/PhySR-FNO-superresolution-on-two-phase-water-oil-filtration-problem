from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ChannelNorm, FNOBlock, ResBlock, spatial_interp, temporal_interp


class StaticFieldFusion(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, c, 3, padding=1),
            nn.GELU(),
            ResBlock(c),
            FNOBlock(c, modes=8),
            ResBlock(c),
        )
        self.fuse = nn.Sequential(nn.Conv2d(2 * c, c, 1), nn.GELU(), nn.Conv2d(c, c, 1))

    def encode(self, K, H_static, m):
        static = torch.stack([K, H_static, m], dim=1)
        return self.enc(static)

    def inject(self, x, g):
        return x + self.fuse(torch.cat([x, g], dim=1))


class PhySR_FNO(nn.Module):
    def __init__(self, mean, std, cfg, n_feats: int = 64, n_fno: int = 4, modes: int = 12):
        super().__init__()
        self.cfg = cfg
        self.norm = ChannelNorm(mean, std)
        self.head_P = nn.Sequential(
            nn.Conv2d(1, n_feats, 3, padding=1), nn.GELU(), ResBlock(n_feats), ResBlock(n_feats)
        )
        self.head_S = nn.Sequential(
            nn.Conv2d(1, n_feats, 3, padding=1), nn.GELU(), ResBlock(n_feats), ResBlock(n_feats)
        )
        self.merge = nn.Sequential(nn.Conv2d(2 * n_feats, n_feats, 1), nn.GELU())
        self.static_fusion = StaticFieldFusion(n_feats)
        self.fno_blocks = nn.Sequential(*[FNOBlock(n_feats, modes) for _ in range(n_fno)])
        self.refine = nn.Sequential(ResBlock(n_feats), ResBlock(n_feats), ResBlock(n_feats))
        self.tail_P = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1), nn.GELU(),
            nn.Conv2d(n_feats, max(1, n_feats // 2), 3, padding=1), nn.GELU(),
            nn.Conv2d(max(1, n_feats // 2), 1, 3, padding=1),
        )
        self.tail_S = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1), nn.GELU(),
            nn.Conv2d(n_feats, max(1, n_feats // 2), 3, padding=1), nn.GELU(),
            nn.Conv2d(max(1, n_feats // 2), 1, 3, padding=1),
        )

    def forward(self, lr, K=None, H_static=None, m=None):
        base = spatial_interp(temporal_interp(lr, self.cfg.nt_hr), (self.cfg.ny_hr, self.cfg.nx_hr))
        z = self.norm.encode(base)
        b, t, c, h, w = z.shape
        has_static = K is not None and H_static is not None and m is not None
        if has_static:
            g = self.static_fusion.encode(K, H_static, m)

        z_flat = z.reshape(b * t, c, h, w)
        feat_P = self.head_P(z_flat[:, 0:1])
        feat_S = self.head_S(z_flat[:, 1:2])
        feat = self.merge(torch.cat([feat_P, feat_S], dim=1))
        if has_static:
            g_rep = g.unsqueeze(1).expand(-1, t, -1, -1, -1).reshape(b * t, -1, h, w)
            feat = self.static_fusion.inject(feat, g_rep)
        feat = self.refine(self.fno_blocks(feat))
        delta_P = self.tail_P(feat).reshape(b, t, 1, h, w)
        delta_S = self.tail_S(feat).reshape(b, t, 1, h, w)
        pred_norm = torch.cat([z[:, :, 0:1] + delta_P, z[:, :, 1:2] + delta_S], dim=2)
        pred = self.norm.decode(pred_norm)
        P_out = pred[:, :, 0:1]
        S_out = torch.clamp(pred[:, :, 1:2], self.cfg.Swc + 1e-4, self.cfg.Smax)
        return torch.cat([P_out, S_out], dim=2), base


PhySRFNO = PhySR_FNO
