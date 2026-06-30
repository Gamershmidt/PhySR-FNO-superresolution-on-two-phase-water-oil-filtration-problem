from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from physr_fno.models.common import ChannelNorm, SpatialNorm, spatial_interp, temporal_interp


def _valid_group_count(c):
    g = min(8, c)
    while c % g != 0 and g > 1:
        g -= 1
    return g


class Time2Vec(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear_w = nn.Parameter(torch.randn(1))
        self.linear_b = nn.Parameter(torch.zeros(1))
        self.periodic_w = nn.Parameter(torch.randn(max(1, dim - 1)))
        self.periodic_b = nn.Parameter(torch.zeros(max(1, dim - 1)))
        self.dim = dim

    def forward(self, t):
        v0 = self.linear_w * t + self.linear_b
        vp = torch.sin(t * self.periodic_w + self.periodic_b)
        out = torch.cat([v0, vp], dim=-1)
        return out[..., : self.dim]


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1_max, modes2_max):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1_max = int(modes1_max)
        self.modes2_max = int(modes2_max)
        self.active_m1 = self.modes1_max
        self.active_m2 = self.modes2_max
        scale = 1.0 / max(1, in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.modes1_max, self.modes2_max, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.modes1_max, self.modes2_max, dtype=torch.cfloat))

    def set_active_modes(self, modes1, modes2):
        self.active_m1 = min(int(modes1), self.modes1_max)
        self.active_m2 = min(int(modes2), self.modes2_max)

    def forward(self, x):
        b, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(b, self.out_channels, h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        m1 = min(self.active_m1, h)
        m2 = min(self.active_m2, w // 2 + 1)
        out_ft[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")


class AFNOBlock2d(nn.Module):
    def __init__(self, width, modes1_max, modes2_max):
        super().__init__()
        g = _valid_group_count(width)
        self.norm_spec = nn.GroupNorm(g, width)
        self.norm_dw = nn.GroupNorm(g, width)
        self.spectral = SpectralConv2d(width, width, modes1_max, modes2_max)
        self.depthwise = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.pointwise = nn.Conv2d(width, width, 1)
        self.mix = nn.Conv2d(width, width, 1)

    def set_active_modes(self, modes1, modes2):
        self.spectral.set_active_modes(modes1, modes2)

    def forward(self, x):
        xs = self.spectral(self.norm_spec(x))
        xd = self.pointwise(self.depthwise(self.norm_dw(x)))
        return x + self.mix(F.gelu(xs + xd))


class SaturationDecoder(nn.Module):
    def __init__(self, width, modes1_max, modes2_max):
        super().__init__()
        self.spec = SpectralConv2d(width, width, modes1_max, modes2_max)
        self.mlp = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, max(1, width // 2), 1), nn.GELU(), nn.Conv2d(max(1, width // 2), 1, 1))

    def set_active_modes(self, modes1, modes2):
        self.spec.set_active_modes(modes1, modes2)

    def forward(self, x):
        return self.mlp(F.gelu(self.spec(x)))


class PressureDecoder(nn.Module):
    def __init__(self, width, dropout=0.2):
        super().__init__()
        half = max(1, width // 2)
        self.net = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU(), nn.Dropout2d(dropout), nn.Conv2d(width, half, 3, padding=1), nn.GELU(), nn.Conv2d(half, 1, 1))

    def forward(self, x):
        return self.net(x)


class DBAFNOModel(nn.Module):
    def __init__(self, state_mean, state_std, static_mean, static_std, cfg, target_mean=None, target_std=None):
        super().__init__()
        target_mean = state_mean if target_mean is None else target_mean
        target_std = state_std if target_std is None else target_std
        self.cfg = cfg
        self.state_norm = ChannelNorm(state_mean, state_std)
        self.target_norm = ChannelNorm(target_mean, target_std)
        self.static_norm = SpatialNorm(static_mean, static_std)
        self.time2vec = Time2Vec(cfg.time_emb_dim)
        self.dt_proj = nn.Linear(1, cfg.time_emb_dim)
        self.shared_time_proj = nn.Linear(cfg.time_emb_dim, cfg.fno_width)
        self.sat_time_proj = nn.Linear(cfg.time_emb_dim, cfg.fno_width)
        self.pres_time_proj = nn.Linear(cfg.time_emb_dim, cfg.fno_width)
        self.input_proj = nn.Conv2d(2 + 5 + 2, cfg.fno_width, 1)
        self.blocks = nn.ModuleList([AFNOBlock2d(cfg.fno_width, cfg.fno_modes_x_max, cfg.fno_modes_y_max) for _ in range(cfg.fno_layers)])
        self.sat_decoder = SaturationDecoder(cfg.fno_width, cfg.fno_modes_x_max, cfg.fno_modes_y_max)
        self.pres_decoder = PressureDecoder(cfg.fno_width, dropout=cfg.pressure_dropout)
        self.gamma_p = nn.Parameter(torch.tensor(cfg.gamma_p_init, dtype=torch.float32))
        self.alpha_cross = nn.Parameter(torch.tensor(cfg.alpha_cross_init, dtype=torch.float32))
        self.beta_cross = nn.Parameter(torch.tensor(cfg.beta_cross_init, dtype=torch.float32))
        self.s_gate = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.GELU(), nn.Conv2d(8, 1, 1))
        self.p_gate = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.GELU(), nn.Conv2d(8, 1, 1))
        self.register_buffer("grid", self._make_grid(cfg.ny_hr, cfg.nx_hr))
        self.set_active_modes(*cfg.curriculum_modes[0])

    def set_active_modes(self, modes1, modes2):
        for block in self.blocks:
            block.set_active_modes(modes1, modes2)
        self.sat_decoder.set_active_modes(modes1, modes2)

    def _make_grid(self, ny, nx):
        y = torch.linspace(-1.0, 1.0, ny)
        x = torch.linspace(-1.0, 1.0, nx)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([xx, yy], dim=0)

    def _time_context(self, b, t, device, dtype):
        tau = torch.linspace(0.0, 1.0, t, device=device, dtype=dtype).view(1, t, 1).expand(b, -1, -1)
        dt = torch.full_like(tau, 1.0 / max(1, t - 1))
        return self.time2vec(tau) + self.dt_proj(dt)

    def _static_features(self, K, H, m, inj_mask, prod_mask):
        feats = torch.stack([torch.log1p(K), H, m, inj_mask.float(), prod_mask.float()], dim=1)
        return self.static_norm.encode(feats)

    def forward(self, lr, K, H, m, inj_mask, prod_mask):
        base = spatial_interp(temporal_interp(lr, self.cfg.nt_hr), (self.cfg.ny_hr, self.cfg.nx_hr))
        state_z = self.state_norm.encode(base)
        b, t, _, h, w = state_z.shape
        static_z = self._static_features(K, H, m, inj_mask, prod_mask)[:, None].expand(-1, t, -1, -1, -1)
        grid = self.grid.to(device=lr.device, dtype=lr.dtype)[None, None].expand(b, t, -1, -1, -1)
        x = torch.cat([state_z, static_z, grid], dim=2).reshape(b * t, 9, h, w)
        x = self.input_proj(x)
        ctx = self._time_context(b, t, lr.device, lr.dtype)
        x = x + self.shared_time_proj(ctx).reshape(b * t, self.cfg.fno_width, 1, 1)
        for block in self.blocks:
            x = block(x)
        sat_feat = x + self.sat_time_proj(ctx).reshape(b * t, self.cfg.fno_width, 1, 1)
        pres_feat = x + self.pres_time_proj(ctx).reshape(b * t, self.cfg.fno_width, 1, 1)
        sat_raw = self.sat_decoder(sat_feat).reshape(b, t, 1, h, w)
        pres_raw = self.pres_decoder(pres_feat).reshape(b, t, 1, h, w)
        pres_amp = self.gamma_p * pres_raw
        s_gate = torch.sigmoid(self.s_gate(pres_amp.reshape(b * t, 1, h, w))).reshape(b, t, 1, h, w)
        p_gate = torch.sigmoid(self.p_gate(sat_raw.reshape(b * t, 1, h, w))).reshape(b, t, 1, h, w)
        pred_z = torch.cat([pres_amp + self.beta_cross * p_gate * sat_raw, sat_raw + self.alpha_cross * s_gate * pres_amp], dim=2)
        pred = self.target_norm.decode(pred_z)
        pred = torch.cat([pred[:, :, 0:1], torch.clamp(pred[:, :, 1:2], self.cfg.Swc + 1e-4, self.cfg.Smax)], dim=2)
        return pred, base
