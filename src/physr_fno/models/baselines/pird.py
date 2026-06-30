from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from physr_fno.models.common import ChannelNorm, spatial_interp, temporal_interp


def linear_beta_schedule(T, beta_start, beta_end, device):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_bar


def get_timestep_embedding_pird(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    scale = math.log(10000) / max(half_dim - 1, 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -scale)
    emb = timesteps.float()[:, None] * emb[None]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return F.pad(emb, (0, 1)) if embedding_dim % 2 == 1 else emb


def pird_nonlinearity(x):
    return x * torch.sigmoid(x)


def pird_group_count(c):
    for g in (8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1


def NormalizePird(in_channels):
    return nn.GroupNorm(num_groups=pird_group_count(in_channels), num_channels=in_channels, eps=1e-6, affine=True)


class UpsamplePird(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, padding_mode="circular") if with_conv else None

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x) if self.conv is not None else x


class DownsamplePird(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=0) if with_conv else None

    def forward(self, x):
        if self.conv is None:
            return F.avg_pool2d(x, kernel_size=2, stride=2)
        return self.conv(F.pad(x, (0, 1, 0, 1), mode="circular"))


class ResnetBlockPird(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0, temb_channels=512):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = NormalizePird(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode="circular")
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = NormalizePird(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode="circular")
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, temb=None):
        h = self.conv1(pird_nonlinearity(self.norm1(x)))
        if temb is not None:
            h = h + self.temb_proj(pird_nonlinearity(temb))[:, :, None, None]
        h = self.conv2(self.dropout(pird_nonlinearity(self.norm2(h))))
        return self.shortcut(x) + h


class AttnBlockPird(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = NormalizePird(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        b, c, h, w = q.shape
        attn = torch.bmm(q.reshape(b, c, h * w).permute(0, 2, 1), k.reshape(b, c, h * w)) * (c ** -0.5)
        attn = torch.softmax(attn, dim=2).permute(0, 2, 1)
        h_ = torch.bmm(v.reshape(b, c, h * w), attn).reshape(b, c, h, w)
        return x + self.proj_out(h_)


class PiRDAttentionUNet(nn.Module):
    def __init__(
        self,
        mean,
        std,
        cfg,
        ctx_frames: int | None = None,
        base_ch: int = 48,
        ch_mult=(1, 2, 2),
        num_res_blocks: int = 1,
        attn_resolutions=(16, 32),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cfg = cfg
        self.norm = ChannelNorm(mean, std)
        self.ctx_frames = int(ctx_frames or cfg.ctx_frames)
        self.ch = int(base_ch)
        self.temb_ch = self.ch * 4
        self.resolution = int(cfg.nx_hr)
        self.in_channels = self.ctx_frames * 2
        self.out_channels = self.ctx_frames * 2
        self.temb = nn.ModuleList([nn.Linear(self.ch, self.temb_ch), nn.Linear(self.temb_ch, self.temb_ch)])
        self.emb_conv = nn.Sequential(nn.Conv2d(self.in_channels, self.ch, 1), nn.GELU(), nn.Conv2d(self.ch, self.ch, 3, padding=1, padding_mode="circular"))
        self.conv_in = nn.Conv2d(self.in_channels, self.ch, 3, padding=1, padding_mode="circular")
        self.combine_conv = nn.Conv2d(self.ch * 2, self.ch, 1)

        curr_res = self.resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level, mult in enumerate(ch_mult):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = self.ch * in_ch_mult[i_level]
            block_out = self.ch * mult
            for _ in range(num_res_blocks):
                block.append(ResnetBlockPird(block_in, block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlockPird(block_in))
            down = nn.Module(); down.block = block; down.attn = attn
            if i_level != len(ch_mult) - 1:
                down.downsample = DownsamplePird(block_in)
                curr_res //= 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlockPird(block_in, block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlockPird(block_in)
        self.mid.block_2 = ResnetBlockPird(block_in, block_in, temb_channels=self.temb_ch, dropout=dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(len(ch_mult))):
            block = nn.ModuleList(); attn = nn.ModuleList()
            block_out = self.ch * ch_mult[i_level]
            skip_in = self.ch * ch_mult[i_level]
            for i_block in range(num_res_blocks + 1):
                if i_block == num_res_blocks:
                    skip_in = self.ch * in_ch_mult[i_level]
                block.append(ResnetBlockPird(block_in + skip_in, block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlockPird(block_in))
            up = nn.Module(); up.block = block; up.attn = attn
            if i_level != 0:
                up.upsample = UpsamplePird(block_in)
                curr_res *= 2
            self.up.insert(0, up)
        self.norm_out = NormalizePird(block_in)
        self.conv_out = nn.Conv2d(block_in, self.out_channels, 3, padding=1, padding_mode="circular")

    def flatten_win(self, x):
        b, w, c, h, k = x.shape
        return x.reshape(b, w * c, h, k)

    def unflatten_win(self, x):
        b, c, h, w = x.shape
        return x.view(b, self.ctx_frames, 2, h, w)

    def forward(self, x_t, cond, t):
        cond_n = self.norm.encode(cond)
        x_t_n = x_t / (self.norm.std + 1e-6)
        x = self.conv_in(self.flatten_win(x_t_n))
        cond_emb = self.emb_conv(self.flatten_win(cond_n))
        x = self.combine_conv(torch.cat([x, cond_emb], dim=1))
        temb = get_timestep_embedding_pird(t, self.ch)
        temb = self.temb[1](pird_nonlinearity(self.temb[0](temb)))
        hs = [x]
        for level in self.down:
            for i, block in enumerate(level.block):
                h = block(hs[-1], temb)
                if len(level.attn) > 0:
                    h = level.attn[min(i, len(level.attn) - 1)](h)
                hs.append(h)
            if hasattr(level, "downsample"):
                hs.append(level.downsample(hs[-1]))
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(hs[-1], temb)), temb)
        for level in reversed(self.up):
            for i, block in enumerate(level.block):
                h = block(torch.cat([h, hs.pop()], dim=1), temb)
                if len(level.attn) > 0:
                    h = level.attn[min(i, len(level.attn) - 1)](h)
            if hasattr(level, "upsample"):
                h = level.upsample(h)
        return self.unflatten_win(self.conv_out(pird_nonlinearity(self.norm_out(h)))) * (self.norm.std + 1e-6)


class PiRDWrapper(nn.Module):
    def __init__(self, mean, std, cfg, base_ch: int = 16):
        super().__init__()
        self.cfg = cfg
        self.unet = PiRDAttentionUNet(mean, std, cfg, base_ch=base_ch, ch_mult=(1, 2), attn_resolutions=(cfg.nx_hr // 2,))

    def forward(self, lr, *_, **__):
        base = spatial_interp(temporal_interp(lr, self.cfg.nt_hr), (self.cfg.ny_hr, self.cfg.nx_hr))
        cond = base[:, -self.cfg.ctx_frames :]
        x_t = torch.zeros_like(cond)
        t = torch.zeros(cond.shape[0], dtype=torch.long, device=lr.device)
        residual = self.unet(x_t, cond, t)
        out_tail = cond + residual
        pred = base.clone()
        pred[:, -self.cfg.ctx_frames :] = out_tail
        pred = torch.cat([pred[:, :, 0:1], torch.clamp(pred[:, :, 1:2], self.cfg.Swc + 1e-4, self.cfg.Smax)], dim=2)
        return pred, base
