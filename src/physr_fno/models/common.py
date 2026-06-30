from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelNorm(nn.Module):
    def __init__(self, mean, std, eps: float = 1e-6):
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)
        if mean.ndim == 1:
            mean = mean.view(1, 1, -1, 1, 1)
        if std.ndim == 1:
            std = std.view(1, 1, -1, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x * (self.std + self.eps) + self.mean


class SpatialNorm(nn.Module):
    def __init__(self, mean, std, eps: float = 1e-6):
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.as_tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)


def temporal_interp(x, t_out: int):
    b, t, c, h, w = x.shape
    z = x.permute(0, 2, 3, 4, 1).reshape(b * c * h * w, 1, t)
    z = F.interpolate(z, size=t_out, mode="linear", align_corners=True)
    return z.reshape(b, c, h, w, t_out).permute(0, 4, 1, 2, 3)


def spatial_interp(x, size_hw: tuple[int, int]):
    b, t, c, h, w = x.shape
    z = x.reshape(b * t, c, h, w)
    z = F.interpolate(z, size=size_hw, mode="bicubic", align_corners=True)
    return z.reshape(b, t, c, size_hw[0], size_hw[1])


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.c1 = nn.Conv2d(c, 2 * c, 3, padding=1)
        self.c2 = nn.Conv2d(2 * c, c, 3, padding=1)

    def forward(self, x):
        return x + 0.1 * self.c2(F.gelu(self.c1(x)))


class SpectralConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, modes: int = 12):
        super().__init__()
        self.modes = int(modes)
        scale = 1.0 / max(1, in_ch * out_ch)
        self.weight = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, self.modes, self.modes, dtype=torch.cfloat)
        )
        self.pw = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        b, _, h, w = x.shape
        xf = torch.fft.rfft2(x, norm="ortho")
        m1 = min(self.modes, h)
        m2 = min(self.modes, w // 2 + 1)
        out = torch.zeros(
            b,
            self.weight.shape[1],
            h,
            w // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out[:, :, :m1, :m2] = torch.einsum(
            "bimn,iomn->bomn",
            xf[:, :, :m1, :m2],
            self.weight[:, :, :m1, :m2],
        )
        return torch.fft.irfft2(out, s=(h, w), norm="ortho") + self.pw(x)


class FNOBlock(nn.Module):
    def __init__(self, c: int, modes: int = 12):
        super().__init__()
        groups = min(8, c)
        while c % groups != 0 and groups > 1:
            groups -= 1
        self.fconv = SpectralConv2d(c, c, modes)
        self.norm = nn.GroupNorm(groups, c)

    def forward(self, x):
        return F.gelu(self.norm(self.fconv(x))) + x
