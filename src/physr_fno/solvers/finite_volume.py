from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch.utils.data import Dataset

from physr_fno.config import Config, finalize_config


def smooth_noise(ny, nx, seed, sweeps=22):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(ny, nx))
    for _ in range(sweeps):
        z = (4.0 * z + np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1)) / 8.0
    return (z - z.mean()) / (z.std() + 1e-8)


def circle_mask(ny, nx, cx, cy, radius):
    yy, xx = np.mgrid[0:ny, 0:nx]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


def relperm_np(S, cfg: Config):
    Se = np.clip((S - cfg.Swc) / (cfg.Smax - cfg.Swc), 0.0, 1.0)
    Kw = np.where(S <= cfg.Swc, 0.0, Se ** cfg.n_w)
    Kn = np.where(S >= cfg.Smax, 0.0, (1.0 - Se) ** cfg.n_n)
    return Kw.astype(np.float32), Kn.astype(np.float32)


def fw_from_s_np(S, cfg: Config):
    Kw, Kn = relperm_np(S, cfg)
    lam_w = Kw / cfg.mu_w
    lam_n = Kn / cfg.mu_n
    return (lam_w / (lam_w + lam_n + 1e-10)).astype(np.float32)


def _rotated_gaussian(X, Y, cx, cy, sx, sy, angle):
    ca, sa = np.cos(angle), np.sin(angle)
    xr = ca * (X - cx) + sa * (Y - cy)
    yr = -sa * (X - cx) + ca * (Y - cy)
    return np.exp(-(xr**2 / (2.0 * sx**2 + 1e-12) + yr**2 / (2.0 * sy**2 + 1e-12)))


def _polyline_field(X, Y, pts, width):
    d2 = np.full_like(X, 1e9, dtype=np.float64)
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        vx, vy = x1 - x0, y1 - y0
        vv = vx * vx + vy * vy + 1e-12
        t = np.clip(((X - x0) * vx + (Y - y0) * vy) / vv, 0.0, 1.0)
        px, py = x0 + t * vx, y0 + t * vy
        d2 = np.minimum(d2, (X - px) ** 2 + (Y - py) ** 2)
    return np.exp(-d2 / (2.0 * width**2 + 1e-12))


def _meandering_path(x0, y0, x1, y1, rng, n_pts=9, amp=0.12):
    ts = np.linspace(0.0, 1.0, n_pts)
    xs = (1.0 - ts) * x0 + ts * x1
    ys = (1.0 - ts) * y0 + ts * y1
    dx, dy = x1 - x0, y1 - y0
    norm = np.sqrt(dx * dx + dy * dy) + 1e-12
    nx, ny = -dy / norm, dx / norm
    wav = 0.65 * np.sin(2.0 * np.pi * ts * rng.uniform(1.0, 2.0) + rng.uniform(0.0, 2.0 * np.pi))
    wav += 0.35 * np.sin(2.0 * np.pi * ts * rng.uniform(2.0, 4.0) + rng.uniform(0.0, 2.0 * np.pi))
    xs = np.clip(xs + amp * wav * nx, 0.04, 0.96)
    ys = np.clip(ys + amp * wav * ny, 0.04, 0.96)
    return list(zip(xs, ys))


def make_coefficients(seed: int, cfg: Config):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, cfg.nx_hr)
    y = np.linspace(0.0, 1.0, cfg.ny_hr)
    X, Y = np.meshgrid(x, y)

    Kf = 0.60 * smooth_noise(cfg.ny_hr, cfg.nx_hr, 100 + seed, sweeps=10)
    Hf = 0.70 * smooth_noise(cfg.ny_hr, cfg.nx_hr, 200 + seed, sweeps=8)
    mf = 0.70 * smooth_noise(cfg.ny_hr, cfg.nx_hr, 300 + seed, sweeps=8)

    inj_cx = int(rng.integers(max(1, int(0.08 * cfg.nx_hr)), max(2, int(0.24 * cfg.nx_hr))))
    inj_cy = int(rng.integers(max(1, int(0.20 * cfg.ny_hr)), max(2, int(0.80 * cfg.ny_hr))))
    prod_cx = int(rng.integers(max(2, int(0.74 * cfg.nx_hr)), max(3, int(0.92 * cfg.nx_hr))))
    prod_cy = int(rng.integers(max(1, int(0.20 * cfg.ny_hr)), max(2, int(0.80 * cfg.ny_hr))))

    inj_mask = circle_mask(cfg.ny_hr, cfg.nx_hr, inj_cx, inj_cy, cfg.well_radius)
    prod_mask = circle_mask(cfg.ny_hr, cfg.nx_hr, prod_cx, prod_cy, cfg.well_radius)

    path = _meandering_path(x[inj_cx], y[inj_cy], x[prod_cx], y[prod_cy], rng, n_pts=7, amp=0.12)
    channel = _polyline_field(X, Y, path, width=max(0.035, 1.0 / cfg.nx_hr))
    Kf += 1.4 * channel

    for _ in range(2):
        barrier = _rotated_gaussian(
            X,
            Y,
            cx=rng.uniform(0.2, 0.8),
            cy=rng.uniform(0.1, 0.9),
            sx=rng.uniform(0.06, 0.14),
            sy=rng.uniform(0.01, 0.04),
            angle=rng.uniform(0.0, np.pi),
        )
        Kf -= rng.uniform(0.6, 1.2) * barrier

    K = np.exp(0.95 * Kf)
    K /= K.mean()
    K = np.clip(K, 0.03, 9.50)
    H = np.clip(cfg.H0 * (1.0 + 0.16 * Hf), 0.62, 1.38)
    m = np.clip(cfg.m0 * (1.0 + 0.12 * mf), 0.15, 0.33)

    S0 = np.full((cfg.ny_hr, cfg.nx_hr), cfg.Swc + rng.uniform(0.006, 0.018), dtype=np.float32)
    plume = np.exp(-((X - x[inj_cx]) ** 2 + (Y - y[inj_cy]) ** 2) / rng.uniform(0.010, 0.028))
    S0 += rng.uniform(0.06, 0.16) * plume
    S0 += rng.uniform(0.03, 0.10) * channel * np.exp(-1.8 * ((X - x[inj_cx]) ** 2 + (Y - y[inj_cy]) ** 2))
    S0 = np.clip(S0, cfg.Swc + 1e-4, cfg.Smax - 0.10)
    S0[inj_mask] = cfg.Smax

    return K.astype(np.float32), H.astype(np.float32), m.astype(np.float32), inj_mask, prod_mask, S0.astype(np.float32)


def solve_pressure(S, K, H, inj_mask, prod_mask, cfg: Config):
    ny, nx = S.shape
    Kw, Kn = relperm_np(np.clip(S, cfg.Swc, cfg.Smax), cfg)
    sigma = H * K * (Kw / cfg.mu_w + Kn / cfg.mu_n) + 1e-10

    fixed = np.zeros((ny, nx), dtype=bool)
    fixed[:, 0] = fixed[:, -1] = True
    fixed[0, :] = fixed[-1, :] = True
    fixed[inj_mask] = True
    fixed[prod_mask] = True

    P = np.full((ny, nx), cfg.Pgamma, dtype=np.float64)
    P[inj_mask] = cfg.Pinj
    P[prod_mask] = cfg.Pprod

    unknowns = [(y, x) for y in range(ny) for x in range(nx) if not fixed[y, x]]
    if not unknowns:
        return P.astype(np.float32), sigma.astype(np.float32), fixed

    index = {yx: i for i, yx in enumerate(unknowns)}
    rows, cols, data = [], [], []
    b = np.zeros(len(unknowns), dtype=np.float64)

    for row, (y, x) in enumerate(unknowns):
        ae = 0.5 * (sigma[y, x] + sigma[y, x + 1])
        aw = 0.5 * (sigma[y, x] + sigma[y, x - 1])
        an = 0.5 * (sigma[y, x] + sigma[y + 1, x])
        ass = 0.5 * (sigma[y, x] + sigma[y - 1, x])
        rows.append(row); cols.append(row); data.append(ae + aw + an + ass)
        for (yy, xx), coef in [((y, x + 1), ae), ((y, x - 1), aw), ((y + 1, x), an), ((y - 1, x), ass)]:
            if fixed[yy, xx]:
                b[row] += coef * P[yy, xx]
            else:
                rows.append(row); cols.append(index[(yy, xx)]); data.append(-coef)

    A = sp.csr_matrix((data, (rows, cols)), shape=(len(unknowns), len(unknowns)))
    sol = spla.spsolve(A, b)
    for (y, x), val in zip(unknowns, sol):
        P[y, x] = val
    return P.astype(np.float32), sigma.astype(np.float32), fixed


def total_flux_faces_np(P, sigma, cfg: Config):
    ny, nx = P.shape
    qx = np.zeros((ny, nx + 1), dtype=np.float32)
    qy = np.zeros((ny + 1, nx), dtype=np.float32)
    sigma_x = 0.5 * (sigma[:, :-1] + sigma[:, 1:])
    qx[:, 1:nx] = -sigma_x * (P[:, 1:] - P[:, :-1]) / cfg.dx
    qx[:, 0] = -sigma[:, 0] * (P[:, 0] - cfg.Pgamma) / (0.5 * cfg.dx)
    qx[:, nx] = -sigma[:, -1] * (cfg.Pgamma - P[:, -1]) / (0.5 * cfg.dx)
    sigma_y = 0.5 * (sigma[:-1, :] + sigma[1:, :])
    qy[1:ny, :] = -sigma_y * (P[1:, :] - P[:-1, :]) / cfg.dy
    qy[0, :] = -sigma[0, :] * (P[0, :] - cfg.Pgamma) / (0.5 * cfg.dy)
    qy[ny, :] = -sigma[-1, :] * (cfg.Pgamma - P[-1, :]) / (0.5 * cfg.dy)
    return qx, qy


def water_flux_faces_np(S, qx, qy, cfg: Config):
    ny, nx = S.shape
    fw = fw_from_s_np(np.clip(S, cfg.Swc, cfg.Smax), cfg)
    fw_plus = float(fw_from_s_np(np.array([[cfg.S_plus]], dtype=np.float32), cfg)[0, 0])
    Fwx = np.zeros((ny, nx + 1), dtype=np.float32)
    Fwy = np.zeros((ny + 1, nx), dtype=np.float32)
    qx_int = qx[:, 1:nx]
    qy_int = qy[1:ny, :]
    Fwx[:, 1:nx] = np.where(qx_int >= 0.0, fw[:, :-1], fw[:, 1:]) * qx_int
    Fwy[1:ny, :] = np.where(qy_int >= 0.0, fw[:-1, :], fw[1:, :]) * qy_int
    Fwx[:, 0] = np.where(qx[:, 0] > 0.0, fw_plus, fw[:, 0]) * qx[:, 0]
    Fwx[:, nx] = np.where(qx[:, nx] < 0.0, fw_plus, fw[:, -1]) * qx[:, nx]
    Fwy[0, :] = np.where(qy[0, :] > 0.0, fw_plus, fw[0, :]) * qy[0, :]
    Fwy[ny, :] = np.where(qy[ny, :] < 0.0, fw_plus, fw[-1, :]) * qy[ny, :]
    return Fwx, Fwy


def sat_rhs_np(S, P, K, H, m, inj_mask, prod_mask, cfg: Config):
    S_work = np.clip(S, cfg.Swc, cfg.Smax).astype(np.float32)
    S_work[inj_mask] = cfg.Smax
    Kw, Kn = relperm_np(S_work, cfg)
    sigma = H * K * (Kw / cfg.mu_w + Kn / cfg.mu_n) + 1e-10
    qx, qy = total_flux_faces_np(P, sigma, cfg)
    Fwx, Fwy = water_flux_faces_np(S_work, qx, qy, cfg)
    div_w = (Fwx[:, 1:] - Fwx[:, :-1]) / cfg.dx + (Fwy[1:, :] - Fwy[:-1, :]) / cfg.dy
    rhs = -div_w / (m * H + 1e-10)
    rhs[0, :] = rhs[-1, :] = 0.0
    rhs[:, 0] = rhs[:, -1] = 0.0
    rhs[inj_mask] = 0.0
    rhs[prod_mask] = 0.0
    return rhs.astype(np.float32), qx, qy, Fwx, Fwy


def update_saturation(S, P, K, H, m, inj_mask, prod_mask, cfg: Config, return_dt=False):
    Kw, Kn = relperm_np(np.clip(S, cfg.Swc, cfg.Smax), cfg)
    sigma = H * K * (Kw / cfg.mu_w + Kn / cfg.mu_n) + 1e-10
    qx, qy = total_flux_faces_np(P, sigma, cfg)
    vel = max(np.max(np.abs(qx)), np.max(np.abs(qy))) / (np.min(m * H) + 1e-10)
    dt = min(cfg.dt, cfg.cfl * min(cfg.dx, cfg.dy) / (vel + 1e-10))
    rhs, *_ = sat_rhs_np(S, P, K, H, m, inj_mask, prod_mask, cfg)
    S_new = np.clip(S + dt * rhs, cfg.Swc, cfg.Smax).astype(np.float32)
    S_new[inj_mask] = cfg.Smax
    return (S_new, float(dt)) if return_dt else S_new


def compute_producer_rates(P, S, K, H, prod_mask, cfg: Config):
    Kw, Kn = relperm_np(np.clip(S, cfg.Swc, cfg.Smax), cfg)
    sigma = H * K * (Kw / cfg.mu_w + Kn / cfg.mu_n) + 1e-10
    qx, qy = total_flux_faces_np(P, sigma, cfg)
    Fwx, Fwy = water_flux_faces_np(np.clip(S, cfg.Swc, cfg.Smax), qx, qy, cfg)
    q_total = 0.0
    q_water = 0.0
    ny, nx = P.shape
    for y, x in np.argwhere(prod_mask):
        if x > 0 and not prod_mask[y, x - 1]: q_total += qx[y, x]; q_water += Fwx[y, x]
        if x < nx - 1 and not prod_mask[y, x + 1]: q_total -= qx[y, x + 1]; q_water -= Fwx[y, x + 1]
        if y > 0 and not prod_mask[y - 1, x]: q_total += qy[y, x]; q_water += Fwy[y, x]
        if y < ny - 1 and not prod_mask[y + 1, x]: q_total -= qy[y + 1, x]; q_water -= Fwy[y + 1, x]
    return float(q_total), float(q_water)


def simulate_case(seed: int, cfg: Config):
    cfg = finalize_config(cfg)
    K, H, m, inj_mask, prod_mask, S = make_coefficients(seed, cfg)
    P_seq, S_seq, qc_seq, qw_seq, dt_frame_seq = [], [], [], [], []

    for t in range(cfg.nt_hr):
        P, _, _ = solve_pressure(S, K, H, inj_mask, prod_mask, cfg)
        qc, qw = compute_producer_rates(P, S, K, H, prod_mask, cfg)
        P_seq.append(P.copy()); S_seq.append(S.copy()); qc_seq.append(qc); qw_seq.append(qw)
        if t < cfg.nt_hr - 1:
            dt_sum = 0.0
            for _ in range(cfg.sat_substeps):
                S, dt_used = update_saturation(S, P, K, H, m, inj_mask, prod_mask, cfg, return_dt=True)
                P, _, _ = solve_pressure(S, K, H, inj_mask, prod_mask, cfg)
                dt_sum += dt_used
            dt_frame_seq.append(dt_sum)

    hr = np.stack([np.stack(P_seq, axis=0), np.stack(S_seq, axis=0)], axis=1).astype(np.float32)
    qc_arr = np.array(qc_seq, dtype=np.float32)
    qw_arr = np.array(qw_seq, dtype=np.float32)
    meta = {
        "K": K,
        "H": H,
        "m": m,
        "inj": inj_mask,
        "prod": prod_mask,
        "Qc": qc_arr,
        "Qw": qw_arr,
        "dt_frame": np.array(dt_frame_seq, dtype=np.float32),
        "watercut": qw_arr / (qc_arr + 1e-10),
    }
    return hr, meta


def _blur2d(z, sweeps=2):
    out = z.copy()
    for _ in range(sweeps):
        out = (4.0 * out + np.roll(out, 1, 0) + np.roll(out, -1, 0) + np.roll(out, 1, 1) + np.roll(out, -1, 1)) / 8.0
    return out


def make_lr_from_hr(hr, cfg: Config, seed: int = 0):
    cfg = finalize_config(cfg)
    rng = np.random.default_rng(5000 + seed)
    obs = hr.copy()
    for t in range(obs.shape[0]):
        obs[t, 0] = _blur2d(obs[t, 0], sweeps=1)
        obs[t, 1] = _blur2d(obs[t, 1], sweeps=1)
    lr = obs[np.arange(0, cfg.nt_hr, cfg.t_up)]
    nt, ch, ny, nx = lr.shape
    lr = lr.reshape(nt, ch, cfg.ny_lr, cfg.s_up, cfg.nx_lr, cfg.s_up).mean(axis=(3, 5))
    noise = rng.normal(size=lr.shape).astype(np.float32)
    lr[:, 0] += 0.0040 * noise[:, 0]
    lr[:, 1] += 0.0030 * noise[:, 1]
    lr[:, 1] = np.clip(lr[:, 1], cfg.Swc, cfg.Smax)
    return lr.astype(np.float32)


class FiltrationDataset(Dataset):
    def __init__(self, items):
        self.items = list(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        lr, hr, meta = self.items[idx]
        return {
            "lr": torch.tensor(lr, dtype=torch.float32),
            "hr": torch.tensor(hr, dtype=torch.float32),
            "K": torch.tensor(meta["K"], dtype=torch.float32),
            "H": torch.tensor(meta["H"], dtype=torch.float32),
            "m": torch.tensor(meta["m"], dtype=torch.float32),
            "inj": torch.tensor(meta["inj"], dtype=torch.bool),
            "prod": torch.tensor(meta["prod"], dtype=torch.bool),
        }


def generate_items(cfg: Config, n_cases: int | None = None, seed_offset: int = 0):
    cfg = finalize_config(cfg)
    items = []
    total = cfg.n_cases if n_cases is None else n_cases
    for seed in range(seed_offset, seed_offset + total):
        hr, meta = simulate_case(seed, cfg)
        lr = make_lr_from_hr(hr, cfg, seed)
        items.append((lr, hr, meta))
    return items


def compute_channel_stats(items):
    hr = np.concatenate([x[1] for x in items], axis=0)
    mean = torch.tensor(hr.mean(axis=(0, 2, 3)), dtype=torch.float32)
    std = torch.tensor(hr.std(axis=(0, 2, 3)) + 1e-6, dtype=torch.float32)
    return mean, std
