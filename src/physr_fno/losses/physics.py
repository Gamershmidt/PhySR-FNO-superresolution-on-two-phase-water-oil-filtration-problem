from __future__ import annotations

import torch
import torch.nn.functional as F


def relperm_torch(S, cfg):
    Se = torch.clamp((S - cfg.Swc) / (cfg.Smax - cfg.Swc), 0.0, 1.0)
    Kw = torch.where(S <= cfg.Swc, torch.zeros_like(S), Se ** cfg.n_w)
    Kn = torch.where(S >= cfg.Smax, torch.zeros_like(S), (1.0 - Se) ** cfg.n_n)
    return Kw, Kn


def fw_from_s_torch(S, cfg):
    Kw, Kn = relperm_torch(S, cfg)
    lam_w = Kw / cfg.mu_w
    lam_n = Kn / cfg.mu_n
    lam_t = lam_w + lam_n + 1e-12
    return lam_w / lam_t, lam_t


def grad_xy(z):
    return z[..., :, 1:] - z[..., :, :-1], z[..., 1:, :] - z[..., :-1, :]


def soft_front_mask(S, cfg):
    theta = cfg.Swc + cfg.front_theta * (cfg.Smax - cfg.Swc)
    return torch.sigmoid(cfg.front_k * (S - theta))


def physics_weight(epoch: int, cfg):
    if epoch < cfg.physics_start_epoch:
        return 0.0
    frac = (epoch - cfg.physics_start_epoch + 1) / max(1, cfg.physics_ramp_epochs)
    return cfg.max_phys_w * min(1.0, max(0.0, frac)) ** 2


def enforce_bc_batch(state, inj_mask, prod_mask, cfg):
    P = state[:, :, 0]
    S = state[:, :, 1]
    b, _, ny, nx = P.shape
    boundary2d = torch.zeros((b, ny, nx), dtype=torch.bool, device=state.device)
    boundary2d[:, 0, :] = True
    boundary2d[:, -1, :] = True
    boundary2d[:, :, 0] = True
    boundary2d[:, :, -1] = True
    boundary = boundary2d[:, None]
    inj = inj_mask[:, None]
    prod = prod_mask[:, None]
    P = torch.where(boundary, torch.full_like(P, cfg.Pgamma), P)
    P = torch.where(inj, torch.full_like(P, cfg.Pinj), P)
    P = torch.where(prod, torch.full_like(P, cfg.Pprod), P)
    S = torch.where(inj, torch.full_like(S, cfg.Smax), S)
    S = torch.clamp(S, cfg.Swc + 1e-4, cfg.Smax)
    return torch.stack([P, S], dim=2)


def total_flux_faces_torch(P, sigma, cfg):
    b, t, ny, nx = P.shape
    qx = torch.zeros(b, t, ny, nx + 1, device=P.device, dtype=P.dtype)
    qy = torch.zeros(b, t, ny + 1, nx, device=P.device, dtype=P.dtype)
    sigma_x = 0.5 * (sigma[:, :, :, :-1] + sigma[:, :, :, 1:])
    qx[:, :, :, 1:nx] = -sigma_x * (P[:, :, :, 1:] - P[:, :, :, :-1]) / cfg.dx
    qx[:, :, :, 0] = -sigma[:, :, :, 0] * (P[:, :, :, 0] - cfg.Pgamma) / (0.5 * cfg.dx)
    qx[:, :, :, nx] = -sigma[:, :, :, -1] * (cfg.Pgamma - P[:, :, :, -1]) / (0.5 * cfg.dx)
    sigma_y = 0.5 * (sigma[:, :, :-1, :] + sigma[:, :, 1:, :])
    qy[:, :, 1:ny, :] = -sigma_y * (P[:, :, 1:, :] - P[:, :, :-1, :]) / cfg.dy
    qy[:, :, 0, :] = -sigma[:, :, 0, :] * (P[:, :, 0, :] - cfg.Pgamma) / (0.5 * cfg.dy)
    qy[:, :, ny, :] = -sigma[:, :, -1, :] * (cfg.Pgamma - P[:, :, -1, :]) / (0.5 * cfg.dy)
    return qx, qy


def water_flux_faces_torch(S, qx, qy, cfg):
    b, t, ny, nx = S.shape
    fw, _ = fw_from_s_torch(torch.clamp(S, cfg.Swc, cfg.Smax), cfg)
    fw_plus = fw_from_s_torch(torch.tensor([[[[cfg.S_plus]]]], device=S.device, dtype=S.dtype), cfg)[0].item()
    Fwx = torch.zeros(b, t, ny, nx + 1, device=S.device, dtype=S.dtype)
    Fwy = torch.zeros(b, t, ny + 1, nx, device=S.device, dtype=S.dtype)
    qx_int = qx[:, :, :, 1:nx]
    qy_int = qy[:, :, 1:ny, :]
    Fwx[:, :, :, 1:nx] = torch.where(qx_int >= 0, fw[:, :, :, :-1], fw[:, :, :, 1:]) * qx_int
    Fwy[:, :, 1:ny, :] = torch.where(qy_int >= 0, fw[:, :, :-1, :], fw[:, :, 1:, :]) * qy_int
    Fwx[:, :, :, 0] = torch.where(qx[:, :, :, 0] > 0, torch.full_like(qx[:, :, :, 0], fw_plus), fw[:, :, :, 0]) * qx[:, :, :, 0]
    Fwx[:, :, :, nx] = torch.where(qx[:, :, :, nx] < 0, torch.full_like(qx[:, :, :, nx], fw_plus), fw[:, :, :, -1]) * qx[:, :, :, nx]
    Fwy[:, :, 0, :] = torch.where(qy[:, :, 0, :] > 0, torch.full_like(qy[:, :, 0, :], fw_plus), fw[:, :, 0, :]) * qy[:, :, 0, :]
    Fwy[:, :, ny, :] = torch.where(qy[:, :, ny, :] < 0, torch.full_like(qy[:, :, ny, :], fw_plus), fw[:, :, -1, :]) * qy[:, :, ny, :]
    return Fwx, Fwy


def active_mask_torch(inj_mask, prod_mask):
    mask = torch.ones_like(inj_mask, dtype=torch.float32)
    mask[:, 0, :] = 0.0; mask[:, -1, :] = 0.0; mask[:, :, 0] = 0.0; mask[:, :, -1] = 0.0
    return mask * (~inj_mask).float() * (~prod_mask).float()


def masked_mean_sq(z, mask, eps=1e-12):
    return (z.pow(2) * mask).sum() / (mask.sum() + eps)


def physics_terms(pred, K_field, H_field, m_field, inj_mask, prod_mask, cfg):
    P, S = pred[:, :, 0], pred[:, :, 1]
    K = K_field[:, None]
    H = H_field[:, None]
    m = m_field[:, None]
    S = torch.clamp(S, cfg.Swc, cfg.Smax)
    S = torch.where(inj_mask[:, None], torch.full_like(S, cfg.Smax), S)
    _, lam_t = fw_from_s_torch(S, cfg)
    sigma = H * K * lam_t + 1e-10
    qx, qy = total_flux_faces_torch(P, sigma, cfg)
    Fwx, Fwy = water_flux_faces_torch(S, qx, qy, cfg)
    div_total = (qx[:, :, :, 1:] - qx[:, :, :, :-1]) / cfg.dx + (qy[:, :, 1:, :] - qy[:, :, :-1, :]) / cfg.dy
    div_water = (Fwx[:, :, :, 1:] - Fwx[:, :, :, :-1]) / cfg.dx + (Fwy[:, :, 1:, :] - Fwy[:, :, :-1, :]) / cfg.dy
    act2d = active_mask_torch(inj_mask, prod_mask)
    act = act2d[:, None].expand(-1, P.shape[1], -1, -1)
    p_res = masked_mean_sq(div_total, act)
    frame_dt = max(cfg.dt * cfg.sat_substeps, 1e-12)
    St = (S[:, 1:] - S[:, :-1]) / frame_dt
    sat_res = m * H * St + div_water[:, :-1]
    s_res = masked_mean_sq(sat_res, act[:, :-1])
    return p_res, s_res


def total_loss(pred_raw, base_raw, hr, K_field, H_field, m_field, inj_mask, prod_mask, epoch, cfg):
    pred = enforce_bc_batch(pred_raw, inj_mask, prod_mask, cfg)
    base = enforce_bc_batch(base_raw, inj_mask, prod_mask, cfg)
    P_pred, S_pred = pred[:, :, 0], pred[:, :, 1]
    P_hr, S_hr = hr[:, :, 0], hr[:, :, 1]
    P_base, S_base = base[:, :, 0], base[:, :, 1]
    rec_p = F.l1_loss(P_pred, P_hr)
    rec_s = F.l1_loss(S_pred, S_hr)
    gp_pred_x, gp_pred_y = grad_xy(P_pred); gp_hr_x, gp_hr_y = grad_xy(P_hr)
    gs_pred_x, gs_pred_y = grad_xy(S_pred); gs_hr_x, gs_hr_y = grad_xy(S_hr)
    grad_p = F.l1_loss(gp_pred_x, gp_hr_x) + F.l1_loss(gp_pred_y, gp_hr_y)
    grad_s = F.l1_loss(gs_pred_x, gs_hr_x) + F.l1_loss(gs_pred_y, gs_hr_y)
    front_loss = F.l1_loss(soft_front_mask(S_pred, cfg), soft_front_mask(S_hr, cfg))
    residual_reg = ((pred - base) ** 2).mean()
    identity_reg = F.l1_loss(P_pred, P_base) + F.l1_loss(S_pred, S_base)
    phys_p, phys_s = physics_terms(pred, K_field, H_field, m_field, inj_mask, prod_mask, cfg)
    phys = physics_weight(epoch, cfg) * phys_s
    loss = (
        cfg.loss_p_l1 * rec_p + cfg.loss_p_grad * grad_p +
        cfg.loss_s_l1 * rec_s + cfg.loss_s_grad * grad_s + cfg.loss_s_front * front_loss +
        cfg.loss_residual_reg * residual_reg + cfg.loss_identity * identity_reg + phys
    )
    parts = {"rec_p": float(rec_p.detach()), "rec_s": float(rec_s.detach()), "front_s": float(front_loss.detach()), "phys_p": float(phys_p.detach()), "phys_s": float(phys_s.detach())}
    return loss, pred, base, parts
