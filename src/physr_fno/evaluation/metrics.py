from __future__ import annotations

import torch


def rel_l2_percent(pred, target, eps: float = 1e-12):
    return 100.0 * torch.linalg.vector_norm(pred - target) / (torch.linalg.vector_norm(target) + eps)


def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))


def mae(pred, target):
    return torch.mean(torch.abs(pred - target))


def front_iou(pred_s, target_s, cfg, eps: float = 1e-12):
    theta = cfg.Swc + 0.5 * (cfg.Smax - cfg.Swc)
    mp = pred_s >= theta
    mt = target_s >= theta
    inter = (mp & mt).float().sum()
    union = (mp | mt).float().sum()
    if union.item() == 0:
        return torch.tensor(1.0, device=pred_s.device)
    return inter / (union + eps)


def saturation_bounds_violation(pred_s, cfg):
    low = torch.clamp(cfg.Swc - pred_s, min=0.0)
    high = torch.clamp(pred_s - cfg.Smax, min=0.0)
    violation = low + high
    return violation.mean(), (violation > 0).float().mean()


def metric_pack(pred, target, cfg):
    p_pred, s_pred = pred[:, :, 0], pred[:, :, 1]
    p_tgt, s_tgt = target[:, :, 0], target[:, :, 1]
    ep = rel_l2_percent(p_pred, p_tgt)
    es = rel_l2_percent(s_pred, s_tgt)
    vi_mean, vi_frac = saturation_bounds_violation(s_pred, cfg)
    return {
        "EP_percent": float(ep.detach().cpu()),
        "ES_percent": float(es.detach().cpu()),
        "E_avg_percent": float(((ep + es) / 2.0).detach().cpu()),
        "RMSE_P": float(rmse(p_pred, p_tgt).detach().cpu()),
        "RMSE_S": float(rmse(s_pred, s_tgt).detach().cpu()),
        "MAE_P": float(mae(p_pred, p_tgt).detach().cpu()),
        "MAE_S": float(mae(s_pred, s_tgt).detach().cpu()),
        "Front_IoU": float(front_iou(s_pred, s_tgt, cfg).detach().cpu()),
        "Sat_violation_mean": float(vi_mean.detach().cpu()),
        "Sat_violation_fraction": float(vi_frac.detach().cpu()),
    }


def evaluate_prediction(pred, target, cfg, prefix: str = ""):
    out = metric_pack(pred, target, cfg)
    return {f"{prefix}{k}": v for k, v in out.items()} if prefix else out
