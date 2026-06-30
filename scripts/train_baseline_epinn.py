#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from physr_fno.losses.physics import active_mask_torch, fw_from_s_torch, masked_mean_sq, total_flux_faces_torch
from physr_fno.models.baselines.epinn import EPINN
from _smoke_utils import device_from_arg, get_mini_batch, save_json


def four_neighbor_adjacency(ny: int, nx: int, device: torch.device) -> torch.Tensor:
    n = ny * nx
    A = torch.zeros((n, n), dtype=torch.float32, device=device)
    for y in range(ny):
        for x in range(nx):
            i = y * nx + x
            neighbors = []
            if x > 0:
                neighbors.append(y * nx + x - 1)
            if x < nx - 1:
                neighbors.append(y * nx + x + 1)
            if y > 0:
                neighbors.append((y - 1) * nx + x)
            if y < ny - 1:
                neighbors.append((y + 1) * nx + x)
            if not neighbors:
                A[i, i] = 1.0
            else:
                w = 1.0 / len(neighbors)
                for j in neighbors:
                    A[i, j] = w
    return A


def build_epinn_batch(batch: dict, cfg, frame_idx: int = 1):
    hr = batch["hr"][:1]
    device = hr.device
    _, _, _, ny, nx = hr.shape
    frame_idx = max(1, min(frame_idx, hr.shape[1] - 1))

    P_prev = hr[0, frame_idx - 1, 0]
    S_prev = hr[0, frame_idx - 1, 1]
    P_target = hr[0, frame_idx, 0]
    K = batch["K"][0]
    H = batch["H"][0]
    m = batch["m"][0]
    inj = batch["inj"][0]
    prod = batch["prod"][0]

    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, ny, device=device),
        torch.linspace(0.0, 1.0, nx, device=device),
        indexing="ij",
    )
    t = torch.full_like(xx, frame_idx / max(1, hr.shape[1] - 1))
    X = torch.stack(
        [
            xx,
            yy,
            t,
            S_prev,
            torch.log1p(K),
            m,
            inj.float(),
            prod.float(),
            P_prev,
        ],
        dim=-1,
    ).reshape(-1, 9)

    boundary = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True
    bc_mask = (boundary | inj | prod).reshape(-1).float()
    bc_value = torch.full((ny, nx), cfg.Pgamma, dtype=torch.float32, device=device)
    bc_value[inj] = cfg.Pinj
    bc_value[prod] = cfg.Pprod

    return {
        "X": X,
        "A": four_neighbor_adjacency(ny, nx, device),
        "bc_mask": bc_mask,
        "bc_value": bc_value.reshape(-1),
        "p_anchor": P_prev.reshape(-1),
        "P_target": P_target,
        "S_prev": S_prev,
        "K": K,
        "H": H,
        "m": m,
        "inj": inj,
        "prod": prod,
        "ny": ny,
        "nx": nx,
    }


def pressure_residual_loss(P_pred: torch.Tensor, S: torch.Tensor, K: torch.Tensor, H: torch.Tensor, inj: torch.Tensor, prod: torch.Tensor, cfg) -> torch.Tensor:
    P = P_pred[None, None]
    S4 = S[None, None]
    _, lam_t = fw_from_s_torch(torch.clamp(S4, cfg.Swc, cfg.Smax), cfg)
    sigma = H[None, None] * K[None, None] * lam_t + 1e-10
    qx, qy = total_flux_faces_torch(P, sigma, cfg)
    div_total = (qx[:, :, :, 1:] - qx[:, :, :, :-1]) / cfg.dx + (qy[:, :, 1:, :] - qy[:, :, :-1, :]) / cfg.dy
    act = active_mask_torch(inj[None], prod[None])[:, None]
    return masked_mean_sq(div_total, act)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mini", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="results/train_baseline_epinn_smoke.json")
    p.add_argument("--steps", type=int, default=3)
    args = p.parse_args()
    if not args.mini:
        raise SystemExit("Use --mini for the smoke run; full ePINN training should be wired to the CFL/PINN HDF5 configs.")

    device = device_from_arg(args.device)
    cfg, batch, _, _ = get_mini_batch(device)
    eb = build_epinn_batch(batch, cfg, frame_idx=1)

    model = EPINN(in_dim=9, hidden=32, depth=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-6)

    history = []
    for _ in range(max(1, args.steps)):
        pred_flat = model(eb["X"], eb["A"], eb["bc_mask"], eb["bc_value"], eb["p_anchor"], corr_scale=0.15).reshape(eb["ny"], eb["nx"])
        data_loss = F.mse_loss(pred_flat, eb["P_target"])
        p_res = pressure_residual_loss(pred_flat, eb["S_prev"], eb["K"], eb["H"], eb["inj"], eb["prod"], cfg)
        loss = data_loss + 1e-4 * p_res
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history.append({"loss": float(loss.detach().cpu()), "mse_pressure": float(data_loss.detach().cpu()), "pressure_residual": float(p_res.detach().cpu())})

    with torch.no_grad():
        pred_flat = model(eb["X"], eb["A"], eb["bc_mask"], eb["bc_value"], eb["p_anchor"], corr_scale=0.15).reshape(eb["ny"], eb["nx"])
        rel_l2 = 100.0 * torch.linalg.vector_norm(pred_flat - eb["P_target"]) / (torch.linalg.vector_norm(eb["P_target"]) + 1e-12)
        max_abs = torch.max(torch.abs(pred_flat - eb["P_target"]))

    save_json(
        {
            "script": "train_baseline_epinn",
            "description": "Mini smoke training for the ePINN pressure branch on one FV-generated frame.",
            "steps": len(history),
            "history": history,
            "final_rel_l2_pressure_percent": float(rel_l2.cpu()),
            "final_max_abs_pressure_error": float(max_abs.cpu()),
            "config": {"nx_hr": cfg.nx_hr, "ny_hr": cfg.ny_hr, "nt_hr": cfg.nt_hr},
        },
        args.out,
    )
    print(f"OK train_baseline_epinn mini: relL2(P)={float(rel_l2):.3f}%; saved {args.out}")


if __name__ == "__main__":
    main()
