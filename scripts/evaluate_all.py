#!/usr/bin/env python
from __future__ import annotations

import argparse
import time

import torch

from physr_fno.losses.physics import enforce_bc_batch
from physr_fno.evaluation.metrics import metric_pack
from physr_fno.models.physr_fno import PhySRFNO
from physr_fno.models.baselines.physr import ResidualPhySRConvLSTM
from physr_fno.models.baselines.pird import PiRDWrapper
from physr_fno.models.baselines.db_afno import DBAFNOModel
from _smoke_utils import device_from_arg, get_mini_batch, save_json


def static_stats(batch):
    feats = torch.stack([torch.log1p(batch["K"]), batch["H"], batch["m"], batch["inj"].float(), batch["prod"].float()], dim=1)
    return feats.mean(dim=(0, 2, 3)), feats.std(dim=(0, 2, 3)) + 1e-6


def run_model(name, model, batch, cfg):
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        if name == "DB-AFNO":
            pred_raw, base = model(batch["lr"], batch["K"], batch["H"], batch["m"], batch["inj"], batch["prod"])
        elif name == "PhySR-FNO":
            pred_raw, base = model(batch["lr"], batch["K"], batch["H"], batch["m"])
        else:
            pred_raw, base = model(batch["lr"])
        pred = enforce_bc_batch(pred_raw, batch["inj"], batch["prod"], cfg)
        metrics = metric_pack(pred, batch["hr"], cfg)
    return {"runtime_s": time.perf_counter() - t0, **metrics}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mini", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="results/smoke_metrics.json")
    args = p.parse_args()
    if not args.mini:
        raise SystemExit("Use --mini for the smoke run. Full evaluation should point to trained checkpoints and HDF5 datasets.")
    device = device_from_arg(args.device)
    cfg, batch, mean, std = get_mini_batch(device)
    static_mean, static_std = static_stats(batch)
    models = {
        "PhySR-FNO": PhySRFNO(mean, std, cfg, n_feats=8, n_fno=1, modes=4).to(device),
        "PhySR": ResidualPhySRConvLSTM(mean, std, cfg, n_feats=8, n_blocks=1, hidden_dim=8).to(device),
        "PiRD": PiRDWrapper(mean, std, cfg, base_ch=8).to(device),
        "DB-AFNO": DBAFNOModel(mean, std, static_mean, static_std, cfg).to(device),
    }
    results = {name: run_model(name, model, batch, cfg) for name, model in models.items()}
    results["config"] = {"nx_hr": cfg.nx_hr, "ny_hr": cfg.ny_hr, "nt_hr": cfg.nt_hr, "nx_lr": cfg.nx_lr, "ny_lr": cfg.ny_lr, "nt_lr": cfg.nt_lr}
    save_json(results, args.out)
    print(f"OK evaluate_all mini: saved {args.out}")
    for name, metrics in results.items():
        if name != "config":
            print(f"{name:10s} E_avg={metrics['E_avg_percent']:.3f}% IoU={metrics['Front_IoU']:.3f} runtime={metrics['runtime_s']:.4f}s")


if __name__ == "__main__":
    main()
