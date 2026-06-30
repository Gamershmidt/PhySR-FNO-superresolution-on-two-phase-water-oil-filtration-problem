#!/usr/bin/env python
from __future__ import annotations

import argparse
import torch

from physr_fno.models.baselines.db_afno import DBAFNOModel
from _smoke_utils import device_from_arg, get_mini_batch, one_supervised_step, save_json


def static_stats(batch):
    feats = torch.stack([
        torch.log1p(batch["K"]),
        batch["H"],
        batch["m"],
        batch["inj"].float(),
        batch["prod"].float(),
    ], dim=1)
    return feats.mean(dim=(0, 2, 3)), feats.std(dim=(0, 2, 3)) + 1e-6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mini", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="results/train_db_afno_smoke.json")
    args = p.parse_args()
    if not args.mini:
        raise SystemExit("Use --mini for the smoke run; full training should be wired to your HDF5 configs.")
    device = device_from_arg(args.device)
    cfg, batch, mean, std = get_mini_batch(device)
    static_mean, static_std = static_stats(batch)
    model = DBAFNOModel(mean, std, static_mean, static_std, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss, parts, metrics = one_supervised_step(model, batch, cfg, opt)
    save_json({"script": "train_baseline_db_afno", "loss": loss, "loss_parts": parts, "metrics": metrics}, args.out)
    print(f"OK train_baseline_db_afno mini: loss={loss:.6f}; saved {args.out}")


if __name__ == "__main__":
    main()
