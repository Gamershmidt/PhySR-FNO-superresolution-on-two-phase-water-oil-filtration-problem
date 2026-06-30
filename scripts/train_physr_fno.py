#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch

from physr_fno.models.physr_fno import PhySRFNO
from _smoke_utils import device_from_arg, get_mini_batch, one_supervised_step, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mini", action="store_true", help="Run a tiny CPU/GPU smoke training step")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="results/train_physr_fno_smoke.json")
    args = p.parse_args()
    if not args.mini:
        raise SystemExit("This refactored entry point currently supports --mini smoke run. Add dataset configs for full training.")
    device = device_from_arg(args.device)
    cfg, batch, mean, std = get_mini_batch(device)
    model = PhySRFNO(mean, std, cfg, n_feats=8, n_fno=1, modes=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss, parts, metrics = one_supervised_step(model, batch, cfg, opt)
    save_json({"script": "train_physr_fno", "loss": loss, "loss_parts": parts, "metrics": metrics}, args.out)
    print(f"OK train_physr_fno mini: loss={loss:.6f}; saved {args.out}")


if __name__ == "__main__":
    main()
