from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from physr_fno.solvers.finite_volume import FiltrationDataset, compute_channel_stats, generate_items


def make_loaders(cfg, n_cases: int | None = None):
    items = generate_items(cfg, n_cases=n_cases)
    train = items[: max(1, len(items) - 1)]
    val = items[max(1, len(items) - 1) :]
    if not val:
        val = train[:1]
    mean, std = compute_channel_stats(train)
    return (
        DataLoader(FiltrationDataset(train), batch_size=cfg.batch_size, shuffle=True),
        DataLoader(FiltrationDataset(val), batch_size=1, shuffle=False),
        mean,
        std,
        items,
    )


def first_batch(cfg, device: str = "cpu"):
    loader, _, mean, std, _ = make_loaders(cfg, n_cases=1)
    batch = next(iter(loader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    return batch, mean.to(device), std.to(device)
