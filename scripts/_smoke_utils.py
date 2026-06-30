from __future__ import annotations

import json
from pathlib import Path

import torch

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from physr_fno.config import make_mini_config
from physr_fno.data import first_batch
from physr_fno.evaluation.metrics import metric_pack
from physr_fno.losses.physics import enforce_bc_batch, total_loss


def device_from_arg(name: str):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def get_mini_batch(device):
    cfg = make_mini_config()
    batch, mean, std = first_batch(cfg, str(device))
    return cfg, batch, mean, std


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def _forward_any(model, batch):
    try:
        return model(batch["lr"], batch.get("K"), batch.get("H"), batch.get("m"), batch.get("inj"), batch.get("prod"))
    except TypeError:
        try:
            return model(batch["lr"], batch.get("K"), batch.get("H"), batch.get("m"))
        except TypeError:
            return model(batch["lr"])


def one_supervised_step(model, batch, cfg, optimizer=None):
    pred_raw, base_raw = _forward_any(model, batch)
    loss, pred, base, parts = total_loss(
        pred_raw,
        base_raw,
        batch["hr"],
        batch["K"],
        batch["H"],
        batch["m"],
        batch["inj"],
        batch["prod"],
        0,
        cfg,
    )
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    metrics = metric_pack(pred.detach(), batch["hr"], cfg)
    return float(loss.detach().cpu()), parts, metrics


def evaluate_sr_model(model, batch, cfg):
    with torch.no_grad():
        out = model(batch["lr"], batch.get("K"), batch.get("H"), batch.get("m"), batch.get("inj"), batch.get("prod"))
        pred_raw, base_raw = out if isinstance(out, tuple) else (out, None)
        pred = enforce_bc_batch(pred_raw, batch["inj"], batch["prod"], cfg)
        return metric_pack(pred, batch["hr"], cfg)
