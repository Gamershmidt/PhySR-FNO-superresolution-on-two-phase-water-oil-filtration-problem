# PhySR-FNO reservoir super-resolution

Refactored contest-ready project for physics-informed super-resolution of two-phase water-oil filtration fields.

The repository keeps the original notebooks as a research archive and exposes the core code as importable modules under `src/physr_fno`.

## Structure

```text
notebooks/
  research_archive/
    PhySR_solver_updated.ipynb
    DB_AFNO_dataset_metrics_final.ipynb
    PiRD_dataset_metrics.ipynb
    PhySR_FNO_solver_updated.ipynb

src/
  physr_fno/
    models/
      physr_fno.py
      baselines/
        physr.py
        pird.py
        db_afno.py
        epinn.py
    solvers/
      finite_volume.py
    losses/
      physics.py
    evaluation/
      metrics.py

scripts/
  train_physr_fno.py
  train_baseline_epinn.py
  train_baseline_physr.py
  train_baseline_pird.py
  train_baseline_db_afno.py
  evaluate_all.py
```

## Smoke test

```bash
python -m pip install -e .
python scripts/evaluate_all.py --mini
python scripts/train_physr_fno.py --mini
python scripts/train_baseline_epinn.py --mini
```

The `--mini` mode generates a tiny synthetic finite-volume sample on a 16×16 grid, builds a 4×4 low-resolution input, instantiates PhySR-FNO and the SR baselines, and runs a forward/evaluation pass. The ePINN smoke script trains the pressure branch on one FV-generated frame using graph-augmented cell features and a small finite-volume pressure residual term.

## Baselines

- Finite-volume synthetic data generator/reference solver
- ePINN pressure baseline with mini smoke-training entry point
- PhySR ConvLSTM super-resolution baseline
- PiRD diffusion-style residual reconstruction baseline
- DB-AFNO dual-branch adaptive Fourier neural operator baseline
- Proposed PhySR-FNO model

## Notes

The notebooks in `notebooks/research_archive/` are preserved as the original research logs. The scripts are intentionally lightweight smoke-test entry points; full reproduction should connect them to the original HDF5 datasets/checkpoints and longer training configs.
