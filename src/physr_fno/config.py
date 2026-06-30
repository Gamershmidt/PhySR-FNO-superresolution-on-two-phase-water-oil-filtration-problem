from __future__ import annotations

from dataclasses import dataclass, asdict
import math


@dataclass
class Config:
    # grid / time
    nx_hr: int = 64
    ny_hr: int = 64
    nt_hr: int = 30
    s_up: int = 8
    t_up: int = 3

    # physical parameters
    mu_w: float = 1.0
    mu_n: float = 4.0
    Swc: float = 0.20
    Smax: float = 0.90
    n_w: float = 3.0
    n_n: float = 2.0

    Pinj: float = 1.00
    Pprod: float = 0.22
    Pgamma: float = 0.22
    S_plus: float = 0.23

    H0: float = 1.0
    m0: float = 0.24

    # synthetic solver
    dt: float = 0.0025
    sat_substeps: int = 30
    well_radius: int = 2
    n_cases: int = 80
    cfl: float = 0.45
    epsilon_sat: float = 1e-4
    epsilon_front: float = 0.05
    split_seed: int = 42

    # training / losses
    batch_size: int = 1
    lr: float = 1.5e-3
    weight_decay: float = 1e-6
    epochs: int = 55
    physics_start_epoch: int = 28
    physics_ramp_epochs: int = 14
    max_phys_w: float = 9e-6
    alpha_p: float = 0.06
    alpha_s: float = 0.08
    front_theta: float = 0.55
    front_k: float = 28.0
    loss_p_l1: float = 1.0
    loss_p_grad: float = 0.12
    loss_s_l1: float = 1.0
    loss_s_grad: float = 0.22
    loss_s_front: float = 0.35
    loss_residual_reg: float = 0.01
    loss_identity: float = 0.01

    # DB-AFNO knobs
    fno_width: int = 64
    fno_layers: int = 8
    fno_modes_x_max: int = 18
    fno_modes_y_max: int = 18
    time_emb_dim: int = 64
    pressure_dropout: float = 0.2
    gamma_p_init: float = 2.0
    alpha_cross_init: float = 0.10
    beta_cross_init: float = 0.10
    curriculum_modes: tuple = ((2, 2), (6, 6), (18, 18), (18, 18))

    # PiRD knobs
    ctx_frames: int = 3
    diffusion_steps: int = 20
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    residual_clip_p: float = 0.08
    residual_clip_s: float = 0.12

    # derived fields populated by finalize_config
    nx_lr: int | None = None
    ny_lr: int | None = None
    nt_lr: int | None = None
    dx: float | None = None
    dy: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def finalize_config(cfg: Config) -> Config:
    if cfg.nx_hr % cfg.s_up != 0 or cfg.ny_hr % cfg.s_up != 0:
        raise ValueError("HR grid must be divisible by s_up")
    cfg.nx_lr = cfg.nx_hr // cfg.s_up
    cfg.ny_lr = cfg.ny_hr // cfg.s_up
    cfg.nt_lr = math.ceil(cfg.nt_hr / cfg.t_up)
    cfg.dx = 1.0 / max(1, cfg.nx_hr - 1)
    cfg.dy = 1.0 / max(1, cfg.ny_hr - 1)
    return cfg


def make_mini_config() -> Config:
    cfg = Config(
        nx_hr=16,
        ny_hr=16,
        nt_hr=6,
        s_up=4,
        t_up=2,
        sat_substeps=1,
        well_radius=1,
        n_cases=2,
        batch_size=1,
        epochs=1,
        physics_start_epoch=10**9,
        fno_width=8,
        fno_layers=1,
        fno_modes_x_max=4,
        fno_modes_y_max=4,
        time_emb_dim=8,
        curriculum_modes=((2, 2),),
        ctx_frames=3,
        diffusion_steps=4,
    )
    return finalize_config(cfg)
