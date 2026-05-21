from dataclasses import dataclass, field
from pathlib import Path

import taichi as ti


@dataclass
class SimConfig:
    arch: object = field(default_factory=lambda: ti.gpu)
    width: int = 1280
    height: int = 720
    grid_n: int = 20
    grid_m: int = 20
    particle_mass: float = 1.0
    dt: float = 1.0 / 240.0
    substeps: int = 6
    gravity: tuple[float, float, float] = (0.0, -9.8, 0.0)
    damping: float = 1.0
    spring_damping: float = 8.0
    air_drag: float = 0.06
    ks_structural: float = 900.0
    ks_shear: float = 650.0
    ks_bending: float = 450.0
    constraint_stiffness: float = 0.55
    constraint_iters: int = 3
    max_velocity: float = 6.0
    implicit_iters: int = 6
    cloth_size: float = 1.0
    cloth_origin: tuple[float, float, float] = (-0.5, 0.7, -0.5)
    sphere_center: tuple[float, float, float] = (0.0, 0.05, 0.0)
    sphere_radius: float = 0.22
    restitution: float = 0.05
    friction: float = 0.12
    ground_height: float = -1.0
    wind_strength: float = 5.0
    wind_frequency: float = 0.8
    run_name: str = "default"


SOLVER_EXPLICIT = 0
SOLVER_SEMI_IMPLICIT = 1
SOLVER_IMPLICIT = 2

SOLVER_NAME = {
    SOLVER_EXPLICIT: "Explicit Euler",
    SOLVER_SEMI_IMPLICIT: "Semi-Implicit Euler",
    SOLVER_IMPLICIT: "Implicit Euler (fixed-point)",
}


def output_root() -> Path:
    return Path(__file__).resolve().parent / "outputs"
