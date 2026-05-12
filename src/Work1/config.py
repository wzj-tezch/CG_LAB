from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class Work1Config:
    seed: int = 42
    output_root: str = "outputs/work1"
    data_root: str = "data/work1"

    image_size: int = 256
    num_views: int = 24
    camera_distance: float = 2.7
    elev_min: float = -30.0
    elev_max: float = 30.0

    sigma: float = 1e-4
    faces_per_pixel_sil: int = 100
    faces_per_pixel_rgb: int = 1

    ico_level: int = 4
    steps: int = 300
    lr: float = 0.08
    lr_textured: float = 0.05

    w_silhouette: float = 1.0
    w_rgb: float = 1.0
    w_laplacian: float = 0.5
    w_edge: float = 1.0
    w_normal: float = 0.01

    log_every: int = 10
    save_every: int = 25

    turntable_frames: int = 48

    def make_run_dir(self, mode: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(self.output_root) / f"{mode}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
