from __future__ import annotations

import random
import urllib.request
from pathlib import Path

import numpy as np
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform


_COW_BASE_URL = "https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/docs/tutorials/data/cow_mesh"
_COW_FILES = ("cow.obj", "cow.mtl", "cow_texture.png")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _download(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return
    urllib.request.urlretrieve(url, target_path.as_posix())


def ensure_cow_files(data_root: str | Path) -> Path:
    mesh_dir = Path(data_root) / "cow_mesh"
    for name in _COW_FILES:
        _download(f"{_COW_BASE_URL}/{name}", mesh_dir / name)
    return mesh_dir / "cow.obj"


def _normalized_mesh(mesh):
    verts = mesh.verts_packed()
    center = verts.mean(0, keepdim=True)
    scale = float((verts - center).abs().max().clamp(min=1e-6).item())
    centered = mesh.offset_verts((-center).expand_as(verts))
    centered.scale_verts_(1.0 / scale)
    return centered


def load_target_mesh(device: torch.device, data_root: str | Path):
    obj_path = ensure_cow_files(data_root)
    mesh = load_objs_as_meshes([obj_path.as_posix()], device=device)
    return _normalized_mesh(mesh)


def build_cameras(
    *,
    num_views: int,
    distance: float,
    elev_min: float,
    elev_max: float,
    device: torch.device,
) -> FoVPerspectiveCameras:
    elev = torch.linspace(elev_min, elev_max, num_views, device=device)
    azim = torch.linspace(-180.0, 180.0, num_views, device=device)
    r, t = look_at_view_transform(dist=distance, elev=elev, azim=azim, device=device)
    return FoVPerspectiveCameras(device=device, R=r, T=t)
