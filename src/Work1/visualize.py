from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.io import save_obj
from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform


def _to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    return np.clip(arr, 0.0, 1.0)


def save_comparison_pair(target: torch.Tensor, pred: torch.Tensor, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(_to_numpy_image(target), cmap="gray" if target.ndim == 2 else None)
    axes[0].set_title("Target")
    axes[0].axis("off")
    axes[1].imshow(_to_numpy_image(pred), cmap="gray" if pred.ndim == 2 else None)
    axes[1].set_title("Optimized")
    axes[1].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_loss_curves(history: List[Dict[str, float]], out_path: Path) -> None:
    if not history:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = [k for k in history[0].keys() if k != "step"]
    steps = [int(item["step"]) for item in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in keys:
        values = [float(item[key]) for item in history]
        ax.plot(steps, values, label=key)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curves")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def export_mesh_as_obj(mesh, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    verts = mesh.verts_packed().detach().cpu()
    faces = mesh.faces_packed().detach().cpu()
    save_obj(out_path.as_posix(), verts=verts, faces=faces)


def save_turntable_gif(
    *,
    mesh,
    renderer,
    device: torch.device,
    distance: float,
    num_frames: int,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    azims = torch.linspace(-180.0, 180.0, num_frames, device=device)
    elevs = torch.zeros_like(azims)
    r, t = look_at_view_transform(dist=distance, elev=elevs, azim=azims, device=device)
    cameras = FoVPerspectiveCameras(device=device, R=r, T=t)
    renders = renderer(mesh.extend(num_frames), cameras=cameras)[..., :3]
    for idx in range(num_frames):
        frame = (_to_numpy_image(renders[idx]) * 255).astype(np.uint8)
        frames.append(frame)
    imageio.mimsave(out_path.as_posix(), frames, fps=12)
