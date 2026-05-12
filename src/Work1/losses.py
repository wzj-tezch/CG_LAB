from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from pytorch3d.loss import mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency

from .config import Work1Config


def regularization_losses(mesh) -> Dict[str, torch.Tensor]:
    return {
        "laplacian": mesh_laplacian_smoothing(mesh, method="uniform"),
        "edge": mesh_edge_loss(mesh),
        "normal": mesh_normal_consistency(mesh),
    }


def compose_silhouette_loss(
    pred_silhouette: torch.Tensor,
    target_silhouette: torch.Tensor,
    mesh,
    cfg: Work1Config,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    sil = F.mse_loss(pred_silhouette, target_silhouette)
    reg = regularization_losses(mesh)
    total = (
        cfg.w_silhouette * sil
        + cfg.w_laplacian * reg["laplacian"]
        + cfg.w_edge * reg["edge"]
        + cfg.w_normal * reg["normal"]
    )
    terms = {
        "total": float(total.detach().cpu()),
        "silhouette": float(sil.detach().cpu()),
        "laplacian": float(reg["laplacian"].detach().cpu()),
        "edge": float(reg["edge"].detach().cpu()),
        "normal": float(reg["normal"].detach().cpu()),
    }
    return total, terms


def compose_textured_loss(
    pred_silhouette: torch.Tensor,
    target_silhouette: torch.Tensor,
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    mesh,
    cfg: Work1Config,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    sil = F.mse_loss(pred_silhouette, target_silhouette)
    rgb = F.mse_loss(pred_rgb, target_rgb)
    reg = regularization_losses(mesh)
    total = (
        cfg.w_silhouette * sil
        + cfg.w_rgb * rgb
        + cfg.w_laplacian * reg["laplacian"]
        + cfg.w_edge * reg["edge"]
        + cfg.w_normal * reg["normal"]
    )
    terms = {
        "total": float(total.detach().cpu()),
        "silhouette": float(sil.detach().cpu()),
        "rgb": float(rgb.detach().cpu()),
        "laplacian": float(reg["laplacian"].detach().cpu()),
        "edge": float(reg["edge"].detach().cpu()),
        "normal": float(reg["normal"].detach().cpu()),
    }
    return total, terms
