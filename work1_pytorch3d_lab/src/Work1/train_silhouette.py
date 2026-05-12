from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

from .config import Work1Config
from .data import build_cameras, load_target_mesh, set_seed
from .losses import compose_silhouette_loss
from .renderers import build_soft_phong_renderer, build_soft_silhouette_renderer
from .visualize import export_mesh_as_obj, save_comparison_pair, save_loss_curves, save_turntable_gif


def _get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _render_silhouette(renderer, mesh, cameras) -> torch.Tensor:
    images = renderer(mesh.extend(len(cameras)), cameras=cameras)
    return images[..., 3]


def run_silhouette_training(cfg: Work1Config) -> Path:
    device = _get_device()
    set_seed(cfg.seed)
    run_dir = cfg.make_run_dir("silhouette")

    target_mesh = load_target_mesh(device, cfg.data_root)
    cameras = build_cameras(
        num_views=cfg.num_views,
        distance=cfg.camera_distance,
        elev_min=cfg.elev_min,
        elev_max=cfg.elev_max,
        device=device,
    )
    sil_renderer = build_soft_silhouette_renderer(
        image_size=cfg.image_size,
        sigma=cfg.sigma,
        faces_per_pixel=cfg.faces_per_pixel_sil,
        device=device,
    )
    phong_renderer = build_soft_phong_renderer(
        image_size=cfg.image_size,
        faces_per_pixel=cfg.faces_per_pixel_rgb,
        device=device,
    )

    with torch.no_grad():
        target_sil = _render_silhouette(sil_renderer, target_mesh, cameras)

    source_mesh = ico_sphere(cfg.ico_level, device)
    deform_verts = torch.zeros_like(source_mesh.verts_packed(), requires_grad=True)
    optimizer = torch.optim.Adam([deform_verts], lr=cfg.lr)

    history: List[Dict[str, float]] = []

    for step in range(1, cfg.steps + 1):
        optimizer.zero_grad()
        deformed_mesh = source_mesh.offset_verts(deform_verts)
        pred_sil = _render_silhouette(sil_renderer, deformed_mesh, cameras)

        total_loss, terms = compose_silhouette_loss(pred_sil, target_sil, deformed_mesh, cfg)
        total_loss.backward()
        optimizer.step()

        terms["step"] = float(step)
        history.append(terms)

        if step % cfg.log_every == 0 or step == 1 or step == cfg.steps:
            print(
                f"[Silhouette] step={step:04d}/{cfg.steps} "
                f"total={terms['total']:.6f} sil={terms['silhouette']:.6f} "
                f"lap={terms['laplacian']:.6f} edge={terms['edge']:.6f} normal={terms['normal']:.6f}"
            )

        if step % cfg.save_every == 0 or step == cfg.steps:
            save_comparison_pair(
                target=target_sil[0],
                pred=pred_sil[0].detach(),
                out_path=run_dir / "images" / f"silhouette_step_{step:04d}.png",
                title=f"Silhouette Optimization Step {step}",
            )

    final_mesh = source_mesh.offset_verts(deform_verts.detach())
    white_rgb = torch.ones((final_mesh.verts_packed().shape[0], 3), device=device)
    final_textured_mesh = Meshes(
        verts=final_mesh.verts_list(),
        faces=final_mesh.faces_list(),
        textures=TexturesVertex(verts_features=white_rgb[None]),
    )
    target_white = torch.ones((target_mesh.verts_packed().shape[0], 3), device=device)
    target_textured_mesh = Meshes(
        verts=target_mesh.verts_list(),
        faces=target_mesh.faces_list(),
        textures=TexturesVertex(verts_features=target_white[None]),
    )
    export_mesh_as_obj(final_mesh, run_dir / "meshes" / "silhouette_final.obj")
    save_loss_curves(history, run_dir / "plots" / "silhouette_losses.png")

    with torch.no_grad():
        final_rgb = phong_renderer(final_textured_mesh.extend(len(cameras)), cameras=cameras)[..., :3]
        target_rgb = phong_renderer(target_textured_mesh.extend(len(cameras)), cameras=cameras)[..., :3]
    save_comparison_pair(
        target=target_rgb[0],
        pred=final_rgb[0],
        out_path=run_dir / "images" / "final_rgb_compare.png",
        title="Final RGB Render Comparison",
    )
    save_turntable_gif(
        mesh=final_textured_mesh,
        renderer=phong_renderer,
        device=device,
        distance=cfg.camera_distance,
        num_frames=cfg.turntable_frames,
        out_path=run_dir / "images" / "silhouette_turntable.gif",
    )
    return run_dir
