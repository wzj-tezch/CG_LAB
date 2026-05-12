from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

from .config import Work1Config
from .data import build_cameras, load_target_mesh, set_seed
from .losses import compose_textured_loss
from .renderers import build_soft_phong_renderer, build_soft_silhouette_renderer
from .visualize import export_mesh_as_obj, save_comparison_pair, save_loss_curves, save_turntable_gif


def _get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _render_silhouette(renderer, mesh, cameras) -> torch.Tensor:
    images = renderer(mesh.extend(len(cameras)), cameras=cameras)
    return images[..., 3]


def run_textured_training(cfg: Work1Config) -> Path:
    device = _get_device()
    set_seed(cfg.seed)
    run_dir = cfg.make_run_dir("textured")

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
        target_rgb = phong_renderer(target_mesh.extend(len(cameras)), cameras=cameras)[..., :3]

    source_mesh = ico_sphere(cfg.ico_level, device)
    num_verts = source_mesh.verts_packed().shape[0]
    deform_verts = torch.zeros_like(source_mesh.verts_packed(), requires_grad=True)
    vertex_logits = torch.zeros((num_verts, 3), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([deform_verts, vertex_logits], lr=cfg.lr_textured)

    history: List[Dict[str, float]] = []

    for step in range(1, cfg.steps + 1):
        optimizer.zero_grad()

        deformed = source_mesh.offset_verts(deform_verts)
        vertex_rgb = torch.sigmoid(vertex_logits)
        textured_mesh = Meshes(
            verts=deformed.verts_list(),
            faces=deformed.faces_list(),
            textures=TexturesVertex(verts_features=vertex_rgb[None]),
        )

        pred_sil = _render_silhouette(sil_renderer, textured_mesh, cameras)
        pred_rgb = phong_renderer(textured_mesh.extend(len(cameras)), cameras=cameras)[..., :3]
        total_loss, terms = compose_textured_loss(
            pred_silhouette=pred_sil,
            target_silhouette=target_sil,
            pred_rgb=pred_rgb,
            target_rgb=target_rgb,
            mesh=textured_mesh,
            cfg=cfg,
        )
        total_loss.backward()
        optimizer.step()

        terms["step"] = float(step)
        history.append(terms)

        if step % cfg.log_every == 0 or step == 1 or step == cfg.steps:
            print(
                f"[Textured] step={step:04d}/{cfg.steps} total={terms['total']:.6f} "
                f"sil={terms['silhouette']:.6f} rgb={terms['rgb']:.6f} "
                f"lap={terms['laplacian']:.6f} edge={terms['edge']:.6f} normal={terms['normal']:.6f}"
            )

        if step % cfg.save_every == 0 or step == cfg.steps:
            save_comparison_pair(
                target=target_rgb[0],
                pred=pred_rgb[0].detach(),
                out_path=run_dir / "images" / f"rgb_step_{step:04d}.png",
                title=f"Textured Optimization Step {step}",
            )
            save_comparison_pair(
                target=target_sil[0],
                pred=pred_sil[0].detach(),
                out_path=run_dir / "images" / f"silhouette_step_{step:04d}.png",
                title=f"Silhouette Constraint Step {step}",
            )

    final_deformed = source_mesh.offset_verts(deform_verts.detach())
    final_rgb = torch.sigmoid(vertex_logits.detach())
    final_mesh = Meshes(
        verts=final_deformed.verts_list(),
        faces=final_deformed.faces_list(),
        textures=TexturesVertex(verts_features=final_rgb[None]),
    )

    export_mesh_as_obj(final_mesh, run_dir / "meshes" / "textured_final.obj")
    torch.save(final_rgb.cpu(), run_dir / "meshes" / "vertex_rgb.pt")
    save_loss_curves(history, run_dir / "plots" / "textured_losses.png")
    save_turntable_gif(
        mesh=final_mesh,
        renderer=phong_renderer,
        device=device,
        distance=cfg.camera_distance,
        num_frames=cfg.turntable_frames,
        out_path=run_dir / "images" / "textured_turntable.gif",
    )
    return run_dir
