from __future__ import annotations

import numpy as np
import torch
from pytorch3d.renderer import (
    BlendParams,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    SoftSilhouetteShader,
)


def build_soft_silhouette_renderer(
    *,
    image_size: int,
    sigma: float,
    faces_per_pixel: int,
    device: torch.device,
) -> MeshRenderer:
    blur_radius = np.log(1.0 / 1e-4 - 1.0) * sigma
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=float(blur_radius),
        faces_per_pixel=faces_per_pixel,
    )
    blend_params = BlendParams(sigma=sigma, gamma=sigma)
    return MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params),
    )


def build_soft_phong_renderer(
    *,
    image_size: int,
    faces_per_pixel: int,
    device: torch.device,
) -> MeshRenderer:
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=faces_per_pixel,
    )
    lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
    return MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, lights=lights),
    )
