"""Hand-written SMPL LBS pipeline with explicit intermediate variables."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from smplx.lbs import (
    batch_rigid_transform,
    batch_rodrigues,
    blend_shapes,
    vertices2joints,
)


@dataclass
class LBSStages:
    v_template: torch.Tensor
    v_shaped: torch.Tensor
    J: torch.Tensor
    pose_offsets: torch.Tensor
    v_posed: torch.Tensor
    J_transformed: torch.Tensor
    verts: torch.Tensor
    rot_mats: torch.Tensor
    transform_mats: torch.Tensor


def manual_lbs(
    betas: torch.Tensor,
    full_pose: torch.Tensor,
    v_template: torch.Tensor,
    shapedirs: torch.Tensor,
    posedirs: torch.Tensor,
    J_regressor: torch.Tensor,
    parents: torch.Tensor,
    lbs_weights: torch.Tensor,
) -> LBSStages:
    """Re-implement smplx.lbs.lbs while exposing all intermediate tensors."""
    batch_size = betas.shape[0]
    device, dtype = betas.device, betas.dtype
    ident = torch.eye(3, dtype=dtype, device=device)

    v_shaped = v_template + blend_shapes(betas, shapedirs)
    J = vertices2joints(J_regressor, v_shaped)

    rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(batch_size, -1, 3, 3)
    pose_feature = (rot_mats[:, 1:, :, :] - ident).reshape(batch_size, -1)
    pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)
    v_posed = v_shaped + pose_offsets

    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    num_joints = J_regressor.shape[0]
    W = lbs_weights.unsqueeze(0).expand(batch_size, -1, -1)
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)

    homo = torch.ones(batch_size, v_posed.shape[1], 1, dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homo], dim=2)
    v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))
    verts = v_homo[:, :, :3, 0]

    return LBSStages(
        v_template=v_template.expand(batch_size, -1, -1),
        v_shaped=v_shaped,
        J=J,
        pose_offsets=pose_offsets,
        v_posed=v_posed,
        J_transformed=J_transformed,
        verts=verts,
        rot_mats=rot_mats,
        transform_mats=T,
    )
