"""SMPL LBS experiment runner – tasks 1–7 + optional animation."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import smplx
import torch

from manual_lbs import manual_lbs
from setup_model import ensure_model
from visualize import make_comparison_grid, render_dominant_joint_map, render_mesh

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
ASSETS = ROOT / "assets"

# Experiment parameters (fixed for reproducibility)
BETAS = torch.tensor([[1.6, 0.9, -0.6, 0.3, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
GLOBAL_ORIENT = torch.tensor([[0.0, 0.15, 0.0]], dtype=torch.float32)
BODY_POSE = torch.zeros(1, 69, dtype=torch.float32)
# right shoulder raise + elbow bend + slight torso twist
BODY_POSE[0, 17 * 3 + 2] = -1.2   # right_shoulder
BODY_POSE[0, 19 * 3 + 2] = -1.4   # right_elbow
BODY_POSE[0, 16 * 3 + 1] = 0.35   # left_shoulder slight
BODY_POSE[0, 5 * 3 + 1] = 0.25    # spine twist via right_knee anchor proxy -> use spine
BODY_POSE[0, 2 * 3 + 1] = 0.18    # right_hip slight
BODY_POSE[0, 6 * 3 + 2] = 0.12    # spine2

WEIGHT_JOINT_IDX = 17  # right_shoulder for stage (a)


def load_model() -> smplx.SMPL:
    model_path, official = ensure_model()
    if not official:
        raise RuntimeError("Official SMPL model required. Place SMPL_NEUTRAL.pkl under models/smpl/.")
    return smplx.create(
        str(model_path),
        model_type="smpl",
        gender="neutral",
        batch_size=1,
        num_betas=10,
    )


def task1_summary(model: smplx.SMPL) -> dict:
    info = {
        "vertices": int(model.get_num_verts()),
        "faces": int(model.faces.shape[0]),
        "joints": int(model.J_regressor.shape[0]),
        "betas_dim": int(model.num_betas),
        "model_path": str(ROOT / "models" / "smpl" / "SMPL_NEUTRAL.pkl"),
    }
    print("=== Task 1: Model Info ===")
    for k, v in info.items():
        print(f"  {k}: {v}")
    return info


def run_stages(model: smplx.SMPL):
    device = torch.device("cpu")
    model = model.to(device)
    betas = BETAS.to(device)
    global_orient = GLOBAL_ORIENT.to(device)
    body_pose = BODY_POSE.to(device)
    full_pose = torch.cat([global_orient, body_pose], dim=1)

    v_template = model.v_template.unsqueeze(0)
    shapedirs = model.shapedirs
    posedirs = model.posedirs
    J_regressor = model.J_regressor
    parents = model.parents
    lbs_weights = model.lbs_weights

    stages = manual_lbs(
        betas=betas,
        full_pose=full_pose,
        v_template=v_template,
        shapedirs=shapedirs,
        posedirs=posedirs,
        J_regressor=J_regressor,
        parents=parents,
        lbs_weights=lbs_weights,
    )

    with torch.no_grad():
        official = model(betas=betas, global_orient=global_orient, body_pose=body_pose)

    return stages, official


def save_visualizations(model: smplx.SMPL, stages, official) -> None:
    faces = np.asarray(model.faces)
    w_np = model.lbs_weights.cpu().numpy()

    v_tpl = stages.v_template[0].cpu().numpy()
    v_shaped = stages.v_shaped[0].cpu().numpy()
    J = stages.J[0].cpu().numpy()
    pose_off = stages.pose_offsets[0].cpu().numpy()
    v_posed = stages.v_posed[0].cpu().numpy()
    verts = stages.verts[0].cpu().numpy()
    J_tr = stages.J_transformed[0].cpu().numpy()

    # Task 2a – single joint weights
    w_joint = w_np[:, WEIGHT_JOINT_IDX]
    render_mesh(
        v_tpl, faces, OUT / "stage_a_template_weights.png",
        vertex_colors=w_joint, title=f"Stage (a): Joint {WEIGHT_JOINT_IDX} LBS weights",
        cmap="plasma",
    )

    # Task 2b optional – dominant joint map
    render_dominant_joint_map(
        v_tpl, faces, w_np, OUT / "all_joint_weights.png",
        title="All Joint LBS Weights (dominant joint)",
    )

    # Task 3 – shaped + joints
    render_mesh(
        v_shaped, faces, OUT / "stage_b_shaped_joints.png",
        joints=J, title="Stage (b): Shape blend + regressed joints", wireframe=True,
    )

    # Task 4 – pose offsets magnitude
    offset_mag = np.linalg.norm(pose_off, axis=1)
    render_mesh(
        v_posed, faces, OUT / "stage_c_pose_offsets.png",
        vertex_colors=offset_mag, title="Stage (c): |pose offsets| on v_posed", cmap="inferno",
    )

    # Task 5 – final LBS
    render_mesh(
        verts, faces, OUT / "stage_d_lbs_result.png",
        joints=J_tr, title="Stage (d): Final LBS result", wireframe=True,
    )

    # Task 6 – comparison grid
    make_comparison_grid(
        {
            "a": OUT / "stage_a_template_weights.png",
            "b": OUT / "stage_b_shaped_joints.png",
            "c": OUT / "stage_c_pose_offsets.png",
            "d": OUT / "stage_d_lbs_result.png",
        },
        OUT / "comparison_grid.png",
    )


def task7_verify(stages, official, info: dict) -> None:
    manual_v = stages.verts.detach()
    official_v = official.vertices.detach()
    abs_err = (manual_v - official_v).abs()
    mae = abs_err.mean().item()
    max_err = abs_err.max().item()
    per_vertex = abs_err.norm(dim=-1).squeeze(0).cpu().numpy()

    info["manual_vs_official_mae"] = mae
    info["manual_vs_official_max"] = max_err

    # error histogram chart
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
    ax.hist(per_vertex, bins=80, color="#4C72B0", edgecolor="white", alpha=0.9)
    ax.set_xlabel("Per-vertex L2 error")
    ax.set_ylabel("Count")
    ax.set_title("Hand-written LBS vs official forward pass")
    ax.axvline(mae, color="crimson", linestyle="--", label=f"MAE={mae:.2e}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "charts" / "lbs_error_hist.png", dpi=160)
    plt.close(fig)

    # summary curves
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
    sorted_err = np.sort(per_vertex)
    ax.plot(sorted_err, color="#55A868", linewidth=1.5)
    ax.set_xlabel("Vertex rank")
    ax.set_ylabel("L2 error")
    ax.set_title("Sorted per-vertex error curve")
    fig.tight_layout()
    fig.savefig(OUT / "charts" / "lbs_error_curve.png", dpi=160)
    plt.close(fig)

    summary_path = OUT / "summary.txt"
    lines = [
        "SMPL LBS Experiment Summary",
        "===========================",
        f"Vertices: {info['vertices']}",
        f"Faces: {info['faces']}",
        f"Joints: {info['joints']}",
        f"Betas dim: {info['betas_dim']}",
        f"Model: {info['model_path']}",
        "",
        "Hand-written LBS vs official smplx forward:",
        f"  Mean absolute error (MAE): {mae:.6e}",
        f"  Max absolute error:        {max_err:.6e}",
        "",
        "Parameters used:",
        f"  betas[0,:3] = {BETAS[0,:3].tolist()}",
        f"  global_orient = {GLOBAL_ORIENT[0].tolist()}",
        f"  body_pose nonzero joints: right_shoulder, right_elbow, left_shoulder, spine",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    (OUT / "charts" / "metrics.json").write_text(
        json.dumps({"mae": mae, "max_error": max_err}, indent=2), encoding="utf-8"
    )
    print("=== Task 7: Verification ===")
    print(f"  MAE:  {mae:.6e}")
    print(f"  MAX:  {max_err:.6e}")


def export_pose_animation(model: smplx.SMPL, frames: int = 48) -> None:
    """Optional: rotate right elbow from 0 to target angle."""
    device = torch.device("cpu")
    model = model.to(device)
    faces = np.asarray(model.faces)
    frame_dir = OUT / "anim_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    betas = BETAS.to(device)
    global_orient = GLOBAL_ORIENT.to(device)
    for i in range(frames):
        t = i / max(frames - 1, 1)
        body_pose = BODY_POSE.clone().to(device)
        body_pose[0, 19 * 3 + 2] = -1.4 * t
        body_pose[0, 17 * 3 + 2] = -1.2 * t
        with torch.no_grad():
            out = model(betas=betas, global_orient=global_orient, body_pose=body_pose)
        v = out.vertices[0].cpu().numpy()
        fp = frame_dir / f"frame_{i:03d}.png"
        render_mesh(v, faces, fp, title=f"Pose animation t={t:.2f}")
        paths.append(fp)

    gif_path = ASSETS / "pose_animation.gif"
    images = [imageio.imread(p) for p in paths]
    imageio.mimsave(gif_path, images, duration=0.08, loop=0)
    print(f"Animation saved: {gif_path}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "charts").mkdir(exist_ok=True)
    ASSETS.mkdir(exist_ok=True)

    model = load_model()
    info = task1_summary(model)
    stages, official = run_stages(model)
    save_visualizations(model, stages, official)
    task7_verify(stages, official, info)
    export_pose_animation(model)
    print("All outputs written to", OUT)


if __name__ == "__main__":
    main()
