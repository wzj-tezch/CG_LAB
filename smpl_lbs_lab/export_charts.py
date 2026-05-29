"""Export supplementary data charts for the SMPL LBS report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import smplx
import torch

from manual_lbs import manual_lbs
from setup_model import ensure_model

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "charts"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model_path, _ = ensure_model()
    model = smplx.create(str(model_path), model_type="smpl", gender="neutral", batch_size=1)

    betas = torch.tensor([[1.6, 0.9, -0.6, 0.3, -0.2, 0, 0, 0, 0, 0]], dtype=torch.float32)
    global_orient = torch.tensor([[0.0, 0.15, 0.0]], dtype=torch.float32)
    body_pose = torch.zeros(1, 69)
    body_pose[0, 17 * 3 + 2] = -1.2
    body_pose[0, 19 * 3 + 2] = -1.4
    full_pose = torch.cat([global_orient, body_pose], dim=1)

    stages = manual_lbs(
        betas, full_pose,
        model.v_template.unsqueeze(0), model.shapedirs, model.posedirs,
        model.J_regressor, model.parents, model.lbs_weights,
    )

    v0 = stages.v_template[0].numpy()
    vs = stages.v_shaped[0].numpy()
    vp = stages.v_posed[0].numpy()
    vf = stages.verts[0].numpy()
    pose_off = stages.pose_offsets[0].numpy()
    w = model.lbs_weights.cpu().numpy()

    shape_disp = np.linalg.norm(vs - v0, axis=1)
    pose_mag = np.linalg.norm(pose_off, axis=1)
    lbs_disp = np.linalg.norm(vf - vp, axis=1)

    stats = {
        "shape_disp_mean_m": float(shape_disp.mean()),
        "shape_disp_max_m": float(shape_disp.max()),
        "pose_offset_mean_m": float(pose_mag.mean()),
        "pose_offset_max_m": float(pose_mag.max()),
        "lbs_disp_mean_m": float(lbs_disp.mean()),
        "lbs_disp_max_m": float(lbs_disp.max()),
        "height_template_m": float(v0[:, 1].max() - v0[:, 1].min()),
        "height_shaped_m": float(vs[:, 1].max() - vs[:, 1].min()),
    }
    (OUT / "stage_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # 1) shape vs pose vs lbs displacement comparison
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), facecolor="white")
    for ax, data, title, color in zip(
        axes,
        [shape_disp * 1000, pose_mag * 1000, lbs_disp * 1000],
        ["|v_shaped - v_template| (mm)", "|pose_offsets| (mm)", "|verts - v_posed| (mm)"],
        ["#4C72B0", "#DD8452", "#55A868"],
    ):
        ax.hist(data, bins=60, color=color, edgecolor="white", alpha=0.9)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("mm")
        ax.set_ylabel("vertices")
    fig.suptitle("Per-vertex displacement statistics across LBS stages", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "displacement_comparison.png", dpi=160)
    plt.close(fig)

    # 2) joint-17 weight distribution
    w17 = w[:, 17]
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    ax.hist(w17, bins=50, color="#C44E52", edgecolor="white", alpha=0.9)
    ax.set_xlabel("LBS weight (joint 17: right shoulder)")
    ax.set_ylabel("vertex count")
    ax.set_title("Single-joint weight distribution")
    ax.axvline(w17.mean(), color="black", linestyle="--", label=f"mean={w17.mean():.4f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "joint17_weight_hist.png", dpi=160)
    plt.close(fig)

    # 3) pipeline bar chart
    labels = ["template→shaped", "shaped→posed", "posed→skinned"]
    means = [stats["shape_disp_mean_m"] * 1000, stats["pose_offset_mean_m"] * 1000, stats["lbs_disp_mean_m"] * 1000]
    maxs = [stats["shape_disp_max_m"] * 1000, stats["pose_offset_max_m"] * 1000, stats["lbs_disp_max_m"] * 1000]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
    ax.bar(x - width / 2, means, width, label="mean (mm)", color="#4C72B0")
    ax.bar(x + width / 2, maxs, width, label="max (mm)", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("displacement (mm)")
    ax.set_title("Stage-wise vertex displacement summary")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "pipeline_displacement_bar.png", dpi=160)
    plt.close(fig)

    # 4) beta height effect
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    ax.bar(["template", "shaped"], [stats["height_template_m"] * 100, stats["height_shaped_m"] * 100],
           color=["#8172B2", "#937860"], width=0.5)
    ax.set_ylabel("body height (cm, Y extent)")
    ax.set_title("Effect of shape parameters on body height")
    for i, v in enumerate([stats["height_template_m"] * 100, stats["height_shaped_m"] * 100]):
        ax.text(i, v + 0.5, f"{v:.1f} cm", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "shape_height_bar.png", dpi=160)
    plt.close(fig)

    print("charts saved to", OUT)


if __name__ == "__main__":
    main()
