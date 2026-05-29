"""Mesh rendering helpers for SMPL LBS experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# SMPL: Y-up, T-pose arms along X, face toward -Z.
# Map (x, y, z)_smpl -> (x, z, y)_plot so the body stands upright in a front view.
FRONT_ELEV = 10.0
FRONT_AZIM = -90.0


def _smpl_to_plot(pts: np.ndarray) -> np.ndarray:
    return pts[:, [0, 2, 1]]


def _set_equal_aspect(ax, pts: np.ndarray, margin: float = 0.08) -> None:
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2 * (1 + margin)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _camera(ax, elev: float = FRONT_ELEV, azim: float = FRONT_AZIM) -> None:
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))


def render_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    out_path: Path,
    *,
    vertex_colors: np.ndarray | None = None,
    joints: np.ndarray | None = None,
    title: str = "",
    elev: float = FRONT_ELEV,
    azim: float = FRONT_AZIM,
    wireframe: bool = False,
    cmap: str = "viridis",
    dpi: int = 160,
) -> None:
    fig = plt.figure(figsize=(6, 7), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    v = _smpl_to_plot(vertices.copy())
    f = faces
    if vertex_colors is None:
        face_colors = (0.82, 0.70, 0.58, 0.95)
        coll = Poly3DCollection(v[f], linewidths=0.05, edgecolors=(0.3, 0.3, 0.3, 0.15))
        coll.set_facecolor(face_colors)
    else:
        vc = vertex_colors
        if vc.ndim == 1:
            norm = Normalize(vmin=float(vc.min()), vmax=float(vc.max()))
            rgba = cm.get_cmap(cmap)(norm(vc))
        else:
            rgba = vc
        coll = Poly3DCollection(v[f], linewidths=0.03, edgecolors=(0, 0, 0, 0.05))
        coll.set_facecolor(rgba[f].mean(axis=1))

    ax.add_collection3d(coll)

    if wireframe:
        mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        edges = mesh.edges_unique
        for e in edges[:: max(1, len(edges) // 6000)]:
            ax.plot(v[e, 0], v[e, 1], v[e, 2], color=(0.2, 0.2, 0.2, 0.25), linewidth=0.2)

    if joints is not None:
        j = _smpl_to_plot(joints)
        ax.scatter(j[:, 0], j[:, 1], j[:, 2], c="crimson", s=10, depthshade=True)

    pts = v if joints is None else np.vstack([v, _smpl_to_plot(joints)])
    _set_equal_aspect(ax, pts)
    _camera(ax, elev=elev, azim=azim)
    if title:
        ax.set_title(title, fontsize=11, pad=8)
    ax.grid(False)
    ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_dominant_joint_map(
    vertices: np.ndarray,
    faces: np.ndarray,
    weights: np.ndarray,
    out_path: Path,
    title: str = "All Joint LBS Weights",
) -> None:
    dominant = weights.argmax(axis=1)
    strength = weights.max(axis=1)
    n_joints = weights.shape[1]
    base = cm.get_cmap("tab20")(np.linspace(0, 1, n_joints))
    colors = base[dominant]
    colors = colors * strength[:, None] + (1 - strength)[:, None] * 0.85
    colors[:, 3] = 1.0
    render_mesh(vertices, faces, out_path, vertex_colors=colors, title=title)


def make_comparison_grid(images: dict[str, Path], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), facecolor="white")
    order = [
        ("(a) template + weights", images["a"]),
        ("(b) shape + joints", images["b"]),
        ("(c) pose offsets", images["c"]),
        ("(d) final skinned mesh", images["d"]),
    ]
    for ax, (title, img_path) in zip(axes.ravel(), order):
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
