"""Taichi GUI MVP experiment: triangle + rotating cube wireframe."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import taichi as ti

from week2.geometry import CUBE_EDGES, CUBE_VERTICES, TRIANGLE_EDGES, TRIANGLE_VERTICES
from week2.mvp import get_model_matrix, get_projection_matrix, get_view_matrix


WINDOW_RES = (700, 700)
EYE_POS = np.array([0.0, 0.0, 5.0], dtype=np.float32)
EYE_FOV = 45.0
Z_NEAR = 0.1
Z_FAR = 50.0
ROTATE_STEP = 3.0
BG_COLOR = 0x112F41

_TI_INITIALIZED = False


def init_taichi() -> None:
    """Initialize Taichi runtime exactly once."""
    global _TI_INITIALIZED
    if _TI_INITIALIZED:
        return
    ti.init(arch=ti.gpu)
    _TI_INITIALIZED = True


def transform_to_screen(vertices: np.ndarray, mvp: np.ndarray) -> np.ndarray:
    """Transform 3D vertices into 2D normalized GUI coordinates."""
    ones = np.ones((vertices.shape[0], 1), dtype=np.float32)
    homogenous_vertices = np.hstack([vertices.astype(np.float32), ones])

    clip_coords = (mvp @ homogenous_vertices.T).T
    # Perspective divide to convert clip space to NDC.
    ndc = clip_coords[:, :3] / clip_coords[:, 3:4]

    screen = np.empty((vertices.shape[0], 2), dtype=np.float32)
    screen[:, 0] = (ndc[:, 0] + 1.0) * 0.5
    screen[:, 1] = (ndc[:, 1] + 1.0) * 0.5
    return screen


def draw_wireframe(gui: ti.GUI, points_2d: np.ndarray, edges: tuple[tuple[int, int], ...], color: int) -> None:
    """Draw line list defined by edge indices."""
    for start, end in edges:
        gui.line(
            begin=(float(points_2d[start, 0]), float(points_2d[start, 1])),
            end=(float(points_2d[end, 0]), float(points_2d[end, 1])),
            radius=2,
            color=color,
        )


def compute_mvp(angle: float) -> np.ndarray:
    """Build MVP matrix from current camera and model angle."""
    model = get_model_matrix(angle)
    view = get_view_matrix(EYE_POS)
    projection = get_projection_matrix(
        eye_fov=EYE_FOV,
        aspect_ratio=WINDOW_RES[0] / WINDOW_RES[1],
        z_near=Z_NEAR,
        z_far=Z_FAR,
    )
    return projection @ view @ model


def render_scene(gui: ti.GUI, angle: float) -> None:
    """Render one frame for the given rotation angle."""
    mvp = compute_mvp(angle)
    tri_points = transform_to_screen(TRIANGLE_VERTICES, mvp)
    cube_points = transform_to_screen(CUBE_VERTICES, mvp)

    gui.clear(BG_COLOR)
    draw_wireframe(gui, tri_points, TRIANGLE_EDGES, color=0xFFD166)
    draw_wireframe(gui, cube_points, CUBE_EDGES, color=0x06D6A0)


def export_frames(output_dir: Path, frame_count: int = 90, angle_step: float = 4.0) -> list[Path]:
    """Render deterministic animation frames to PNG files."""
    init_taichi()
    output_dir.mkdir(parents=True, exist_ok=True)
    gui = ti.GUI("Week2 MVP Export", res=WINDOW_RES, show_gui=False)

    frame_paths: list[Path] = []
    angle = 0.0
    for idx in range(frame_count):
        render_scene(gui, angle)
        frame_path = output_dir / f"frame_{idx:04d}.png"
        gui.show(str(frame_path))
        frame_paths.append(frame_path)
        angle += angle_step
    return frame_paths


def main() -> None:
    init_taichi()
    gui = ti.GUI("Week2 MVP Transform", res=WINDOW_RES)
    angle = 0.0

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                gui.running = False
            elif gui.event.key == "a":
                angle += ROTATE_STEP
            elif gui.event.key == "d":
                angle -= ROTATE_STEP

        render_scene(gui, angle)
        gui.show()


if __name__ == "__main__":
    main()
