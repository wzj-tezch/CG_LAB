"""MVP matrix construction utilities."""

from __future__ import annotations

import math

import numpy as np


def get_model_matrix(angle: float) -> np.ndarray:
    """Return model matrix for rotating around Z axis by angle degrees."""
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    return np.array(
        [
            [cos_a, -sin_a, 0.0, 0.0],
            [sin_a, cos_a, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def get_view_matrix(eye_pos: np.ndarray) -> np.ndarray:
    """Return view matrix by translating world opposite to camera position."""
    view = np.eye(4, dtype=np.float32)
    view[0, 3] = -float(eye_pos[0])
    view[1, 3] = -float(eye_pos[1])
    view[2, 3] = -float(eye_pos[2])
    return view


def get_projection_matrix(
    eye_fov: float,
    aspect_ratio: float,
    z_near: float,
    z_far: float,
) -> np.ndarray:
    """
    Return perspective projection matrix.

    Build M_proj by:
    1) Perspective frustum -> orthographic box (M_persp_to_ortho)
    2) Orthographic projection (M_ortho)
    """
    n = -float(z_near)
    f = -float(z_far)

    fov_rad = math.radians(eye_fov)
    t = math.tan(fov_rad / 2.0) * abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r

    m_persp_to_ortho = np.array(
        [
            [n, 0.0, 0.0, 0.0],
            [0.0, n, 0.0, 0.0],
            [0.0, 0.0, n + f, -n * f],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    m_ortho_translate = np.array(
        [
            [1.0, 0.0, 0.0, -(r + l) / 2.0],
            [0.0, 1.0, 0.0, -(t + b) / 2.0],
            [0.0, 0.0, 1.0, -(n + f) / 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    m_ortho_scale = np.array(
        [
            [2.0 / (r - l), 0.0, 0.0, 0.0],
            [0.0, 2.0 / (t - b), 0.0, 0.0],
            [0.0, 0.0, 2.0 / (n - f), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    m_ortho = m_ortho_scale @ m_ortho_translate
    return m_ortho @ m_persp_to_ortho
