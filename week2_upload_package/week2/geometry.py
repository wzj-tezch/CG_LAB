"""Geometry definitions for Week2 MVP experiment."""

from __future__ import annotations

import numpy as np


TRIANGLE_VERTICES = np.array(
    [
        [2.0, 0.0, -2.0],
        [0.0, 2.0, -2.0],
        [-2.0, 0.0, -2.0],
    ],
    dtype=np.float32,
)

TRIANGLE_EDGES = (
    (0, 1),
    (1, 2),
    (2, 0),
)

CUBE_VERTICES = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)

CUBE_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)
