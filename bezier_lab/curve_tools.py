"""曲线采样与 CPU 端反走样光栅化（主程序与导出动图共用）。"""
import numpy as np


def de_casteljau(points: list, t: float) -> np.ndarray:
    if len(points) == 0:
        return np.array([0.0, 0.0], dtype=np.float32)
    pts = [np.array((float(p[0]), float(p[1])), dtype=np.float64) for p in points]
    while len(pts) > 1:
        pts = [(1.0 - t) * pts[k] + t * pts[k + 1] for k in range(len(pts) - 1)]
    return pts[0].astype(np.float32)


def sample_bezier(control: list, num: int) -> np.ndarray:
    out = np.zeros((num, 2), dtype=np.float32)
    if len(control) < 2:
        return out
    denom = num - 1 if num > 1 else 1
    for i in range(num):
        t = i / denom
        out[i] = de_casteljau(control, t)
    return out


def _bspline_segment_blends(u: float) -> tuple[float, float, float, float]:
    t = float(np.clip(u, 0.0, 1.0))
    b0 = (1.0 - t) ** 3 / 6.0
    b1 = (3.0 * t**3 - 6.0 * t**2 + 4.0) / 6.0
    b2 = (-3.0 * t**3 + 3.0 * t**2 + 3.0 * t + 1.0) / 6.0
    b3 = t**3 / 6.0
    return b0, b1, b2, b3


def eval_bspline_segment(
    p0: tuple, p1: tuple, p2: tuple, p3: tuple, u: float
) -> np.ndarray:
    b0, b1, b2, b3 = _bspline_segment_blends(u)
    v0 = np.array(p0, dtype=np.float64)
    v1 = np.array(p1, dtype=np.float64)
    v2 = np.array(p2, dtype=np.float64)
    v3 = np.array(p3, dtype=np.float64)
    return (b0 * v0 + b1 * v1 + b2 * v2 + b3 * v3).astype(np.float32)


def sample_uniform_cubic_bspline(control: list, num: int) -> np.ndarray:
    """均匀三次 B 样条；控制点 n>=4，共 n-3 段，全局参数映射到段内 u。"""
    out = np.zeros((num, 2), dtype=np.float32)
    n = len(control)
    if n < 4:
        return sample_bezier(control, num)
    n_seg = n - 3
    denom = num - 1 if num > 1 else 1
    for k in range(num):
        t_global = (k / denom) * n_seg
        if t_global >= n_seg:
            seg = n_seg - 1
            u = 1.0
        else:
            seg = int(np.floor(t_global))
            u = t_global - seg
        p0, p1, p2, p3 = control[seg : seg + 4]
        out[k] = eval_bspline_segment(p0, p1, p2, p3, u)
    return out


def raster_aliased(
    curve: np.ndarray,
    width: int,
    height: int,
    bg: tuple[float, float, float],
    color: tuple[float, float, float],
) -> np.ndarray:
    img = np.zeros((width, height, 3), dtype=np.float32)
    img[:, :] = np.array(bg, dtype=np.float32)
    for i in range(curve.shape[0]):
        fx = float(curve[i, 0]) * width
        fy = float(curve[i, 1]) * height
        ix = int(fx)
        iy = int(fy)
        if 0 <= ix < width and 0 <= iy < height:
            img[ix, iy] = color
    return img


def raster_antialiased(
    curve: np.ndarray,
    width: int,
    height: int,
    bg: tuple[float, float, float],
    color: tuple[float, float, float],
    falloff: float = 1.25,
    gain: float = 0.35,
) -> np.ndarray:
    """
    3×3 邻域 + 到像素中心距离线性衰减，累加到背景上（简单反走样）。
    curve: 归一化 [0,1]²
    """
    img = np.zeros((width, height, 3), dtype=np.float32)
    img[:, :] = np.array(bg, dtype=np.float32)
    c = np.array(color, dtype=np.float32)
    for i in range(curve.shape[0]):
        fx = float(curve[i, 0]) * width
        fy = float(curve[i, 1]) * height
        ix = int(fx)
        iy = int(fy)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                px, py = ix + dx, iy + dy
                if not (0 <= px < width and 0 <= py < height):
                    continue
                cx = px + 0.5
                cy = py + 0.5
                d = float(np.hypot(fx - cx, fy - cy))
                wgt = max(0.0, 1.0 - d / falloff)
                img[px, py] += c * (wgt * gain)
    return np.clip(img, 0.0, 1.0)
