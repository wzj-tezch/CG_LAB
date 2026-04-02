"""
贝塞尔曲线实验：De Casteljau + Taichi GGUI + GPU 光栅化。
"""
import numpy as np
import taichi as ti

WIDTH = 800
HEIGHT = 800
NUM_SEGMENTS = 1000
NUM_CURVE_POINTS = NUM_SEGMENTS + 1
MAX_CONTROL_POINTS = 100
MAX_LINE_VERTICES = 2 * (MAX_CONTROL_POINTS - 1)

try:
    ti.init(arch=ti.gpu)
except Exception:
    ti.init(arch=ti.cpu)

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=(NUM_CURVE_POINTS,))
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=(MAX_CONTROL_POINTS,))
line_vertices = ti.Vector.field(2, dtype=ti.f32, shape=(MAX_LINE_VERTICES,))


@ti.kernel
def clear_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.06, 0.06, 0.08])


@ti.kernel
def draw_curve_kernel(n: ti.i32):
    for i in range(n):
        p = curve_points_field[i]
        ix = int(p[0] * ti.static(WIDTH))
        iy = int(p[1] * ti.static(HEIGHT))
        if 0 <= ix < ti.static(WIDTH) and 0 <= iy < ti.static(HEIGHT):
            pixels[ix, iy] = ti.Vector([0.15, 1.0, 0.25])


def de_casteljau(points: list, t: float) -> np.ndarray:
    """De Casteljau：纯 Python，返回参数 t 处曲线点 [x, y]，坐标为归一化浮点。"""
    if len(points) == 0:
        return np.array([0.0, 0.0], dtype=np.float32)
    pts = [np.array((float(p[0]), float(p[1])), dtype=np.float64) for p in points]
    while len(pts) > 1:
        pts = [(1.0 - t) * pts[k] + t * pts[k + 1] for k in range(len(pts) - 1)]
    return pts[0].astype(np.float32)


def fill_line_vertices(control: list[tuple[float, float]], out: np.ndarray) -> None:
    """控制多边形：线段列表顺序 (P0,P1), (P1,P2), ...，未使用位置放在屏外。"""
    out.fill(-10.0)
    n = len(control)
    if n < 2:
        return
    for i in range(n - 1):
        out[2 * i, 0] = control[i][0]
        out[2 * i, 1] = control[i][1]
        out[2 * i + 1, 0] = control[i + 1][0]
        out[2 * i + 1, 1] = control[i + 1][1]


def main() -> None:
    control: list[tuple[float, float]] = []
    curve_np = np.zeros((NUM_CURVE_POINTS, 2), dtype=np.float32)
    gui_np = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
    line_np = np.full((MAX_LINE_VERTICES, 2), -10.0, dtype=np.float32)

    window = ti.ui.Window("贝塞尔曲线 (De Casteljau)", (WIDTH, HEIGHT), vsync=True)
    canvas = window.get_canvas()
    line_width = 2.0 / float(HEIGHT)

    while window.running:
        clear_pixels()

        while window.get_event(ti.ui.PRESS):
            key = window.event.key
            if key == "c" or key == "C":
                control.clear()
            elif key == ti.ui.LMB and len(control) < MAX_CONTROL_POINTS:
                x, y = window.get_cursor_pos()
                control.append((float(x), float(y)))

        if len(control) >= 2:
            for i in range(NUM_CURVE_POINTS):
                t = i / float(NUM_SEGMENTS)
                curve_np[i] = de_casteljau(control, t)
            curve_points_field.from_numpy(curve_np)
            draw_curve_kernel(NUM_CURVE_POINTS)

        canvas.set_image(pixels)

        fill_line_vertices(control, line_np)
        line_vertices.from_numpy(line_np)
        if len(control) >= 2:
            canvas.lines(
                line_vertices,
                width=line_width,
                color=(0.55, 0.55, 0.58),
            )

        gui_np.fill(-10.0)
        for i in range(len(control)):
            gui_np[i, 0] = control[i][0]
            gui_np[i, 1] = control[i][1]
        gui_points.from_numpy(gui_np)
        canvas.circles(gui_points, radius=5.0, color=(1.0, 0.15, 0.15))

        window.show()


if __name__ == "__main__":
    main()
