"""
贝塞尔曲线实验：De Casteljau + Taichi GGUI + GPU 光栅化。
选做：A 键开关 3×3 距离反走样（CPU 批量写入帧缓冲）；B 键在贝塞尔 / 均匀三次 B 样条之间切换。
"""
import numpy as np
import taichi as ti

from curve_tools import raster_antialiased, sample_bezier, sample_uniform_cubic_bspline

WIDTH = 800
HEIGHT = 800
NUM_SEGMENTS = 1000
NUM_CURVE_POINTS = NUM_SEGMENTS + 1
MAX_CONTROL_POINTS = 100
MAX_LINE_VERTICES = 2 * (MAX_CONTROL_POINTS - 1)
BG = (0.06, 0.06, 0.08)
CURVE_RGB = (0.15, 1.0, 0.25)

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


def fill_line_vertices(control: list[tuple[float, float]], out: np.ndarray) -> None:
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
    use_aa = False
    use_bspline = False
    curve_np = np.zeros((NUM_CURVE_POINTS, 2), dtype=np.float32)
    gui_np = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
    line_np = np.full((MAX_LINE_VERTICES, 2), -10.0, dtype=np.float32)

    window = ti.ui.Window(
        "贝塞尔/B样条 | A 反走样  B 切换曲线模式  C 清空",
        (WIDTH, HEIGHT),
        vsync=True,
    )
    canvas = window.get_canvas()
    line_width = 2.0 / float(HEIGHT)

    while window.running:
        while window.get_event(ti.ui.PRESS):
            key = window.event.key
            if key == "c" or key == "C":
                control.clear()
            elif key == "a" or key == "A":
                use_aa = not use_aa
            elif key == "b" or key == "B":
                use_bspline = not use_bspline
            elif key == ti.ui.LMB and len(control) < MAX_CONTROL_POINTS:
                x, y = window.get_cursor_pos()
                control.append((float(x), float(y)))

        if len(control) >= 2:
            if use_bspline:
                curve_np = sample_uniform_cubic_bspline(control, NUM_CURVE_POINTS)
            else:
                curve_np = sample_bezier(control, NUM_CURVE_POINTS)

        if use_aa:
            if len(control) >= 2:
                frame = raster_antialiased(
                    curve_np, WIDTH, HEIGHT, BG, CURVE_RGB
                )
            else:
                frame = np.empty((WIDTH, HEIGHT, 3), dtype=np.float32)
                frame[:, :] = BG
            pixels.from_numpy(frame)
        else:
            clear_pixels()
            if len(control) >= 2:
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
