"""
生成选做功能演示动图（反走样对比、贝塞尔 vs 均匀三次 B 样条）。
依赖：pillow、numpy。运行：python export_optional_gifs.py
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from curve_tools import (
    raster_aliased,
    raster_antialiased,
    sample_bezier,
    sample_uniform_cubic_bspline,
)

W = H = 480
NUM = 600
BG = (0.06, 0.06, 0.08)
GREEN = (0.15, 1.0, 0.25)


def load_font(size: int):
    import os as _os

    for name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf"):
        p = _os.path.join(_os.environ.get("WINDIR", "C:\\Windows"), "Fonts", name)
        if _os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


def np_to_pil(arr: np.ndarray) -> Image.Image:
    """(W,H,3) float [0,1] -> PIL RGB"""
    x = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(np.transpose(x, (1, 0, 2)), mode="RGB")


def draw_overlay(img: Image.Image, text: str, pos: tuple[int, int], font) -> None:
    d = ImageDraw.Draw(img)
    d.text(pos, text, fill=(240, 240, 245), font=font, stroke_width=2, stroke_fill=(0, 0, 0))


def export_aa_gif(out_path: str) -> None:
    font = load_font(18)
    control = [
        (0.12, 0.52),
        (0.32, 0.88),
        (0.68, 0.82),
        (0.88, 0.18),
    ]
    curve = sample_bezier(control, NUM)
    ali = raster_aliased(curve, W, H, BG, GREEN)
    aa = raster_antialiased(curve, W, H, BG, GREEN)

    gap = 8
    tw = W * 2 + gap
    th = H
    frames: list[Image.Image] = []
    durations: list[int] = []

    def compose(left: np.ndarray, right: np.ndarray) -> Image.Image:
        pl = np_to_pil(left)
        pr = np_to_pil(right)
        canvas = Image.new("RGB", (tw, th), (10, 10, 14))
        canvas.paste(pl, (0, 0))
        canvas.paste(pr, (W + gap, 0))
        draw_overlay(canvas, "光栅化（走样）", (12, 10), font)
        draw_overlay(canvas, "3×3 距离反走样", (W + gap + 12, 10), font)
        return canvas

    base = compose(ali, aa)
    for _ in range(25):
        frames.append(base.copy())
        durations.append(80)
    for _ in range(8):
        frames.append(base.copy())
        durations.append(200)

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print("Wrote", out_path)


def export_bspline_gif(out_path: str) -> None:
    font = load_font(17)
    # 6 点：B 样条有 3 段，与贝塞尔全局形状差异明显
    control = [
        (0.08, 0.45),
        (0.22, 0.88),
        (0.45, 0.72),
        (0.62, 0.88),
        (0.78, 0.25),
        (0.92, 0.55),
    ]
    bz = sample_bezier(control, NUM)
    bs = sample_uniform_cubic_bspline(control, NUM)

    frames: list[Image.Image] = []
    durations: list[int] = []

    def frame_from_curve(curve: np.ndarray, label: str) -> Image.Image:
        arr = raster_aliased(curve, W, H, BG, GREEN)
        im = np_to_pil(arr)
        draw_overlay(im, label, (10, 8), font)
        draw_overlay(im, "按 B 切换模式（程序内）", (10, H - 32), load_font(15))
        return im

    fb = frame_from_curve(bz, "贝塞尔曲线（全局控制）")
    fbs = frame_from_curve(bs, "均匀三次 B 样条（局部控制）")

    for _ in range(22):
        frames.append(fb.copy())
        durations.append(90)
    for _ in range(22):
        frames.append(fbs.copy())
        durations.append(90)

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print("Wrote", out_path)


def main() -> None:
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    os.makedirs(d, exist_ok=True)
    export_aa_gif(os.path.join(d, "optional_antialiasing.gif"))
    export_bspline_gif(os.path.join(d, "optional_bspline.gif"))


if __name__ == "__main__":
    main()
