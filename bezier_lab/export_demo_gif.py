"""
生成 README 用演示动图（不依赖 Taichi 窗口，仅需 pillow + numpy）。
运行：python export_demo_gif.py
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw

W = H = 520
NUM_SAMPLES = 400


def de_casteljau(points: list[tuple[float, float]], t: float) -> tuple[float, float]:
    if not points:
        return (0.0, 0.0)
    pts = [(float(p[0]), float(p[1])) for p in points]
    while len(pts) > 1:
        pts = [
            ((1.0 - t) * pts[i][0] + t * pts[i + 1][0], (1.0 - t) * pts[i][1] + t * pts[i + 1][1])
            for i in range(len(pts) - 1)
        ]
    return pts[0]


def to_px(x: float, y: float) -> tuple[int, int]:
    return int(x * W), int((1.0 - y) * H)


def draw_frame(control: list[tuple[float, float]]) -> Image.Image:
    img = Image.new("RGB", (W, H), (16, 16, 20))
    draw = ImageDraw.Draw(img)

    if len(control) >= 2:
        for i in range(len(control) - 1):
            a, b = to_px(*control[i]), to_px(*control[i + 1])
            draw.line([a, b], fill=(140, 140, 148), width=2)

    if len(control) >= 2:
        for i in range(NUM_SAMPLES + 1):
            t = i / NUM_SAMPLES
            x, y = de_casteljau(control, t)
            px, py = to_px(x, y)
            if 0 <= px < W and 0 <= py < H:
                draw.point((px, py), fill=(40, 255, 65))

    for p in control:
        cx, cy = to_px(*p)
        r = 6
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 40, 40), outline=(80, 20, 20))

    return img


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "assets")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bezier_demo.gif")

    # 依次「点击」添加控制点，与实验说明一致
    keyframes = [
        [(0.12, 0.55), (0.35, 0.88)],
        [(0.12, 0.55), (0.35, 0.88), (0.72, 0.82)],
        [(0.12, 0.55), (0.35, 0.88), (0.72, 0.82), (0.88, 0.22)],
    ]

    frames: list[Image.Image] = []
    durations: list[int] = []
    for ctrl in keyframes:
        for _ in range(18):
            frames.append(draw_frame(ctrl))
            durations.append(55)
        for _ in range(12):
            frames.append(draw_frame(ctrl))
            durations.append(120)

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
