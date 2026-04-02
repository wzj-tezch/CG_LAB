"""
生成与实验「参考效果」一致的静态示意图（深底、灰控制多边形、绿色光栅化曲线、红控制点）。
飞书文档通常无法外链预览，本图按课程说明绘制。依赖：pillow。

运行：python export_reference_figure.py
输出：assets/reference_effect.png
"""
import os

from PIL import Image, ImageDraw, ImageFont


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


def load_cn_font(size: int):
    candidates = [
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "msyh.ttc"),
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "msyhbd.ttc"),
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "simhei.ttf"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "assets")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "reference_effect.png")

    img_w, img_h = 1280, 800
    ox, oy = 80, 110
    iw, ih = img_w - ox - 80, img_h - oy - 100

    img = Image.new("RGB", (img_w, img_h), (22, 22, 28))
    draw = ImageDraw.Draw(img)

    title_font = load_cn_font(30)
    small_font = load_cn_font(20)
    tiny_font = load_cn_font(16)

    draw.text((ox, 36), "贝塞尔曲线 · 参考效果（De Casteljau + 光栅化）", fill=(230, 230, 235), font=title_font)

    def to_px(x: float, y: float) -> tuple[int, int]:
        px = ox + int(x * iw)
        py = oy + int((1.0 - y) * ih)
        return px, py

    control = [
        (0.10, 0.50),
        (0.30, 0.90),
        (0.70, 0.85),
        (0.90, 0.15),
    ]

    draw.rectangle([ox, oy, ox + iw - 1, oy + ih - 1], outline=(55, 55, 65), width=1)

    if len(control) >= 2:
        for i in range(len(control) - 1):
            a, b = to_px(*control[i]), to_px(*control[i + 1])
            draw.line([a, b], fill=(130, 130, 138), width=3)

    num_seg = 2000
    green = (35, 255, 70)
    for i in range(num_seg + 1):
        t = i / num_seg
        x, y = de_casteljau(control, t)
        px, py = to_px(x, y)
        if ox <= px < ox + iw and oy <= py < oy + ih:
            draw.rectangle([px - 1, py - 1, px + 1, py + 1], fill=green)

    r = 9
    for p in control:
        cx, cy = to_px(*p)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 55, 55), outline=(120, 30, 30), width=2)

    lx, ly = ox + 12, oy + ih - 78
    draw.rounded_rectangle([lx - 8, ly - 8, lx + 460, ly + 72], radius=6, fill=(35, 35, 42), outline=(70, 70, 80))
    draw.ellipse([lx, ly + 4, lx + 14, ly + 18], fill=(255, 55, 55))
    draw.text((lx + 22, ly), "控制点（鼠标左键添加）", fill=(210, 210, 215), font=tiny_font)
    draw.line([lx, ly + 32, lx + 36, ly + 32], fill=(130, 130, 138), width=3)
    draw.text((lx + 44, ly + 22), "控制多边形", fill=(210, 210, 215), font=tiny_font)
    draw.rectangle([lx, ly + 44, lx + 14, ly + 58], fill=green)
    draw.text((lx + 22, ly + 42), "贝塞尔曲线（GPU 并行光栅化）", fill=(210, 210, 215), font=tiny_font)

    draw.text((ox + iw - 280, oy + 12), "键盘 C：清空画布", fill=(160, 160, 170), font=small_font)

    img.save(out_path, "PNG", optimize=True)
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
