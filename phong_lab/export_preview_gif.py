"""
无窗口渲染多帧并写出 assets/preview.gif，供 README 效果展示。
运行前请确保已安装: pip install imageio numpy taichi
"""
import math
import os
import sys

# 必须在 import phong_raycast 之前设置
os.environ["PHONG_HEADLESS"] = "1"

import numpy as np  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

import phong_raycast as pr  # noqa: E402


def cone_cos2_value() -> float:
    hyp = math.sqrt(pr.CONE_H * pr.CONE_H + pr.CONE_BASE_R * pr.CONE_BASE_R)
    c = pr.CONE_H / hyp
    return float(c * c)


def main() -> int:
    c2 = cone_cos2_value()
    out_dir = os.path.join(os.path.dirname(__file__), "assets")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "preview.gif")

    n = 40
    frames = []
    for k in range(n):
        u = k / n
        # 周期调节材质，突出高光与漫反射变化（便于 GIF 看出交互感）
        shininess = float(6.0 + 100.0 * (0.5 - 0.5 * math.cos(2.0 * math.pi * u)))
        ka = float(0.15 + 0.2 * (0.5 - 0.5 * math.cos(2.0 * math.pi * u)))
        kd = float(0.5 + 0.35 * (0.5 + 0.5 * math.sin(2.0 * math.pi * u + 0.7)))
        ks = float(0.3 + 0.55 * (0.5 - 0.5 * math.cos(2.0 * math.pi * u + 1.2)))
        pr.render(ka, kd, ks, shininess, float(c2))
        img = pr.pixels.to_numpy()
        # Taichi field (W,H,3) -> GIF 常用 (H,W,3)
        img = np.ascontiguousarray(np.transpose(img, (1, 0, 2)))
        frames.append(np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8))

    imageio.mimsave(out_path, frames, duration=0.07, loop=0)
    print(f"已写入: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
