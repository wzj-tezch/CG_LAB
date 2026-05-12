from __future__ import annotations

import argparse

from .config import Work1Config
from .train_silhouette import run_silhouette_training
from .train_textured import run_textured_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Work1: PyTorch3D differentiable rendering experiment")
    parser.add_argument(
        "--mode",
        choices=("silhouette", "textured", "both"),
        default="silhouette",
        help="运行模式：必做剪影 / 选做纹理 / 两者都跑",
    )
    parser.add_argument("--steps", type=int, default=None, help="优化迭代步数")
    parser.add_argument("--image-size", type=int, default=None, help="渲染分辨率")
    parser.add_argument("--num-views", type=int, default=None, help="相机视角数量")
    parser.add_argument("--output-root", type=str, default=None, help="输出目录根路径")
    return parser


def apply_overrides(cfg: Work1Config, args: argparse.Namespace) -> Work1Config:
    if args.steps is not None:
        cfg.steps = args.steps
    if args.image_size is not None:
        cfg.image_size = args.image_size
    if args.num_views is not None:
        cfg.num_views = args.num_views
    if args.output_root is not None:
        cfg.output_root = args.output_root
    return cfg


def run() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = apply_overrides(Work1Config(), args)

    if args.mode in ("silhouette", "both"):
        run_dir = run_silhouette_training(cfg)
        print(f"Silhouette 实验输出目录: {run_dir}")
    if args.mode in ("textured", "both"):
        run_dir = run_textured_training(cfg)
        print(f"Textured 实验输出目录: {run_dir}")


if __name__ == "__main__":
    run()
