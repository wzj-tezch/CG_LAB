from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import taichi as ti
from PIL import Image, ImageDraw

try:
    from .config import SOLVER_EXPLICIT, SOLVER_IMPLICIT, SOLVER_NAME, SOLVER_SEMI_IMPLICIT, SimConfig, output_root
    from .simulation import ClothSimulation
except ImportError:
    from config import SOLVER_EXPLICIT, SOLVER_IMPLICIT, SOLVER_NAME, SOLVER_SEMI_IMPLICIT, SimConfig, output_root
    from simulation import ClothSimulation


def save_gif(frames: list[np.ndarray], path: Path, fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, duration=1.0 / fps)


def save_mp4(frames: list[np.ndarray], path: Path, fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=1)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def look_at_basis(camera_pos: np.ndarray, camera_target: np.ndarray, world_up: np.ndarray):
    forward = camera_target - camera_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)
    return right, up, forward


def project_points(points: np.ndarray, width: int, height: int, cam_pos: np.ndarray, cam_target: np.ndarray):
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right, up, forward = look_at_basis(cam_pos, cam_target, world_up)

    rel = points - cam_pos[None, :]
    cx = rel @ right
    cy = rel @ up
    cz = rel @ forward

    fov_y = np.deg2rad(45.0)
    inv_tan = 1.0 / np.tan(fov_y * 0.5)
    aspect = width / max(height, 1)

    ndc_x = (cx * inv_tan / aspect) / (cz + 1e-8)
    ndc_y = (cy * inv_tan) / (cz + 1e-8)

    px = (ndc_x * 0.5 + 0.5) * width
    py = (0.5 - ndc_y * 0.5) * height
    return np.stack([px, py], axis=1), cz, forward


def shade_color(base_color: np.ndarray, normal: np.ndarray, light_dirs: list[np.ndarray]):
    n = normal / (np.linalg.norm(normal) + 1e-8)
    ambient = 0.26
    diffuse = 0.0
    for ld in light_dirs:
        diffuse += max(float(np.dot(n, ld)), 0.0)
    intensity = np.clip(ambient + 0.55 * diffuse, 0.0, 1.0)
    c = np.clip(base_color * intensity, 0.0, 255.0).astype(np.uint8)
    return int(c[0]), int(c[1]), int(c[2])


def draw_frame(
    points: np.ndarray,
    tri_indices: np.ndarray,
    line_indices: np.ndarray,
    width: int,
    height: int,
    sphere_center: np.ndarray,
    sphere_radius: float,
    cam_pos: np.ndarray,
    cam_target: np.ndarray,
    show_wireframe: bool,
):
    screen_xy, depth, view_dir = project_points(points, width, height, cam_pos=cam_pos, cam_target=cam_target)
    sphere_xy, sphere_depth, _ = project_points(sphere_center[None, :], width, height, cam_pos=cam_pos, cam_target=cam_target)

    img = Image.new("RGB", (width, height), color=(216, 224, 236))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, height * 0.68, width, height), fill=(204, 211, 220))

    valid = depth > 1e-4
    tri_depth = []
    for tri in tri_indices.reshape(-1, 3):
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        if valid[i] and valid[j] and valid[k]:
            avg_depth = float((depth[i] + depth[j] + depth[k]) / 3.0)
            tri_depth.append((avg_depth, i, j, k))
    tri_depth.sort(reverse=True)

    base_color = np.array([184, 141, 225], dtype=np.float32)
    light_dirs = [
        np.array([0.6, 0.8, 0.3], dtype=np.float32),
        np.array([-0.5, 0.6, -0.4], dtype=np.float32),
    ]
    light_dirs = [d / (np.linalg.norm(d) + 1e-8) for d in light_dirs]

    if sphere_depth[0] > 1e-4:
        scale = max(8.0, 130.0 / sphere_depth[0])
        sr = sphere_radius * scale * 1.3
        sx, sy = sphere_xy[0]
        draw.ellipse((sx - sr, sy - sr, sx + sr, sy + sr), fill=(247, 118, 88), outline=(185, 78, 56), width=2)
        draw.ellipse((sx - 0.35 * sr, sy - 0.35 * sr, sx + 0.05 * sr, sy + 0.05 * sr), fill=(255, 177, 150))

    for _, i, j, k in tri_depth:
        p0 = points[i]
        p1 = points[j]
        p2 = points[k]
        normal = np.cross(p1 - p0, p2 - p0)
        if np.dot(normal, view_dir) > 0:
            normal = -normal
        color = shade_color(base_color, normal, light_dirs)
        poly = [tuple(screen_xy[i]), tuple(screen_xy[j]), tuple(screen_xy[k])]
        draw.polygon(poly, fill=color)

    if show_wireframe:
        edge_depth = []
        for e in line_indices.reshape(-1, 2):
            i, j = int(e[0]), int(e[1])
            if valid[i] and valid[j]:
                edge_depth.append((0.5 * (depth[i] + depth[j]), i, j))
        edge_depth.sort(reverse=True)
        for _, i, j in edge_depth:
            p0 = tuple(screen_xy[i].tolist())
            p1 = tuple(screen_xy[j].tolist())
            draw.line((p0, p1), fill=(48, 36, 80), width=1)

    return np.array(img, dtype=np.uint8)


def render_case(
    case_name: str,
    cfg: SimConfig,
    solver: int,
    damping: float,
    frame_count: int,
    warmup_frames: int,
    sample_stride: int,
    fps: int,
    shear_enabled: bool,
    bending_enabled: bool,
    collision_enabled: bool,
    show_wireframe: bool,
):
    sim = ClothSimulation(cfg)
    sim.set_runtime_params(
        damping=damping,
        spring_damping=12.0,
        air_drag=0.08,
        constraint_iters=4,
        constraint_stiffness=0.65,
        friction=0.2,
        restitution=0.06,
        wind_strength=cfg.wind_strength,
        wind_frequency=cfg.wind_frequency,
    )
    sim.set_feature_toggles(shear=shear_enabled, bending=bending_enabled, collision=collision_enabled)

    tri_indices = sim.triangle_indices.to_numpy()
    line_indices = sim.line_indices.to_numpy()[: sim.spring_count[None] * 2]
    frames: list[np.ndarray] = []
    total_steps = frame_count + warmup_frames
    for frame_id in range(total_steps):
        for _ in range(sim.substeps[None]):
            sim.substep(solver)

        if frame_id >= warmup_frames and ((frame_id - warmup_frames) % sample_stride == 0):
            positions = sim.x.to_numpy()
            t = (frame_id - warmup_frames) * 0.03
            cam_pos = np.array([2.1 * np.sin(t), 0.82, 2.1 * np.cos(t)], dtype=np.float32)
            cam_target = np.array([0.0, 0.18, 0.0], dtype=np.float32)
            frames.append(
                draw_frame(
                    points=positions,
                    tri_indices=tri_indices,
                    line_indices=line_indices,
                    width=cfg.width,
                    height=cfg.height,
                    sphere_center=sim.sphere_center.to_numpy()[()],
                    sphere_radius=float(sim.sphere_radius[None]),
                    cam_pos=cam_pos,
                    cam_target=cam_target,
                    show_wireframe=show_wireframe,
                )
            )

    gif_path = output_root() / "gifs" / f"{case_name}.gif"
    mp4_path = output_root() / "mp4" / f"{case_name}.mp4"
    save_gif(frames, gif_path, fps=fps)
    save_mp4(frames, mp4_path, fps=fps)
    print(f"[OK] {case_name}: {gif_path}")
    print(f"[OK] {case_name}: {mp4_path}")


def main():
    parser = argparse.ArgumentParser(description="Export comparison GIFs for Taichi cloth simulation.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--frame-count", type=int, default=240)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-wireframe", action="store_true")
    args = parser.parse_args()

    cfg = SimConfig(
        width=args.width,
        height=args.height,
        substeps=8,
        dt=1.0 / 300.0,
        arch=ti.cpu,
        wind_strength=6.0,
        wind_frequency=0.9,
    )
    ti.init(arch=cfg.arch)

    solver_cases = [
        ("explicit", SOLVER_EXPLICIT),
        ("semi_implicit", SOLVER_SEMI_IMPLICIT),
        ("implicit", SOLVER_IMPLICIT),
    ]

    for damping in [1.0, 5.0]:
        for name, solver in solver_cases:
            render_case(
                case_name=f"{name}_damping_{damping:.1f}",
                cfg=cfg,
                solver=solver,
                damping=damping,
                frame_count=args.frame_count,
                warmup_frames=args.warmup,
                sample_stride=args.stride,
                fps=args.fps,
                shear_enabled=True,
                bending_enabled=True,
                collision_enabled=True,
                show_wireframe=not args.no_wireframe,
            )

    render_case(
        case_name="optional_without_shear_bending",
        cfg=cfg,
        solver=SOLVER_SEMI_IMPLICIT,
        damping=2.5,
        frame_count=args.frame_count,
        warmup_frames=args.warmup,
        sample_stride=args.stride,
        fps=args.fps,
        shear_enabled=False,
        bending_enabled=False,
        collision_enabled=True,
        show_wireframe=not args.no_wireframe,
    )
    render_case(
        case_name="optional_with_shear_bending",
        cfg=cfg,
        solver=SOLVER_SEMI_IMPLICIT,
        damping=2.5,
        frame_count=args.frame_count,
        warmup_frames=args.warmup,
        sample_stride=args.stride,
        fps=args.fps,
        shear_enabled=True,
        bending_enabled=True,
        collision_enabled=True,
        show_wireframe=not args.no_wireframe,
    )
    render_case(
        case_name="optional_without_collision",
        cfg=cfg,
        solver=SOLVER_SEMI_IMPLICIT,
        damping=2.5,
        frame_count=args.frame_count,
        warmup_frames=args.warmup,
        sample_stride=args.stride,
        fps=args.fps,
        shear_enabled=True,
        bending_enabled=True,
        collision_enabled=False,
        show_wireframe=not args.no_wireframe,
    )
    render_case(
        case_name="optional_with_collision",
        cfg=cfg,
        solver=SOLVER_SEMI_IMPLICIT,
        damping=2.5,
        frame_count=args.frame_count,
        warmup_frames=args.warmup,
        sample_stride=args.stride,
        fps=args.fps,
        shear_enabled=True,
        bending_enabled=True,
        collision_enabled=True,
        show_wireframe=not args.no_wireframe,
    )

    print("=== Export finished ===")
    print(f"Generated GIF under: {output_root() / 'gifs'}")
    print(f"Generated MP4 under: {output_root() / 'mp4'}")
    for solver_id, label in SOLVER_NAME.items():
        print(f"Solver {solver_id}: {label}")


if __name__ == "__main__":
    main()
