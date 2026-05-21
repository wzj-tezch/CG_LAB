from __future__ import annotations

import argparse

import taichi as ti

try:
    from .config import SOLVER_EXPLICIT, SOLVER_IMPLICIT, SOLVER_NAME, SOLVER_SEMI_IMPLICIT, SimConfig
    from .simulation import ClothSimulation
except ImportError:
    from config import SOLVER_EXPLICIT, SOLVER_IMPLICIT, SOLVER_NAME, SOLVER_SEMI_IMPLICIT, SimConfig
    from simulation import ClothSimulation


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Taichi cloth simulation with multiple integrators.")
    parser.add_argument("--solver", type=int, default=SOLVER_SEMI_IMPLICIT, choices=[0, 1, 2])
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--substeps", type=int, default=6)
    parser.add_argument("--grid", type=int, default=20)
    parser.add_argument("--dt", type=float, default=1.0 / 240.0)
    parser.add_argument("--wind", type=float, default=5.0)
    return parser


def run_app(cfg: SimConfig, init_solver: int):
    try:
        ti.init(arch=cfg.arch)
    except Exception:
        ti.init(arch=ti.cpu)

    sim = ClothSimulation(cfg)
    solver_mode = init_solver
    paused = False

    shear_enabled = True
    bending_enabled = True
    collision_enabled = True

    damping = cfg.damping
    ks_structural = cfg.ks_structural
    ks_shear = cfg.ks_shear
    ks_bending = cfg.ks_bending
    max_velocity = cfg.max_velocity
    implicit_iters = cfg.implicit_iters
    constraint_iters = cfg.constraint_iters
    constraint_stiffness = cfg.constraint_stiffness
    spring_damping = cfg.spring_damping
    air_drag = cfg.air_drag
    restitution = cfg.restitution
    friction = cfg.friction
    wind_strength = cfg.wind_strength
    wind_frequency = cfg.wind_frequency
    substeps = cfg.substeps
    sphere_radius = cfg.sphere_radius
    dt = cfg.dt
    ground_height = cfg.ground_height
    auto_rotate = True
    show_wireframe = True
    fabric_tone = 0.82
    orbit_t = 0.0

    window = ti.ui.Window("Mass-Spring Cloth (Taichi GGUI)", (cfg.width, cfg.height), vsync=True)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 0.5, 2.2)
    camera.lookat(0.0, 0.2, 0.0)

    sphere_vis = ti.Vector.field(3, dtype=ti.f32, shape=1)

    while window.running:
        if auto_rotate:
            orbit_t += 0.01
            camera.position(2.15 * ti.math.sin(orbit_t), 0.8, 2.15 * ti.math.cos(orbit_t))
            camera.lookat(0.0, 0.2, 0.0)
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.point_light(pos=(1.4, 1.8, 1.2), color=(1.0, 0.98, 0.95))
        scene.point_light(pos=(-1.6, 1.2, -1.0), color=(0.45, 0.5, 0.7))
        scene.ambient_light((0.28, 0.28, 0.33))

        gui = window.get_gui()
        with gui.sub_window("Controls", 0.02, 0.02, 0.34, 0.72):
            gui.text(f"Current Solver: {SOLVER_NAME[solver_mode]}")
            if gui.button("Explicit Euler"):
                solver_mode = SOLVER_EXPLICIT
            if gui.button("Semi-Implicit Euler"):
                solver_mode = SOLVER_SEMI_IMPLICIT
            if gui.button("Implicit Euler (fixed-point)"):
                solver_mode = SOLVER_IMPLICIT

            if gui.button("Pause / Resume"):
                paused = not paused
            if gui.button("Reset Cloth"):
                sim.reset()

            gui.text("Runtime Parameters")
            damping = gui.slider_float("Damping", damping, 0.0, 10.0)
            ks_structural = gui.slider_float("Ks Structural", ks_structural, 100.0, 3000.0)
            ks_shear = gui.slider_float("Ks Shear", ks_shear, 100.0, 3000.0)
            ks_bending = gui.slider_float("Ks Bending", ks_bending, 50.0, 3000.0)
            spring_damping = gui.slider_float("Spring Damping", spring_damping, 0.0, 30.0)
            air_drag = gui.slider_float("Air Drag", air_drag, 0.0, 0.4)
            dt = gui.slider_float("dt", dt, 1e-4, 1.0 / 60.0)
            substeps = gui.slider_int("Substeps", substeps, 1, 16)
            constraint_iters = gui.slider_int("Constraint Iters", constraint_iters, 0, 12)
            constraint_stiffness = gui.slider_float("Constraint Stiff.", constraint_stiffness, 0.0, 1.0)
            max_velocity = gui.slider_float("Max Velocity", max_velocity, 0.5, 20.0)
            implicit_iters = gui.slider_int("Implicit Iters", implicit_iters, 1, 20)
            sphere_radius = gui.slider_float("Sphere Radius", sphere_radius, 0.05, 0.5)
            restitution = gui.slider_float("Restitution", restitution, 0.0, 1.0)
            friction = gui.slider_float("Friction", friction, 0.0, 1.0)
            ground_height = gui.slider_float("Ground Height", ground_height, -1.4, 0.2)
            wind_strength = gui.slider_float("Wind Strength", wind_strength, 0.0, 15.0)
            wind_frequency = gui.slider_float("Wind Frequency", wind_frequency, 0.0, 3.0)
            fabric_tone = gui.slider_float("Fabric Tone", fabric_tone, 0.4, 1.0)

            shear_enabled = gui.checkbox("Enable Shear Springs", shear_enabled)
            bending_enabled = gui.checkbox("Enable Bending Springs", bending_enabled)
            collision_enabled = gui.checkbox("Enable Sphere Collision", collision_enabled)
            show_wireframe = gui.checkbox("Show Wireframe", show_wireframe)
            auto_rotate = gui.checkbox("Auto Orbit Camera", auto_rotate)
            gui.text("Camera: RMB + WASD / drag")

        sim.set_runtime_params(
            dt=dt,
            damping=damping,
            spring_damping=spring_damping,
            air_drag=air_drag,
            ks_structural=ks_structural,
            ks_shear=ks_shear,
            ks_bending=ks_bending,
            constraint_stiffness=constraint_stiffness,
            constraint_iters=constraint_iters,
            max_velocity=max_velocity,
            implicit_iters=implicit_iters,
            substeps=substeps,
            sphere_radius=sphere_radius,
            restitution=restitution,
            friction=friction,
            ground_height=ground_height,
            wind_strength=wind_strength,
            wind_frequency=wind_frequency,
        )
        sim.set_feature_toggles(shear=shear_enabled, bending=bending_enabled, collision=collision_enabled)

        if not paused:
            for _ in range(sim.substeps[None]):
                sim.substep(solver_mode)

        scene.mesh(
            sim.x,
            indices=sim.triangle_indices,
            color=(0.84 * fabric_tone, 0.66 * fabric_tone, 0.95 * fabric_tone),
            two_sided=True,
        )
        if show_wireframe:
            scene.lines(sim.x, indices=sim.line_indices, color=(0.18, 0.18, 0.22), width=1.0)

        sphere_vis[0] = sim.sphere_center[None]
        scene.particles(sphere_vis, radius=sim.sphere_radius[None], color=(0.95, 0.35, 0.25))

        canvas.scene(scene)
        window.show()


def main():
    args = build_arg_parser().parse_args()
    cfg = SimConfig(
        grid_n=args.grid,
        grid_m=args.grid,
        damping=args.damping,
        substeps=args.substeps,
        dt=args.dt,
        wind_strength=args.wind,
    )
    run_app(cfg, init_solver=args.solver)


if __name__ == "__main__":
    main()
