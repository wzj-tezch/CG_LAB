"""
Mass-spring cloth simulation with three integrators in Taichi.
- Structural springs only
- Explicit Euler / Semi-Implicit Euler / Approximate Implicit Euler
- GGUI controls for solver switching, pause and reset
"""

import taichi as ti


def init_taichi():
    try:
        ti.init(arch=ti.vulkan)
    except Exception:
        try:
            ti.init(arch=ti.cuda)
        except Exception:
            ti.init(arch=ti.cpu)


init_taichi()

GRID_N = 20
NUM_VERTS = GRID_N * GRID_N
# 结构弹簧 (Structural): 水平 + 垂直
NUM_STRUCT_SPRINGS = 2 * GRID_N * (GRID_N - 1)
# 剪切弹簧 (Shear): 对角线
NUM_SHEAR_SPRINGS = 2 * (GRID_N - 1) * (GRID_N - 1)
# 弯曲弹簧 (Bending): 跨越一个质点
NUM_BENDING_SPRINGS = 2 * GRID_N * (GRID_N - 2)
NUM_SPRINGS = NUM_STRUCT_SPRINGS + NUM_SHEAR_SPRINGS + NUM_BENDING_SPRINGS

NUM_TRIS = 2 * (GRID_N - 1) * (GRID_N - 1)
WIDTH = 1024
HEIGHT = 1024

SOLVER_EXPLICIT = 0
SOLVER_SEMI_IMPLICIT = 1
SOLVER_IMPLICIT = 2
IMPLICIT_MAX_ITERS = 8

CLOTH_WIDTH = 0.8
CLOTH_HEIGHT = 0.8
PARTICLE_RADIUS = 0.008

positions = ti.Vector.field(3, dtype=ti.f32, shape=NUM_VERTS)
velocities = ti.Vector.field(3, dtype=ti.f32, shape=NUM_VERTS)
forces = ti.Vector.field(3, dtype=ti.f32, shape=NUM_VERTS)
rest_positions = ti.Vector.field(3, dtype=ti.f32, shape=NUM_VERTS)
vertex_colors = ti.Vector.field(3, dtype=ti.f32, shape=NUM_VERTS)

old_positions = ti.Vector.field(3, dtype=ti.f32, shape=NUM_VERTS)
old_velocities = ti.Vector.field(3, dtype=ti.f32, shape=NUM_VERTS)
predicted_positions = ti.Vector.field(3, dtype=ti.f32, shape=NUM_VERTS)
predicted_velocities = ti.Vector.field(3, dtype=ti.f32, shape=NUM_VERTS)
predicted_forces = ti.Vector.field(3, dtype=ti.f32, shape=NUM_VERTS)

spring_indices = ti.Vector.field(2, dtype=ti.i32, shape=NUM_SPRINGS)
spring_rest_lengths = ti.field(dtype=ti.f32, shape=NUM_SPRINGS)
spring_types = ti.field(dtype=ti.i32, shape=NUM_SPRINGS) # 0: Struct, 1: Shear, 2: Bend
triangle_indices = ti.field(dtype=ti.i32, shape=NUM_TRIS * 3)
pinned = ti.field(dtype=ti.i32, shape=NUM_VERTS)

# 地面数据
floor_pos = ti.Vector.field(3, dtype=ti.f32, shape=4)
floor_indices = ti.field(dtype=ti.i32, shape=6)

# 球体障碍物
sphere_center = ti.Vector.field(3, dtype=ti.f32, shape=(1,))
sphere_radius = ti.field(dtype=ti.f32, shape=())
sphere_enabled = ti.field(dtype=ti.i32, shape=())


@ti.func
def vertex_id(i: ti.i32, j: ti.i32) -> ti.i32:
    return i * GRID_N + j


@ti.kernel
def init_particle_positions():
    for i, j in ti.ndrange(GRID_N, GRID_N):
        idx = vertex_id(i, j)
        u = ti.cast(j, ti.f32) / ti.cast(GRID_N - 1, ti.f32)
        v = ti.cast(i, ti.f32) / ti.cast(GRID_N - 1, ti.f32)
        x = (u - 0.5) * CLOTH_WIDTH
        y = 0.5 # 降低高度
        z = (v - 0.5) * CLOTH_HEIGHT
        pos = ti.Vector([x, y, z])
        positions[idx] = pos
        rest_positions[idx] = pos

    # 初始化地面
    floor_pos[0] = ti.Vector([-2.0, -0.6, -2.0])
    floor_pos[1] = ti.Vector([ 2.0, -0.6, -2.0])
    floor_pos[2] = ti.Vector([ 2.0, -0.6,  2.0])
    floor_pos[3] = ti.Vector([-2.0, -0.6,  2.0])
    floor_indices[0], floor_indices[1], floor_indices[2] = 0, 1, 2
    floor_indices[3], floor_indices[4], floor_indices[5] = 0, 2, 3

    # 初始化球体
    sphere_center[0] = ti.Vector([0.0, -0.1, 0.0])
    sphere_radius[None] = 0.25
    sphere_enabled[None] = 1


@ti.kernel
def init_particle_state():
    for i, j in ti.ndrange(GRID_N, GRID_N):
        idx = vertex_id(i, j)
        velocities[idx] = ti.Vector([0.0, 0.0, 0.0])
        forces[idx] = ti.Vector([0.0, 0.0, 0.0])
        predicted_positions[idx] = positions[idx]
        predicted_velocities[idx] = ti.Vector([0.0, 0.0, 0.0])
        predicted_forces[idx] = ti.Vector([0.0, 0.0, 0.0])
        old_positions[idx] = positions[idx]
        old_velocities[idx] = ti.Vector([0.0, 0.0, 0.0])
        u = ti.cast(j, ti.f32) / ti.cast(GRID_N - 1, ti.f32)
        v = ti.cast(i, ti.f32) / ti.cast(GRID_N - 1, ti.f32)
        color = ti.Vector([0.2 + 0.6 * u, 0.35 + 0.35 * (1.0 - v), 0.9 - 0.35 * u])
        vertex_colors[idx] = color


@ti.kernel
def init_pin_constraints():
    for i, j in ti.ndrange(GRID_N, GRID_N):
        idx = vertex_id(i, j)
        pinned[idx] = 0
        if i == 0 and (j == 0 or j == GRID_N - 1):
            pinned[idx] = 1
            vertex_colors[idx] = ti.Vector([1.0, 0.2, 0.2])


@ti.kernel
def init_all_springs():
    # 1. Structural Springs
    for i, j in ti.ndrange(GRID_N, GRID_N - 1):
        s = i * (GRID_N - 1) + j
        a, b = vertex_id(i, j), vertex_id(i, j + 1)
        spring_indices[s] = ti.Vector([a, b])
        spring_rest_lengths[s] = (rest_positions[a] - rest_positions[b]).norm()
        spring_types[s] = 0

    base = GRID_N * (GRID_N - 1)
    for i, j in ti.ndrange(GRID_N - 1, GRID_N):
        s = base + i * GRID_N + j
        a, b = vertex_id(i, j), vertex_id(i + 1, j)
        spring_indices[s] = ti.Vector([a, b])
        spring_rest_lengths[s] = (rest_positions[a] - rest_positions[b]).norm()
        spring_types[s] = 0

    # 2. Shear Springs
    base = 2 * GRID_N * (GRID_N - 1)
    for i, j in ti.ndrange(GRID_N - 1, GRID_N - 1):
        # 对角线 1
        s1 = base + (i * (GRID_N - 1) + j) * 2
        a1, b1 = vertex_id(i, j), vertex_id(i + 1, j + 1)
        spring_indices[s1] = ti.Vector([a1, b1])
        spring_rest_lengths[s1] = (rest_positions[a1] - rest_positions[b1]).norm()
        spring_types[s1] = 1
        # 对角线 2
        s2 = s1 + 1
        a2, b2 = vertex_id(i, j + 1), vertex_id(i + 1, j)
        spring_indices[s2] = ti.Vector([a2, b2])
        spring_rest_lengths[s2] = (rest_positions[a2] - rest_positions[b2]).norm()
        spring_types[s2] = 1

    # 3. Bending Springs
    base = 2 * GRID_N * (GRID_N - 1) + 2 * (GRID_N - 1) * (GRID_N - 1)
    for i, j in ti.ndrange(GRID_N, GRID_N - 2):
        s = base + i * (GRID_N - 2) + j
        a, b = vertex_id(i, j), vertex_id(i, j + 2)
        spring_indices[s] = ti.Vector([a, b])
        spring_rest_lengths[s] = (rest_positions[a] - rest_positions[b]).norm()
        spring_types[s] = 2

    base += GRID_N * (GRID_N - 2)
    for i, j in ti.ndrange(GRID_N - 2, GRID_N):
        s = base + i * GRID_N + j
        a, b = vertex_id(i, j), vertex_id(i + 2, j)
        spring_indices[s] = ti.Vector([a, b])
        spring_rest_lengths[s] = (rest_positions[a] - rest_positions[b]).norm()
        spring_types[s] = 2


@ti.kernel
def init_render_indices():
    for i, j in ti.ndrange(GRID_N - 1, GRID_N - 1):
        cell = i * (GRID_N - 1) + j
        base = cell * 6
        v00 = vertex_id(i, j)
        v01 = vertex_id(i, j + 1)
        v10 = vertex_id(i + 1, j)
        v11 = vertex_id(i + 1, j + 1)
        triangle_indices[base + 0] = v00
        triangle_indices[base + 1] = v10
        triangle_indices[base + 2] = v11
        triangle_indices[base + 3] = v00
        triangle_indices[base + 4] = v11
        triangle_indices[base + 5] = v01


def reset_simulation():
    init_particle_positions()
    init_particle_state()
    init_pin_constraints()


def initialize_scene_data():
    init_particle_positions()
    init_particle_state()
    init_pin_constraints()
    init_all_springs()
    init_render_indices()


@ti.func
def clamp_velocity(v: ti.math.vec3, max_speed: ti.f32) -> ti.math.vec3:
    speed = v.norm()
    if speed > max_speed:
        v = v / speed * max_speed
    return v


@ti.func
def compute_forces_on(
    pos: ti.template(),
    vel: ti.template(),
    out_force: ti.template(),
    ks: ti.f32,
    kd: ti.f32,
    gravity_y: ti.f32,
    mass: ti.f32,
    wind_strength: ti.f32,
    time: ti.f32,
):
    for i in range(NUM_VERTS):
        # 基础力：重力 + 阻尼
        f = ti.Vector([0.0, gravity_y * mass, 0.0]) - kd * vel[i]
        
        # 动态风力 (随时间变化的正弦波风)
        wind_dir = ti.Vector([ti.sin(time), 0.0, ti.cos(time * 0.5)]).normalized()
        f += wind_dir * wind_strength * (ti.sin(time * 2.0 + pos[i].x * 5.0) + 1.0)
        
        out_force[i] = f

    for s in range(NUM_SPRINGS):
        a = spring_indices[s][0]
        b = spring_indices[s][1]
        stype = spring_types[s]
        
        # 不同类型的弹簧可以有不同的劲度系数系数
        # 0: Struct, 1: Shear, 2: Bend
        current_ks = ks
        if stype == 1: current_ks = ks * 0.5
        elif stype == 2: current_ks = ks * 0.2
        
        delta = pos[a] - pos[b]
        dist = delta.norm()
        if dist > 1e-6:
            direction = delta / dist
            extension = dist - spring_rest_lengths[s]
            spring_force = -current_ks * extension * direction
            for k in ti.static(range(3)):
                ti.atomic_add(out_force[a][k], spring_force[k])
                ti.atomic_add(out_force[b][k], -spring_force[k])

@ti.func
def handle_collisions(pos: ti.template(), vel: ti.template()):
    for i in range(NUM_VERTS):
        # 地面碰撞
        if pos[i].y < -0.6:
            pos[i].y = -0.6
            if vel[i].y < 0:
                vel[i].y = 0
        
        # 球体碰撞
        if sphere_enabled[None] == 1:
            delta = pos[i] - sphere_center[0]
            dist = delta.norm()
            if dist < sphere_radius[None] + 0.01: # 略大一点作为碰撞裕量
                normal = delta / dist
                pos[i] = sphere_center[0] + normal * (sphere_radius[None] + 0.01)
                # 简单的碰撞响应：消除法向速度
                v_normal = vel[i].dot(normal)
                if v_normal < 0:
                    vel[i] -= v_normal * normal


@ti.kernel
def step_explicit(
    dt: ti.f32,
    mass: ti.f32,
    ks: ti.f32,
    kd: ti.f32,
    gravity_y: ti.f32,
    max_speed: ti.f32,
    wind_strength: ti.f32,
    time: ti.f32,
):
    compute_forces_on(positions, velocities, forces, ks, kd, gravity_y, mass, wind_strength, time)
    for i in range(NUM_VERTS):
        if pinned[i] == 1:
            positions[i] = rest_positions[i]
            velocities[i] = ti.Vector([0.0, 0.0, 0.0])
        else:
            accel = forces[i] / mass
            current_v = velocities[i]
            next_x = positions[i] + current_v * dt
            next_v = clamp_velocity(current_v + accel * dt, max_speed)
            positions[i] = next_x
            velocities[i] = next_v
    handle_collisions(positions, velocities)


@ti.kernel
def step_semi_implicit(
    dt: ti.f32,
    mass: ti.f32,
    ks: ti.f32,
    kd: ti.f32,
    gravity_y: ti.f32,
    max_speed: ti.f32,
    wind_strength: ti.f32,
    time: ti.f32,
):
    compute_forces_on(positions, velocities, forces, ks, kd, gravity_y, mass, wind_strength, time)
    for i in range(NUM_VERTS):
        if pinned[i] == 1:
            positions[i] = rest_positions[i]
            velocities[i] = ti.Vector([0.0, 0.0, 0.0])
        else:
            accel = forces[i] / mass
            next_v = clamp_velocity(velocities[i] + accel * dt, max_speed)
            next_x = positions[i] + next_v * dt
            positions[i] = next_x
            velocities[i] = next_v
    handle_collisions(positions, velocities)


@ti.kernel
def step_implicit_iter(
    dt: ti.f32,
    mass: ti.f32,
    ks: ti.f32,
    kd: ti.f32,
    gravity_y: ti.f32,
    max_speed: ti.f32,
    iterations: ti.i32,
    wind_strength: ti.f32,
    time: ti.f32,
):
    for i in range(NUM_VERTS):
        old_positions[i] = positions[i]
        old_velocities[i] = velocities[i]
        predicted_positions[i] = positions[i]
        predicted_velocities[i] = velocities[i]

    for it in range(IMPLICIT_MAX_ITERS):
        if it < iterations:
            compute_forces_on(
                predicted_positions,
                predicted_velocities,
                predicted_forces,
                ks,
                kd,
                gravity_y,
                mass,
                wind_strength,
                time,
            )
            for i in range(NUM_VERTS):
                if pinned[i] == 1:
                    predicted_positions[i] = rest_positions[i]
                    predicted_velocities[i] = ti.Vector([0.0, 0.0, 0.0])
                else:
                    next_v = clamp_velocity(
                        old_velocities[i] + dt * predicted_forces[i] / mass,
                        max_speed,
                    )
                    next_x = old_positions[i] + dt * next_v
                    predicted_velocities[i] = next_v
                    predicted_positions[i] = next_x

    for i in range(NUM_VERTS):
        positions[i] = predicted_positions[i]
        velocities[i] = predicted_velocities[i]
    handle_collisions(positions, velocities)


def solver_name(solver: int) -> str:
    if solver == SOLVER_EXPLICIT:
        return "Explicit Euler"
    if solver == SOLVER_SEMI_IMPLICIT:
        return "Semi-Implicit Euler"
    return "Implicit Euler (Fixed-Point)"


def main():
    initialize_scene_data()

    dt = 1.0 / 240.0
    mass = 1.0
    ks = 1400.0
    kd = 5.0
    gravity_y = -9.8
    max_speed = 3.0
    substeps = 12
    implicit_iters = 4
    solver = SOLVER_SEMI_IMPLICIT
    paused = False
    
    wind_strength = 0.5
    show_sphere = True
    current_time = 0.0

    window = ti.ui.Window("Mass Spring Cloth", (WIDTH, HEIGHT), vsync=True)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 0.4, 2.0) # 调远一点，且视角稍微向下俯视
    camera.lookat(0.0, -0.1, 0.0)
    camera.up(0.0, 1.0, 0.0)

    while window.running:
        camera.track_user_inputs(window, movement_speed=0.02, hold_key=ti.ui.RMB)

        window.GUI.begin("Controls", 0.02, 0.02, 0.30, 0.45)
        if window.GUI.button("Explicit Euler"):
            solver = SOLVER_EXPLICIT
        if window.GUI.button("Semi-Implicit Euler"):
            solver = SOLVER_SEMI_IMPLICIT
        if window.GUI.button("Implicit Euler"):
            solver = SOLVER_IMPLICIT
        
        paused = window.GUI.checkbox("Pause", paused)
        if window.GUI.button("Reset Cloth"):
            reset_simulation()
            current_time = 0.0
            
        window.GUI.text("Physics Parameters")
        ks = window.GUI.slider_float("Spring Ks", ks, 100.0, 5000.0)
        kd = window.GUI.slider_float("Damping Kd", kd, 0.1, 30.0)
        dt = window.GUI.slider_float("Time Step", dt, 1.0 / 2000.0, 1.0 / 60.0)
        max_speed = window.GUI.slider_float("Max Speed", max_speed, 0.5, 10.0)
        substeps = window.GUI.slider_int("Substeps", substeps, 1, 24)
        
        window.GUI.text("Environment")
        wind_strength = window.GUI.slider_float("Wind Strength", wind_strength, 0.0, 5.0)
        show_sphere = window.GUI.checkbox("Show Sphere Obstacle", show_sphere)
        sphere_enabled[None] = 1 if show_sphere else 0
        
        if solver == SOLVER_IMPLICIT:
            implicit_iters = window.GUI.slider_int("Implicit Iters", implicit_iters, 1, IMPLICIT_MAX_ITERS)
            
        window.GUI.text("Current Solver:")
        window.GUI.text(solver_name(solver))
        window.GUI.text("RMB + drag to move camera")
        window.GUI.end()

        if not paused:
            for _ in range(substeps):
                if solver == SOLVER_EXPLICIT:
                    step_explicit(dt, mass, ks, kd, gravity_y, max_speed, wind_strength, current_time)
                elif solver == SOLVER_SEMI_IMPLICIT:
                    step_semi_implicit(dt, mass, ks, kd, gravity_y, max_speed, wind_strength, current_time)
                else:
                    step_implicit_iter(
                        dt,
                        mass,
                        ks,
                        kd,
                        gravity_y,
                        max_speed,
                        implicit_iters,
                        wind_strength,
                        current_time,
                    )
                current_time += dt

        scene.set_camera(camera)
        scene.ambient_light((0.4, 0.4, 0.4))
        scene.point_light(pos=(2.0, 2.5, 2.0), color=(1.0, 1.0, 1.0))
        scene.point_light(pos=(-2.0, 2.5, 2.0), color=(0.8, 0.8, 1.0))
        
        # 渲染布料
        scene.mesh(
            positions,
            indices=triangle_indices,
            per_vertex_color=vertex_colors,
            two_sided=True,
        )
        
        # 渲染球体
        if show_sphere:
            scene.particles(
                sphere_center,
                radius=sphere_radius[None],
                color=(0.9, 0.4, 0.1),
            )
            
        # 渲染地面
        scene.mesh(
            floor_pos,
            indices=floor_indices,
            color=(0.3, 0.3, 0.3),
        )
        
        canvas.scene(scene)
        
        # 新增：按 S 键保存当前帧截图
        if window.get_event(ti.ui.PRESS):
            if window.event.key == 's':
                os.makedirs("assets", exist_ok=True)
                window.save_image("assets/screenshot.png")
                print("Screenshot saved to assets/screenshot.png")
        
        window.show()


if __name__ == "__main__":
    main()
