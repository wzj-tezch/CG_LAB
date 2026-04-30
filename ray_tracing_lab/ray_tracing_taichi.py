"""
Whitted-Style 光线追踪实验（Taichi）
- 场景：无限平面 + 玻璃球 + 银色镜面球
- 功能：硬阴影、理想镜面反射、玻璃折射（含全反射）、迭代式多次弹射
- 交互：光源位置、最大弹射次数、MSAA 采样数滑动条
"""

import math

import taichi as ti

WIDTH = 960
HEIGHT = 540
ASPECT = WIDTH / HEIGHT
FOV_DEG = 55.0
TAN_HALF_FOV = math.tan(math.radians(FOV_DEG * 0.5))

EPS = 1e-4
INF = 1e9
MAX_BOUNCES_CAP = 5
MAX_SPP_CAP = 8

MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_GLASS = 2

PLANE_Y = -1.0
SPHERE_R = 1.0
RED_CENTER = ti.Vector([-1.5, 0.0, 0.0])
MIRROR_CENTER = ti.Vector([1.5, 0.0, 0.0])

LIGHT_COLOR = ti.Vector([1.0, 1.0, 1.0])
BG_COLOR = ti.Vector([0.04, 0.06, 0.10])

CAM_POS = ti.Vector([0.0, 0.8, 5.2])
CAM_FORWARD = ti.Vector([0.0, -0.05, -1.0]).normalized()
WORLD_UP = ti.Vector([0.0, 1.0, 0.0])
CAM_RIGHT = CAM_FORWARD.cross(WORLD_UP).normalized()
CAM_UP = CAM_RIGHT.cross(CAM_FORWARD).normalized()

try:
    ti.init(arch=ti.gpu)
except Exception:
    ti.init(arch=ti.cpu)

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))


@ti.func
def ray_sphere_intersect(ro, rd, center, radius):
    oc = ro - center
    a = rd.dot(rd)
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    disc = b * b - 4.0 * a * c
    t = INF
    if disc >= 0.0:
        sqrt_disc = ti.sqrt(disc)
        t0 = (-b - sqrt_disc) / (2.0 * a)
        t1 = (-b + sqrt_disc) / (2.0 * a)
        if t0 > EPS:
            t = t0
        elif t1 > EPS:
            t = t1
    return t


@ti.func
def ray_plane_intersect(ro, rd):
    t = INF
    if ti.abs(rd.y) > 1e-6:
        cand = (PLANE_Y - ro.y) / rd.y
        if cand > EPS:
            t = cand
    return t


@ti.func
def board_color(hit_p):
    ix = ti.cast(ti.floor(hit_p.x), ti.i32)
    iz = ti.cast(ti.floor(hit_p.z), ti.i32)
    c = ti.Vector([0.95, 0.95, 0.95])
    if (ix + iz) & 1:
        c = ti.Vector([0.12, 0.12, 0.12])
    return c


@ti.func
def in_shadow(p, n, light_pos):
    shadow_origin = p + n * EPS
    to_light = light_pos - shadow_origin
    dist_to_light = to_light.norm()
    ldir = to_light / dist_to_light

    blocked = 0
    t_plane = ray_plane_intersect(shadow_origin, ldir)
    if t_plane < dist_to_light - EPS:
        blocked = 1

    t_red = ray_sphere_intersect(shadow_origin, ldir, RED_CENTER, SPHERE_R)
    if t_red < dist_to_light - EPS:
        blocked = 1

    t_m = ray_sphere_intersect(shadow_origin, ldir, MIRROR_CENTER, SPHERE_R)
    if t_m < dist_to_light - EPS:
        blocked = 1
    return blocked


@ti.kernel
def render(
    light_x: ti.f32,
    light_y: ti.f32,
    light_z: ti.f32,
    max_bounces: ti.i32,
    spp: ti.i32,
):
    light_pos = ti.Vector([light_x, light_y, light_z])

    for i, j in pixels:
        acc = ti.Vector([0.0, 0.0, 0.0])
        valid_spp = ti.max(1, ti.min(spp, MAX_SPP_CAP))

        for s in range(MAX_SPP_CAP):
            if s >= valid_spp:
                break

            # MSAA: 每像素随机抖动多次采样并做平均，平滑物体边缘锯齿。
            jx = ti.random(ti.f32) - 0.5
            jy = ti.random(ti.f32) - 0.5
            u = (2.0 * ((i + 0.5 + jx) / WIDTH) - 1.0) * ASPECT * TAN_HALF_FOV
            v = (2.0 * ((j + 0.5 + jy) / HEIGHT) - 1.0) * TAN_HALF_FOV
            rd = (CAM_FORWARD + u * CAM_RIGHT + v * CAM_UP).normalized()
            ro = CAM_POS

            throughput = ti.Vector([1.0, 1.0, 1.0])
            final_color = ti.Vector([0.0, 0.0, 0.0])
            ray_alive = 1

            for bounce in range(MAX_BOUNCES_CAP):
                if bounce >= max_bounces:
                    if ray_alive == 1:
                        final_color += throughput * BG_COLOR
                    ray_alive = 0
                    break

                hit_t = INF
                hit_n = ti.Vector([0.0, 1.0, 0.0])
                hit_albedo = ti.Vector([0.0, 0.0, 0.0])
                hit_mat = -1

                t_plane = ray_plane_intersect(ro, rd)
                if t_plane < hit_t:
                    hit_t = t_plane
                    hp = ro + hit_t * rd
                    hit_n = ti.Vector([0.0, 1.0, 0.0])
                    hit_albedo = board_color(hp)
                    hit_mat = MAT_DIFFUSE

                t_red = ray_sphere_intersect(ro, rd, RED_CENTER, SPHERE_R)
                if t_red < hit_t:
                    hit_t = t_red
                    hp = ro + hit_t * rd
                    hit_n = (hp - RED_CENTER).normalized()
                    hit_albedo = ti.Vector([0.95, 0.98, 1.0])
                    hit_mat = MAT_GLASS

                t_mirror = ray_sphere_intersect(ro, rd, MIRROR_CENTER, SPHERE_R)
                if t_mirror < hit_t:
                    hit_t = t_mirror
                    hp = ro + hit_t * rd
                    hit_n = (hp - MIRROR_CENTER).normalized()
                    hit_albedo = ti.Vector([0.80, 0.80, 0.80])
                    hit_mat = MAT_MIRROR

                if hit_mat < 0:
                    final_color += throughput * BG_COLOR
                    ray_alive = 0
                    break

                hit_p = ro + hit_t * rd

                if hit_mat == MAT_MIRROR:
                    reflect_dir = (rd - 2.0 * rd.dot(hit_n) * hit_n).normalized()
                    ro = hit_p + hit_n * EPS
                    rd = reflect_dir
                    throughput *= 0.8
                    continue

                if hit_mat == MAT_GLASS:
                    ior_air = 1.0
                    ior_glass = 1.5
                    n = hit_n
                    eta_i = ior_air
                    eta_t = ior_glass

                    if rd.dot(n) > 0.0:
                        # 射线在球体内部时，翻转法线并交换折射率。
                        n = -n
                        eta_i = ior_glass
                        eta_t = ior_air

                    cos_i = ti.max((-rd).dot(n), 0.0)
                    eta = eta_i / eta_t
                    k = 1.0 - eta * eta * (1.0 - cos_i * cos_i)

                    reflect_dir = (rd - 2.0 * rd.dot(n) * n).normalized()
                    if k < 0.0:
                        # 全反射：无折射解，退化为镜面反射。
                        ro = hit_p + n * EPS
                        rd = reflect_dir
                    else:
                        refract_dir = (
                            eta * rd + (eta * cos_i - ti.sqrt(k)) * n
                        ).normalized()
                        ro = hit_p - n * EPS
                        rd = refract_dir

                    throughput *= 0.96 * hit_albedo
                    continue

                ambient = 0.08 * hit_albedo
                color = ambient

                if in_shadow(hit_p, hit_n, light_pos) == 0:
                    ldir = (light_pos - hit_p).normalized()
                    n_dot_l = ti.max(hit_n.dot(ldir), 0.0)
                    diffuse = hit_albedo * n_dot_l

                    view_dir = (-rd).normalized()
                    refl_l = (-ldir - 2.0 * (-ldir).dot(hit_n) * hit_n).normalized()
                    spec = ti.pow(ti.max(refl_l.dot(view_dir), 0.0), 32.0)
                    specular = 0.25 * spec * LIGHT_COLOR
                    color += diffuse + specular

                final_color += throughput * color
                ray_alive = 0
                break

            acc += ti.min(final_color, 1.0)

        pixels[i, j] = acc / ti.cast(valid_spp, ti.f32)


def main():
    window = ti.ui.Window("Whitted Ray Tracing Lab", (WIDTH, HEIGHT), vsync=True)
    canvas = window.get_canvas()
    gui = window.get_gui()

    light_x = 3.0
    light_y = 5.0
    light_z = 2.0
    max_bounces = 3
    msaa_spp = 4

    while window.running:
        with gui.sub_window("Controls", 0.02, 0.02, 0.30, 0.38):
            light_x = gui.slider_float("Light X", light_x, -8.0, 8.0)
            light_y = gui.slider_float("Light Y", light_y, 0.5, 10.0)
            light_z = gui.slider_float("Light Z", light_z, -8.0, 8.0)
            max_bounces = gui.slider_int("Max Bounces", max_bounces, 1, 5)
            msaa_spp = gui.slider_int("MSAA Samples", msaa_spp, 1, 8)
            gui.text("Left sphere: Glass (IOR=1.5)")
            gui.text("Right sphere: Mirror (reflect=0.8)")
            gui.text("Epsilon offset = 1e-4")

        render(light_x, light_y, light_z, max_bounces, msaa_spp)
        canvas.set_image(pixels)
        window.show()


if __name__ == "__main__":
    main()
