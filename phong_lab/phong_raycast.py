"""
Phong 光照模型实验 — 光线投射 + Z 竞争 + ti.ui 交互
课程参考: https://zhanghongwen.cn/cg

环境变量 PHONG_HEADLESS=1 时使用 CPU 后端，便于无窗口导出 GIF（见 export_preview_gif.py）。
"""
import math
import os
import sys

import taichi as ti
import taichi.math as tm


def _init_taichi():
    if os.environ.get("PHONG_HEADLESS") == "1":
        ti.init(arch=ti.cpu)
        return
    try:
        ti.init(arch=ti.vulkan)
    except Exception:
        try:
            ti.init(arch=ti.cuda)
        except Exception:
            ti.init(arch=ti.cpu)


_init_taichi()

WIDTH, HEIGHT = 800, 600
ASPECT = WIDTH / HEIGHT

CAM = tm.vec3(0.0, 0.0, 5.0)
LIGHT_POS = tm.vec3(2.0, 3.0, 4.0)
LIGHT_COLOR = tm.vec3(1.0, 1.0, 1.0)
BG_COLOR = tm.vec3(0.0, 0.25, 0.32)  # 深青色背景

# 红色球体：略靠左、向画面中心收拢，与圆锥留出间隙避免穿插
SPHERE_C = tm.vec3(-0.62, -0.05, 0.0)
SPHERE_R = 0.58
COLOR_SPHERE = tm.vec3(0.8, 0.1, 0.1)

# 紫色圆锥：与球大致对称，略靠右
CONE_APEX = tm.vec3(0.62, 0.65, 0.0)
CONE_AXIS = tm.vec3(0.0, -1.0, 0.0)  # 单位轴，由顶点指向底面
CONE_H = 1.28
CONE_BASE_R = 0.58
COLOR_CONE = tm.vec3(0.6, 0.2, 0.8)

# 底面圆心（与顶点同 x,z）：apex.y - CONE_H
CONE_BASE_C = tm.vec3(0.62, -0.63, 0.0)

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))


@ti.func
def ray_sphere(o: tm.vec3, d: tm.vec3, center: tm.vec3, rad: ti.f32) -> ti.f32:
    """返回最近正交点 t；无命中返回 -1.0。假定 d 已单位化。"""
    oc = o - center
    b = tm.dot(oc, d)
    c = tm.dot(oc, oc) - rad * rad
    disc = b * b - c
    t_hit = -1.0
    if disc >= 0.0:
        s = ti.sqrt(disc)
        t0 = -b - s
        t1 = -b + s
        if t0 > 1e-4:
            t_hit = t0
        elif t1 > 1e-4:
            t_hit = t1
    return t_hit


@ti.func
def sphere_normal(p: tm.vec3, center: tm.vec3) -> tm.vec3:
    return tm.normalize(p - center)


@ti.func
def ray_cone_side(o: tm.vec3, d: tm.vec3, cos2: ti.f32) -> ti.f32:
    """
    有限圆锥侧面（单叶）：轴过顶点，单位轴 CONE_AXIS，cos^2(theta)=cos2。
    限制：0 < (P-A)·axis < CONE_H
    """
    w0 = o - CONE_APEX
    a = tm.dot(w0, CONE_AXIS)
    b = tm.dot(d, CONE_AXIS)
    dd = tm.dot(d, d)  # 应为 1
    w0d = tm.dot(w0, d)
    w02 = tm.dot(w0, w0)

    A = b * b - cos2 * dd
    B = 2.0 * (a * b - cos2 * w0d)
    C = a * a - cos2 * w02

    t_hit = -1.0
    disc = B * B - 4.0 * A * C
    if ti.abs(A) > 1e-8 and disc >= 0.0:
        s = ti.sqrt(disc)
        t0 = (-B - s) / (2.0 * A)
        t1 = (-B + s) / (2.0 * A)
        if t0 > 1e-4:
            p0 = o + t0 * d
            h0 = tm.dot(p0 - CONE_APEX, CONE_AXIS)
            if h0 > 1e-4 and h0 < CONE_H - 1e-4:
                t_hit = t0
        if t1 > 1e-4:
            p1 = o + t1 * d
            h1 = tm.dot(p1 - CONE_APEX, CONE_AXIS)
            if h1 > 1e-4 and h1 < CONE_H - 1e-4:
                if t_hit < 0.0 or t1 < t_hit:
                    t_hit = t1
    return t_hit


@ti.func
def cone_side_normal(p: tm.vec3, cos2: ti.f32) -> tm.vec3:
    """圆锥侧面外法向（梯度法）。"""
    w = p - CONE_APEX
    g = 2.0 * cos2 * w - 2.0 * tm.dot(w, CONE_AXIS) * CONE_AXIS
    return tm.normalize(g)


@ti.func
def ray_disk(o: tm.vec3, d: tm.vec3, center: tm.vec3, nrm: tm.vec3, rad: ti.f32) -> ti.f32:
    """与圆盘（有限）求交；法向 nrm 指向外侧（圆锥外）。"""
    denom = tm.dot(d, nrm)
    t_hit = -1.0
    if ti.abs(denom) > 1e-6:
        t = tm.dot(center - o, nrm) / denom
        if t > 1e-4:
            p = o + t * d
            if tm.length(p - center) <= rad + 1e-5:
                t_hit = t
    return t_hit


@ti.func
def disk_normal() -> tm.vec3:
    """底面朝 -y 为外法向（从圆锥内部看向外）。"""
    return tm.vec3(0.0, -1.0, 0.0)


@ti.kernel
def render(ka: ti.f32, kd: ti.f32, ks: ti.f32, shininess: ti.f32, cone_cos2: ti.f32):
    for i, j in pixels:
        u = (ti.cast(i, ti.f32) + 0.5) / ti.cast(WIDTH, ti.f32)
        j_up = HEIGHT - 1 - j
        v = (ti.cast(j_up, ti.f32) + 0.5) / ti.cast(HEIGHT, ti.f32)
        ndc_x = (u - 0.5) * 2.0 * ASPECT
        ndc_y = (v - 0.5) * 2.0
        plane = tm.vec3(ndc_x, ndc_y, 0.0)
        d = tm.normalize(plane - CAM)

        t_s = ray_sphere(CAM, d, SPHERE_C, SPHERE_R)
        t_cone_side = ray_cone_side(CAM, d, cone_cos2)
        t_cone_cap = ray_disk(CAM, d, CONE_BASE_C, disk_normal(), CONE_BASE_R)

        t_best = 1e30
        obj = 0  # 0 无, 1 球, 2 锥侧面, 3 锥底

        if t_s > 0.0 and t_s < t_best:
            t_best = t_s
            obj = 1
        if t_cone_side > 0.0 and t_cone_side < t_best:
            t_best = t_cone_side
            obj = 2
        if t_cone_cap > 0.0 and t_cone_cap < t_best:
            t_best = t_cone_cap
            obj = 3

        if t_best >= 1e29:
            pixels[i, j] = BG_COLOR
        else:
            p = CAM + t_best * d
            n = tm.vec3(0.0)
            base_color = tm.vec3(0.0)
            if obj == 1:
                n = sphere_normal(p, SPHERE_C)
                base_color = COLOR_SPHERE
            elif obj == 2:
                n = cone_side_normal(p, cone_cos2)
                base_color = COLOR_CONE
            else:
                n = disk_normal()
                base_color = COLOR_CONE

            L = tm.normalize(LIGHT_POS - p)
            V = tm.normalize(CAM - p)
            if tm.dot(n, V) < 0.0:
                n = -n
            R = 2.0 * tm.dot(n, L) * n - L

            amb = ka * LIGHT_COLOR * base_color
            ndotl = ti.max(0.0, tm.dot(n, L))
            diff = kd * ndotl * LIGHT_COLOR * base_color
            rv = ti.max(0.0, tm.dot(R, V))
            spec = ks * ti.pow(rv, shininess) * LIGHT_COLOR

            c = amb + diff + spec
            pixels[i, j] = tm.clamp(c, 0.0, 1.0)


def main():
    hyp = math.sqrt(CONE_H * CONE_H + CONE_BASE_R * CONE_BASE_R)
    cos_theta = CONE_H / hyp
    cone_cos2 = float(cos_theta * cos_theta)

    ka, kd, ks = 0.2, 0.7, 0.5
    shininess = 32.0

    window = ti.ui.Window("Phong 光照模型", (WIDTH, HEIGHT), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((0.0, 0.25, 0.32))

    while window.running:
        render(ka, kd, ks, shininess, cone_cos2)

        canvas.set_image(pixels)
        # ImGui 默认字体不含中文，控件标签请用 ASCII，避免显示为 ???
        window.GUI.begin("Material", 0.02, 0.02, 0.32, 0.22)
        ka = window.GUI.slider_float("Ka (ambient)", ka, 0.0, 1.0)
        kd = window.GUI.slider_float("Kd (diffuse)", kd, 0.0, 1.0)
        ks = window.GUI.slider_float("Ks (specular)", ks, 0.0, 1.0)
        shininess = window.GUI.slider_float("Shininess", shininess, 1.0, 128.0)
        window.GUI.end()

        window.show()

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
