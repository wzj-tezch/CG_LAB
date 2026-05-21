from __future__ import annotations

import math

import numpy as np
import taichi as ti

try:
    from .config import SOLVER_EXPLICIT, SOLVER_IMPLICIT, SOLVER_SEMI_IMPLICIT, SimConfig
except ImportError:
    from config import SOLVER_EXPLICIT, SOLVER_IMPLICIT, SOLVER_SEMI_IMPLICIT, SimConfig


SPRING_STRUCTURAL = 0
SPRING_SHEAR = 1
SPRING_BENDING = 2


@ti.data_oriented
class ClothSimulation:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.n = cfg.grid_n
        self.m = cfg.grid_m
        self.particle_count = self.n * self.m

        structural_edges = self.n * (self.m - 1) + (self.n - 1) * self.m
        shear_edges = 2 * (self.n - 1) * (self.m - 1)
        bending_edges = self.n * (self.m - 2) + (self.n - 2) * self.m
        self.max_springs = structural_edges + shear_edges + bending_edges
        self.triangle_count = (self.n - 1) * (self.m - 1) * 2

        self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_count)
        self.x0 = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_count)
        self.x_prev = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_count)
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_count)
        self.f = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_count)

        self.x_pred = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_count)
        self.v_pred = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_count)
        self.f_pred = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_count)

        self.constraint_delta = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_count)
        self.constraint_count = ti.field(dtype=ti.f32, shape=self.particle_count)

        self.fixed = ti.field(dtype=ti.i32, shape=self.particle_count)
        self.inv_mass = ti.field(dtype=ti.f32, shape=self.particle_count)

        self.spring_i = ti.field(dtype=ti.i32, shape=self.max_springs)
        self.spring_j = ti.field(dtype=ti.i32, shape=self.max_springs)
        self.spring_type = ti.field(dtype=ti.i32, shape=self.max_springs)
        self.spring_rest = ti.field(dtype=ti.f32, shape=self.max_springs)
        self.spring_count = ti.field(dtype=ti.i32, shape=())

        self.triangle_indices = ti.field(dtype=ti.i32, shape=self.triangle_count * 3)
        self.line_indices = ti.field(dtype=ti.i32, shape=self.max_springs * 2)

        self.gravity = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.wind_vec = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.dt = ti.field(dtype=ti.f32, shape=())
        self.damping = ti.field(dtype=ti.f32, shape=())
        self.spring_damping = ti.field(dtype=ti.f32, shape=())
        self.air_drag = ti.field(dtype=ti.f32, shape=())
        self.ks_structural = ti.field(dtype=ti.f32, shape=())
        self.ks_shear = ti.field(dtype=ti.f32, shape=())
        self.ks_bending = ti.field(dtype=ti.f32, shape=())
        self.constraint_stiffness = ti.field(dtype=ti.f32, shape=())
        self.constraint_iters = ti.field(dtype=ti.i32, shape=())
        self.max_velocity = ti.field(dtype=ti.f32, shape=())
        self.restitution = ti.field(dtype=ti.f32, shape=())
        self.friction = ti.field(dtype=ti.f32, shape=())
        self.ground_height = ti.field(dtype=ti.f32, shape=())
        self.wind_strength = ti.field(dtype=ti.f32, shape=())
        self.wind_frequency = ti.field(dtype=ti.f32, shape=())
        self.sim_time = ti.field(dtype=ti.f32, shape=())

        self.implicit_iters = ti.field(dtype=ti.i32, shape=())
        self.substeps = ti.field(dtype=ti.i32, shape=())
        self.particle_inv_mass = ti.field(dtype=ti.f32, shape=())
        self.cloth_size = ti.field(dtype=ti.f32, shape=())
        self.cloth_origin = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.enable_shear = ti.field(dtype=ti.i32, shape=())
        self.enable_bending = ti.field(dtype=ti.i32, shape=())
        self.enable_collision = ti.field(dtype=ti.i32, shape=())

        self.sphere_center = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.sphere_radius = ti.field(dtype=ti.f32, shape=())

        self.max_implicit_iters = 20
        self.max_constraint_iters = 20

        self._set_constants_from_config(cfg)
        self.initialize_scene()

    @ti.func
    def idx(self, i, j):
        return i * self.m + j

    @ti.func
    def spring_enabled(self, sp_type):
        enabled = 1
        if sp_type == SPRING_SHEAR:
            enabled = self.enable_shear[None]
        elif sp_type == SPRING_BENDING:
            enabled = self.enable_bending[None]
        return enabled

    @ti.func
    def spring_stiffness(self, sp_type):
        ks = self.ks_structural[None]
        if sp_type == SPRING_SHEAR:
            ks = self.ks_shear[None]
        elif sp_type == SPRING_BENDING:
            ks = self.ks_bending[None]
        return ks

    @ti.func
    def particle_weight(self, p):
        w = self.inv_mass[p]
        if self.fixed[p] == 1:
            w = 0.0
        return w

    @ti.func
    def clamp_velocity(self, vel):
        speed = vel.norm()
        if speed > self.max_velocity[None]:
            vel = vel * (self.max_velocity[None] / (speed + 1e-6))
        return vel

    @ti.func
    def apply_friction_to_velocity(self, vel, nrm):
        vn = vel.dot(nrm)
        vt = vel - vn * nrm
        vel = vn * nrm + (1.0 - self.friction[None]) * vt
        return vel

    @ti.func
    def resolve_collision(self, pos, vel):
        if self.enable_collision[None] == 1:
            offset = pos - self.sphere_center[None]
            dist = offset.norm()
            if dist < self.sphere_radius[None]:
                nrm = offset / (dist + 1e-6)
                pos = self.sphere_center[None] + nrm * (self.sphere_radius[None] + 1e-4)
                vn = vel.dot(nrm)
                if vn < 0:
                    vel = vel - (1.0 + self.restitution[None]) * vn * nrm
                vel = self.apply_friction_to_velocity(vel, nrm)

        if pos.y < self.ground_height[None]:
            pos.y = self.ground_height[None]
            nrm = ti.Vector([0.0, 1.0, 0.0])
            if vel.y < 0:
                vel.y = -vel.y * self.restitution[None]
            vel = self.apply_friction_to_velocity(vel, nrm)
        return pos, vel

    @ti.func
    def compute_forces_on(self, p):
        rel = self.v[p] - self.wind_vec[None]
        self.f[p] = self.gravity[None] / self.inv_mass[p] - self.damping[None] * self.v[p] - self.air_drag[None] * rel * rel.norm()

    @ti.func
    def compute_forces_on_pred(self, p):
        rel = self.v_pred[p] - self.wind_vec[None]
        self.f_pred[p] = self.gravity[None] / self.inv_mass[p] - self.damping[None] * self.v_pred[p] - self.air_drag[None] * rel * rel.norm()

    @ti.kernel
    def init_particles_kernel(self):
        for i, j in ti.ndrange(self.n, self.m):
            p = self.idx(i, j)
            u = ti.cast(j, ti.f32) / ti.cast(self.m - 1, ti.f32)
            v = ti.cast(i, ti.f32) / ti.cast(self.n - 1, ti.f32)
            px = self.cloth_origin[None][0] + self.cloth_size[None] * u
            py = self.cloth_origin[None][1]
            pz = self.cloth_origin[None][2] + self.cloth_size[None] * v
            self.x[p] = ti.Vector([px, py, pz])
            self.x0[p] = self.x[p]
            self.x_prev[p] = self.x[p]
            self.v[p] = ti.Vector([0.0, 0.0, 0.0])
            self.f[p] = ti.Vector([0.0, 0.0, 0.0])
            self.constraint_delta[p] = ti.Vector([0.0, 0.0, 0.0])
            self.constraint_count[p] = 0.0
            self.fixed[p] = 1 if (i == 0 and (j == 0 or j == self.m - 1)) else 0
            self.inv_mass[p] = self.particle_inv_mass[None]

    @ti.kernel
    def init_spring_kernel(self):
        for s in range(self.spring_count[None]):
            self.spring_rest[s] = ti.max(self.spring_rest[s], 1e-6)

    @ti.kernel
    def init_triangle_indices_kernel(self):
        for i, j in ti.ndrange(self.n - 1, self.m - 1):
            c = i * (self.m - 1) + j
            base = c * 6
            p00 = self.idx(i, j)
            p01 = self.idx(i, j + 1)
            p10 = self.idx(i + 1, j)
            p11 = self.idx(i + 1, j + 1)
            self.triangle_indices[base + 0] = p00
            self.triangle_indices[base + 1] = p10
            self.triangle_indices[base + 2] = p11
            self.triangle_indices[base + 3] = p00
            self.triangle_indices[base + 4] = p11
            self.triangle_indices[base + 5] = p01

    @ti.kernel
    def init_line_indices_kernel(self):
        for s in range(self.spring_count[None]):
            self.line_indices[s * 2] = self.spring_i[s]
            self.line_indices[s * 2 + 1] = self.spring_j[s]

    @ti.kernel
    def begin_substep(self):
        for p in range(self.particle_count):
            self.x_prev[p] = self.x[p]

    @ti.func
    def apply_spring_forces(self):
        for s in range(self.spring_count[None]):
            t = self.spring_type[s]
            if self.spring_enabled(t) == 1:
                i = self.spring_i[s]
                j = self.spring_j[s]
                xij = self.x[i] - self.x[j]
                vij = self.v[i] - self.v[j]
                dist = xij.norm() + 1e-6
                nrm = xij / dist
                ks = self.spring_stiffness(t)
                fs = -ks * (dist - self.spring_rest[s]) * nrm
                fs += -self.spring_damping[None] * (vij.dot(nrm)) * nrm
                for k in ti.static(range(3)):
                    ti.atomic_add(self.f[i][k], fs[k])
                    ti.atomic_add(self.f[j][k], -fs[k])

    @ti.func
    def apply_spring_forces_pred(self):
        for s in range(self.spring_count[None]):
            t = self.spring_type[s]
            if self.spring_enabled(t) == 1:
                i = self.spring_i[s]
                j = self.spring_j[s]
                xij = self.x_pred[i] - self.x_pred[j]
                vij = self.v_pred[i] - self.v_pred[j]
                dist = xij.norm() + 1e-6
                nrm = xij / dist
                ks = self.spring_stiffness(t)
                fs = -ks * (dist - self.spring_rest[s]) * nrm
                fs += -self.spring_damping[None] * (vij.dot(nrm)) * nrm
                for k in ti.static(range(3)):
                    ti.atomic_add(self.f_pred[i][k], fs[k])
                    ti.atomic_add(self.f_pred[j][k], -fs[k])

    @ti.kernel
    def step_explicit(self):
        for p in range(self.particle_count):
            self.compute_forces_on(p)

        self.apply_spring_forces()

        for p in range(self.particle_count):
            if self.fixed[p] == 1:
                self.v[p] = ti.Vector([0.0, 0.0, 0.0])
                self.x[p] = self.x0[p]
            else:
                old_v = self.v[p]
                acc = self.f[p] * self.inv_mass[p]
                new_v = self.clamp_velocity(old_v + self.dt[None] * acc)
                new_x = self.x[p] + self.dt[None] * old_v
                new_x, new_v = self.resolve_collision(new_x, new_v)
                self.v[p] = new_v
                self.x[p] = new_x

    @ti.kernel
    def step_semi_implicit(self):
        for p in range(self.particle_count):
            self.compute_forces_on(p)

        self.apply_spring_forces()

        for p in range(self.particle_count):
            if self.fixed[p] == 1:
                self.v[p] = ti.Vector([0.0, 0.0, 0.0])
                self.x[p] = self.x0[p]
            else:
                acc = self.f[p] * self.inv_mass[p]
                new_v = self.clamp_velocity(self.v[p] + self.dt[None] * acc)
                new_x = self.x[p] + self.dt[None] * new_v
                new_x, new_v = self.resolve_collision(new_x, new_v)
                self.v[p] = new_v
                self.x[p] = new_x

    @ti.kernel
    def step_implicit_iter(self):
        for p in range(self.particle_count):
            self.x_pred[p] = self.x[p]
            self.v_pred[p] = self.v[p]

        for it in range(self.max_implicit_iters):
            if it < self.implicit_iters[None]:
                for p in range(self.particle_count):
                    self.compute_forces_on_pred(p)

                self.apply_spring_forces_pred()

                for p in range(self.particle_count):
                    if self.fixed[p] == 1:
                        self.v_pred[p] = ti.Vector([0.0, 0.0, 0.0])
                        self.x_pred[p] = self.x0[p]
                    else:
                        acc = self.f_pred[p] * self.inv_mass[p]
                        next_v = self.clamp_velocity(self.v[p] + self.dt[None] * acc)
                        self.v_pred[p] = next_v
                        self.x_pred[p] = self.x[p] + self.dt[None] * next_v

        for p in range(self.particle_count):
            if self.fixed[p] == 1:
                self.v[p] = ti.Vector([0.0, 0.0, 0.0])
                self.x[p] = self.x0[p]
            else:
                nx, nv = self.resolve_collision(self.x_pred[p], self.v_pred[p])
                self.v[p] = nv
                self.x[p] = nx

    @ti.kernel
    def project_constraints(self):
        for p in range(self.particle_count):
            self.constraint_delta[p] = ti.Vector([0.0, 0.0, 0.0])
            self.constraint_count[p] = 0.0

        for s in range(self.spring_count[None]):
            t = self.spring_type[s]
            if self.spring_enabled(t) == 1:
                i = self.spring_i[s]
                j = self.spring_j[s]
                xi = self.x[i]
                xj = self.x[j]
                xij = xi - xj
                dist = xij.norm() + 1e-6
                nrm = xij / dist
                c = dist - self.spring_rest[s]
                wi = self.particle_weight(i)
                wj = self.particle_weight(j)
                wsum = wi + wj
                if wsum > 0:
                    corr = self.constraint_stiffness[None] * c * nrm / (wsum + 1e-6)
                    if wi > 0:
                        for k in ti.static(range(3)):
                            ti.atomic_add(self.constraint_delta[i][k], -wi * corr[k])
                        ti.atomic_add(self.constraint_count[i], 1.0)
                    if wj > 0:
                        for k in ti.static(range(3)):
                            ti.atomic_add(self.constraint_delta[j][k], wj * corr[k])
                        ti.atomic_add(self.constraint_count[j], 1.0)

        for p in range(self.particle_count):
            if self.fixed[p] == 1:
                self.x[p] = self.x0[p]
            elif self.constraint_count[p] > 0:
                self.x[p] += self.constraint_delta[p] / self.constraint_count[p]

    @ti.kernel
    def resolve_collision_positions(self):
        for p in range(self.particle_count):
            if self.fixed[p] == 0:
                pos = self.x[p]
                vel = self.v[p]
                pos, vel = self.resolve_collision(pos, vel)
                self.x[p] = pos
                self.v[p] = vel

    @ti.kernel
    def update_velocity_from_position(self):
        for p in range(self.particle_count):
            if self.fixed[p] == 1:
                self.v[p] = ti.Vector([0.0, 0.0, 0.0])
                self.x[p] = self.x0[p]
            else:
                self.v[p] = self.clamp_velocity((self.x[p] - self.x_prev[p]) / self.dt[None])

    def _set_constants_from_config(self, cfg: SimConfig):
        self.gravity[None] = ti.Vector(cfg.gravity)
        self.dt[None] = cfg.dt
        self.damping[None] = cfg.damping
        self.spring_damping[None] = cfg.spring_damping
        self.air_drag[None] = cfg.air_drag
        self.ks_structural[None] = cfg.ks_structural
        self.ks_shear[None] = cfg.ks_shear
        self.ks_bending[None] = cfg.ks_bending
        self.constraint_stiffness[None] = cfg.constraint_stiffness
        self.constraint_iters[None] = min(max(cfg.constraint_iters, 0), self.max_constraint_iters)
        self.max_velocity[None] = cfg.max_velocity
        self.restitution[None] = cfg.restitution
        self.friction[None] = cfg.friction
        self.ground_height[None] = cfg.ground_height
        self.wind_strength[None] = cfg.wind_strength
        self.wind_frequency[None] = cfg.wind_frequency
        self.wind_vec[None] = ti.Vector([0.0, 0.0, 0.0])
        self.sim_time[None] = 0.0
        self.sphere_center[None] = ti.Vector(cfg.sphere_center)
        self.sphere_radius[None] = cfg.sphere_radius
        self.particle_inv_mass[None] = 1.0 / cfg.particle_mass
        self.cloth_size[None] = cfg.cloth_size
        self.cloth_origin[None] = ti.Vector(cfg.cloth_origin)
        self.implicit_iters[None] = min(max(cfg.implicit_iters, 1), self.max_implicit_iters)
        self.substeps[None] = max(cfg.substeps, 1)
        self.enable_shear[None] = 1
        self.enable_bending[None] = 1
        self.enable_collision[None] = 1

    def _build_springs_topology(self):
        si: list[int] = []
        sj: list[int] = []
        st: list[int] = []
        sr: list[float] = []

        dx = self.cfg.cloth_size / max(self.m - 1, 1)
        dz = self.cfg.cloth_size / max(self.n - 1, 1)

        for i in range(self.n):
            for j in range(self.m):
                p = i * self.m + j
                if j + 1 < self.m:
                    q = i * self.m + (j + 1)
                    si.append(p)
                    sj.append(q)
                    st.append(SPRING_STRUCTURAL)
                    sr.append(dx)
                if i + 1 < self.n:
                    q = (i + 1) * self.m + j
                    si.append(p)
                    sj.append(q)
                    st.append(SPRING_STRUCTURAL)
                    sr.append(dz)

        diag = math.sqrt(dx * dx + dz * dz)
        for i in range(self.n - 1):
            for j in range(self.m - 1):
                p00 = i * self.m + j
                p01 = i * self.m + (j + 1)
                p10 = (i + 1) * self.m + j
                p11 = (i + 1) * self.m + (j + 1)
                si.extend([p00, p01])
                sj.extend([p11, p10])
                st.extend([SPRING_SHEAR, SPRING_SHEAR])
                sr.extend([diag, diag])

        for i in range(self.n):
            for j in range(self.m - 2):
                p = i * self.m + j
                q = i * self.m + (j + 2)
                si.append(p)
                sj.append(q)
                st.append(SPRING_BENDING)
                sr.append(2.0 * dx)
        for i in range(self.n - 2):
            for j in range(self.m):
                p = i * self.m + j
                q = (i + 2) * self.m + j
                si.append(p)
                sj.append(q)
                st.append(SPRING_BENDING)
                sr.append(2.0 * dz)

        return (
            np.array(si, dtype=np.int32),
            np.array(sj, dtype=np.int32),
            np.array(st, dtype=np.int32),
            np.array(sr, dtype=np.float32),
        )

    def initialize_scene(self):
        self.init_particles_kernel()
        spring_i_arr, spring_j_arr, spring_t_arr, spring_r_arr = self._build_springs_topology()
        spring_count = spring_i_arr.shape[0]
        if spring_count > self.max_springs:
            raise ValueError(f"Spring count overflow: {spring_count} > {self.max_springs}")

        i_pad = np.zeros(self.max_springs, dtype=np.int32)
        j_pad = np.zeros(self.max_springs, dtype=np.int32)
        t_pad = np.zeros(self.max_springs, dtype=np.int32)
        r_pad = np.zeros(self.max_springs, dtype=np.float32)
        i_pad[:spring_count] = spring_i_arr
        j_pad[:spring_count] = spring_j_arr
        t_pad[:spring_count] = spring_t_arr
        r_pad[:spring_count] = spring_r_arr

        self.spring_i.from_numpy(i_pad)
        self.spring_j.from_numpy(j_pad)
        self.spring_type.from_numpy(t_pad)
        self.spring_rest.from_numpy(r_pad)
        self.spring_count[None] = spring_count
        self.init_spring_kernel()
        self.init_triangle_indices_kernel()
        self.init_line_indices_kernel()

    def reset(self):
        self.initialize_scene()

    def _advance_wind(self):
        self.sim_time[None] += self.dt[None]
        phase = self.sim_time[None] * self.wind_frequency[None]
        wx = self.wind_strength[None] * math.sin(phase)
        wz = self.wind_strength[None] * math.cos(phase * 0.7)
        self.wind_vec[None] = ti.Vector([wx, 0.0, wz])

    def set_runtime_params(
        self,
        dt: float | None = None,
        damping: float | None = None,
        spring_damping: float | None = None,
        air_drag: float | None = None,
        ks_structural: float | None = None,
        ks_shear: float | None = None,
        ks_bending: float | None = None,
        constraint_stiffness: float | None = None,
        constraint_iters: int | None = None,
        max_velocity: float | None = None,
        implicit_iters: int | None = None,
        substeps: int | None = None,
        sphere_radius: float | None = None,
        restitution: float | None = None,
        friction: float | None = None,
        ground_height: float | None = None,
        wind_strength: float | None = None,
        wind_frequency: float | None = None,
    ):
        if dt is not None:
            self.dt[None] = max(1e-5, float(dt))
        if damping is not None:
            self.damping[None] = max(0.0, float(damping))
        if spring_damping is not None:
            self.spring_damping[None] = max(0.0, float(spring_damping))
        if air_drag is not None:
            self.air_drag[None] = max(0.0, float(air_drag))
        if ks_structural is not None:
            self.ks_structural[None] = max(1.0, float(ks_structural))
        if ks_shear is not None:
            self.ks_shear[None] = max(1.0, float(ks_shear))
        if ks_bending is not None:
            self.ks_bending[None] = max(1.0, float(ks_bending))
        if constraint_stiffness is not None:
            self.constraint_stiffness[None] = min(max(float(constraint_stiffness), 0.0), 1.0)
        if constraint_iters is not None:
            self.constraint_iters[None] = min(max(int(constraint_iters), 0), self.max_constraint_iters)
        if max_velocity is not None:
            self.max_velocity[None] = max(0.1, float(max_velocity))
        if implicit_iters is not None:
            self.implicit_iters[None] = min(max(int(implicit_iters), 1), self.max_implicit_iters)
        if substeps is not None:
            self.substeps[None] = max(int(substeps), 1)
        if sphere_radius is not None:
            self.sphere_radius[None] = max(float(sphere_radius), 1e-3)
        if restitution is not None:
            self.restitution[None] = min(max(float(restitution), 0.0), 1.0)
        if friction is not None:
            self.friction[None] = min(max(float(friction), 0.0), 1.0)
        if ground_height is not None:
            self.ground_height[None] = float(ground_height)
        if wind_strength is not None:
            self.wind_strength[None] = max(float(wind_strength), 0.0)
        if wind_frequency is not None:
            self.wind_frequency[None] = max(float(wind_frequency), 0.0)

    def set_feature_toggles(self, shear: bool | None = None, bending: bool | None = None, collision: bool | None = None):
        if shear is not None:
            self.enable_shear[None] = 1 if shear else 0
        if bending is not None:
            self.enable_bending[None] = 1 if bending else 0
        if collision is not None:
            self.enable_collision[None] = 1 if collision else 0

    def substep(self, solver: int):
        self._advance_wind()
        self.begin_substep()

        if solver == SOLVER_EXPLICIT:
            self.step_explicit()
        elif solver == SOLVER_SEMI_IMPLICIT:
            self.step_semi_implicit()
        elif solver == SOLVER_IMPLICIT:
            self.step_implicit_iter()
        else:
            raise ValueError(f"Unknown solver mode: {solver}")

        for _ in range(self.constraint_iters[None]):
            self.project_constraints()
            self.resolve_collision_positions()
        self.update_velocity_from_position()
