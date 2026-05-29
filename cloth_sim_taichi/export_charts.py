"""定量分析：三种积分器稳定性、能量与性能对比图表导出。"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import taichi as ti

try:
    from .config import SOLVER_EXPLICIT, SOLVER_IMPLICIT, SOLVER_SEMI_IMPLICIT, SOLVER_NAME, SimConfig, output_root
    from .simulation import ClothSimulation
except ImportError:
    from config import SOLVER_EXPLICIT, SOLVER_IMPLICIT, SOLVER_SEMI_IMPLICIT, SOLVER_NAME, SimConfig, output_root
    from simulation import ClothSimulation

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CHART_DIR = output_root() / "charts"
ASSET_DIR = Path(__file__).resolve().parent / "assets"


def measure_case(solver: int, steps: int, dt: float, damping: float, ks_scale: float = 1.0):
    cfg = SimConfig(arch=ti.cpu, dt=dt, substeps=1, damping=damping)
    sim = ClothSimulation(cfg)
    sim.set_runtime_params(
        ks_structural=900.0 * ks_scale,
        ks_shear=650.0 * ks_scale,
        ks_bending=450.0 * ks_scale,
        constraint_iters=2,
        wind_strength=4.0,
    )
    sim.set_feature_toggles(shear=True, bending=True, collision=True)

    inv_mass = 1.0 / cfg.particle_mass
    times, ke, max_v, com_y, spring_err = [], [], [], [], []
    exploded = False

    t0 = time.perf_counter()
    for step in range(steps):
        sim.substep(solver)
        v = sim.v.to_numpy()
        x = sim.x.to_numpy()
        speeds = np.linalg.norm(v, axis=1)
        mv = float(speeds.max())
        if mv > cfg.max_velocity * 0.99 or not np.isfinite(mv):
            exploded = True
        ke_val = 0.5 * inv_mass * float(np.sum(np.sum(v * v, axis=1)))
        times.append(step * dt)
        ke.append(ke_val)
        max_v.append(mv)
        com_y.append(float(x[:, 1].mean()))

        sc = int(sim.spring_count[None])
        si = sim.spring_i.to_numpy()[:sc]
        sj = sim.spring_j.to_numpy()[:sc]
        sr = sim.spring_rest.to_numpy()[:sc]
        dist = np.linalg.norm(x[si] - x[sj], axis=1)
        spring_err.append(float(np.mean(np.abs(dist - sr))))

        if exploded and step > 50:
            break

    elapsed = time.perf_counter() - t0
    return {
        "solver": solver,
        "label": SOLVER_NAME[solver],
        "times": np.array(times),
        "ke": np.array(ke),
        "max_v": np.array(max_v),
        "com_y": np.array(com_y),
        "spring_err": np.array(spring_err),
        "exploded": exploded,
        "ms_per_step": elapsed / max(len(times), 1) * 1000.0,
    }


def plot_energy_curves(cases: list[dict], out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    colors = {"Explicit Euler": "#e74c3c", "Semi-Implicit Euler": "#3498db", "Implicit Euler (fixed-point)": "#2ecc71"}
    for c in cases:
        ax.plot(c["times"], c["ke"], label=c["label"], color=colors.get(c["label"], None), linewidth=1.8)
    ax.set_xlabel("仿真时间 (s)")
    ax.set_ylabel("总动能 (J)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_max_velocity(cases: list[dict], out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    colors = {"Explicit Euler": "#e74c3c", "Semi-Implicit Euler": "#3498db", "Implicit Euler (fixed-point)": "#2ecc71"}
    for c in cases:
        ax.plot(c["times"], c["max_v"], label=c["label"], color=colors.get(c["label"], None), linewidth=1.8)
    ax.axhline(6.0, color="#888", linestyle="--", linewidth=1, label="速度钳制上限")
    ax.set_xlabel("仿真时间 (s)")
    ax.set_ylabel("最大质点速度 (m/s)")
    ax.set_title("三种积分器最大速度随时间变化")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_com_height(cases: list[dict], out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    for c in cases:
        ax.plot(c["times"], c["com_y"], label=c["label"], linewidth=1.8)
    ax.set_xlabel("仿真时间 (s)")
    ax.set_ylabel("质心高度 (m)")
    ax.set_title("布料质心下落轨迹对比")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_stability_bar(records: list[dict], out_path: Path):
    labels = [r["label"].replace(" (fixed-point)", "") for r in records]
    max_dt = [r["max_stable_dt"] * 1000 for r in records]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    bars = ax.bar(labels, max_dt, color=["#e74c3c", "#3498db", "#2ecc71"], edgecolor="#333", linewidth=0.8)
    ax.set_ylabel("最大稳定时间步长 (ms)")
    ax.set_title("积分器稳定性对比（无爆炸的最大 dt）")
    for bar, val in zip(bars, max_dt):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.2f}", ha="center", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_performance_bar(cases: list[dict], out_path: Path):
    labels = [c["label"].replace(" (fixed-point)", "") for c in cases]
    ms = [c["ms_per_step"] for c in cases]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    bars = ax.bar(labels, ms, color=["#e74c3c", "#3498db", "#2ecc71"], edgecolor="#333", linewidth=0.8)
    ax.set_ylabel("单步耗时 (ms)")
    ax.set_title("三种积分器单步计算性能对比")
    for bar, val in zip(bars, ms):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def find_max_stable_dt(solver: int, base_dt: float = 1.0 / 240.0) -> float:
    dt = base_dt
    while dt < base_dt * 32:
        c = measure_case(solver, steps=120, dt=dt, damping=1.0, ks_scale=1.5)
        if c["exploded"]:
            break
        dt *= 1.25
    return dt / 1.25


def main():
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    ti.init(arch=ti.cpu)

    solvers = [SOLVER_EXPLICIT, SOLVER_SEMI_IMPLICIT, SOLVER_IMPLICIT]
    damping_cases = []
    for s in solvers:
        damping_cases.append(measure_case(s, steps=400, dt=1.0 / 240.0, damping=1.0))
    plot_energy_curves(damping_cases, CHART_DIR / "energy_damping_1.0.png", "damping=1.0 时总动能变化")
    plot_max_velocity(damping_cases, CHART_DIR / "max_velocity_damping_1.0.png")
    plot_com_height(damping_cases, CHART_DIR / "com_height.png")

    damping5 = []
    for s in solvers:
        damping5.append(measure_case(s, steps=400, dt=1.0 / 240.0, damping=5.0))
    plot_energy_curves(damping5, CHART_DIR / "energy_damping_5.0.png", "damping=5.0 时总动能变化")

    perf = [measure_case(s, steps=200, dt=1.0 / 240.0, damping=2.5) for s in solvers]
    plot_performance_bar(perf, CHART_DIR / "performance.png")

    stability = []
    for s in solvers:
        stability.append({"label": SOLVER_NAME[s], "max_stable_dt": find_max_stable_dt(s)})
    plot_stability_bar(stability, CHART_DIR / "stability.png")

    # 复制图表到 assets 便于报告引用
    for p in CHART_DIR.glob("*.png"):
        target = ASSET_DIR / p.name
        target.write_bytes(p.read_bytes())
        print(f"[OK] chart -> {target}")

    csv_path = CHART_DIR / "benchmark_summary.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("solver,damping,exploded,ms_per_step,final_ke,final_max_v\n")
        for c, d in [(c, 1.0) for c in damping_cases] + [(c, 5.0) for c in damping5]:
            f.write(
                f"{c['label']},{d},{c['exploded']},{c['ms_per_step']:.4f},"
                f"{c['ke'][-1]:.4f},{c['max_v'][-1]:.4f}\n"
            )
    print(f"[OK] summary -> {csv_path}")
    print("=== Chart export finished ===")


if __name__ == "__main__":
    main()
