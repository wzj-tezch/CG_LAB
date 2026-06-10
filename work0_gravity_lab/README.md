# Taichi 引力粒子群仿真（Work0）

**课程对应**：Work0 / 课程 `work1`  
**目录**：`work0_gravity_lab/`

![Work0 演示](https://github.com/user-attachments/assets/6717c6be-05a2-4fa7-bf48-79086b92e6b9)

## 项目简介

基于 Taichi 框架实现的万有引力粒子群仿真，使用 GPU 并行计算模拟大量粒子在鼠标引力场作用下的运动效果。

## 项目结构

```
work0_gravity_lab/
├── pyproject.toml
├── src/
│   └── Work0/
│       ├── config.py      # 可调参数
│       ├── physics.py     # GPU 并行物理逻辑
│       └── main.py        # GUI 入口
├── .python-version
└── uv.lock
```

## 安装与运行

```bash
cd work0_gravity_lab
uv sync
uv run -m src.Work0.main
```

## 功能说明

- **粒子系统**：10000 个粒子的随机初始化
- **鼠标交互**：鼠标位置产生引力场，粒子会被吸引
- **物理模拟**：引力、阻力与边界碰撞
- **GPU 加速**：Taichi 并行计算
- **实时渲染**：实时显示粒子运动

## 参数配置

在 `src/Work0/config.py` 中可调整：

- `NUM_PARTICLES`：粒子总数（卡顿可调小，如 2000）
- `GRAVITY_STRENGTH`：鼠标引力强度
- `DRAG_COEF`：空气阻力系数
- `BOUNCE_COEF`：边界反弹能量损耗
- `WINDOW_RES`：窗口分辨率
- `PARTICLE_RADIUS`：粒子绘制半径
- `PARTICLE_COLOR`：粒子颜色

## 依赖

- Python >= 3.12
- taichi >= 1.7.4

## 注意事项

- 首次运行会编译 GPU 内核，需稍等片刻
- 移动鼠标可观察粒子群动态效果
