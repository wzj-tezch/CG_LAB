# 光线追踪实验（Whitted-Style, Taichi）

本实验实现了一个可交互的 Whitted-Style 光线追踪器，覆盖以下核心点：

- 光线投射（Ray Casting）与光线追踪（Ray Tracing）的差异
- 次级射线：硬阴影（Shadow Ray）与理想镜面反射（Reflection Ray）
- 将递归追踪改写为 GPU 友好的迭代循环（Bounce Loop）

## 1. 环境与运行

```bash
pip install -r requirements.txt
python ray_tracing_taichi.py
```

建议 Python 3.8+，Taichi 1.7+。

## 2. 场景配置（隐式几何体）

代码中未导入任何外部模型，全部由解析几何定义：

- 无限平面（Ground Plane）：`y = -1.0`，法线 `(0, 1, 0)`，黑白棋盘格纹理
- 红色漫反射球（Diffuse）：中心 `(-1.5, 0, 0)`，半径 `1.0`
- 银色镜面球（Mirror）：中心 `(1.5, 0, 0)`，半径 `1.0`

材质系统：

- `MAT_DIFFUSE = 0`
- `MAT_MIRROR = 1`

## 3. 迭代式光线弹射

每个像素中使用固定上限循环（`MAX_BOUNCES_CAP = 5`）+ 运行时 `max_bounces` 控制：

- `throughput` 初始为 `1.0`
- `final_color` 初始为 `0.0`
- 命中镜面：更新反射方向与新起点，`throughput *= 0.8`，继续弹射
- 命中漫反射：计算光照并累积到 `final_color`，随后终止该像素路径

反射向量实现：

\[
\mathbf{R} = \mathbf{L}_{in} - 2(\mathbf{L}_{in}\cdot\mathbf{N})\mathbf{N}
\]

## 4. 硬阴影与自相交修复（Shadow Acne）

在暗影射线和反射射线中均使用法线偏移：

\[
\mathbf{P}_{new} = \mathbf{P} + \epsilon \mathbf{N},\quad \epsilon=10^{-4}
\]

这样可避免射线与自身表面立即相交导致的黑斑噪点。

## 5. UI 交互项

通过 `ti.ui.Window` + `gui.sub_window` 提供实时滑动条：

- `Light X`
- `Light Y`
- `Light Z`
- `Max Bounces`（1~5，默认 3）

你可以直接观察：

- 光源位置变化导致阴影实时移动
- `Max Bounces = 1` 时几乎无镜面反射
- `Max Bounces > 1` 时镜面球出现“镜中世界”效果

## 6. 对应作业要求映射

- 任务 1（场景与材质 ID）：已完成
- 任务 2（迭代追踪 + throughput）：已完成
- 任务 3（硬阴影 + epsilon 偏移）：已完成
- 任务 4（UI 滑动条）：已完成

## 7. 提交建议

- 提交目录：`CG_LAB/ray_tracing_lab/`
- 提交文件至少包含：
  - `ray_tracing_taichi.py`
  - `README.md`
  - `requirements.txt`

将仓库推送到 Git 平台（如 GitHub），在课程系统提交仓库链接即可。
