# 光线追踪实验（Whitted-Style, Taichi）

本实验实现了可交互的 Whitted-Style 光线追踪器，包含必做 + 选做功能：

- 必做：硬阴影、理想镜面反射、迭代弹射、交互式光源与弹射次数
- 选做 1：玻璃折射（Snell）+ 全反射（TIR）
- 选做 2：MSAA 多重采样抗锯齿

## 演示录制

> 按你的录屏文件 `D:\桌面\屏幕录制 2026-04-30 165037.gif` 拷贝到仓库：
> `ray_tracing_lab/assets/ray_tracing_demo.gif`
>
> 按你的选做录屏 `D:\桌面\屏幕录制 2026-04-30 165927.gif` 拷贝到仓库：
> `ray_tracing_lab/assets/optional_features_demo.gif`

| 必做功能演示 | 选做功能演示 |
|---|---|
| ![ray tracing demo](assets/ray_tracing_demo.gif)<br/>硬阴影 + 镜面反射 + 迭代弹射 + 基础交互。 | ![optional features demo](assets/optional_features_demo.gif)<br/>玻璃折射/全反射 + MSAA 抗锯齿效果。 |

## 1. 环境与运行

```bash
pip install -r requirements.txt
python ray_tracing_taichi.py
```

建议 Python 3.8+，Taichi 1.7+。

## 2. 场景配置（隐式几何体）

代码中未导入任何外部模型，全部由解析几何定义：

- 无限平面（Ground Plane）：`y = -1.0`，法线 `(0, 1, 0)`，黑白棋盘格纹理
- 玻璃球（Glass）：中心 `(-1.5, 0, 0)`，半径 `1.0`，折射率 `IOR = 1.5`
- 银色镜面球（Mirror）：中心 `(1.5, 0, 0)`，半径 `1.0`

材质系统：

- `MAT_DIFFUSE = 0`
- `MAT_MIRROR = 1`
- `MAT_GLASS = 2`

## 3. 迭代式光线弹射

每个像素中使用固定上限循环（`MAX_BOUNCES_CAP = 5`）+ 运行时 `max_bounces` 控制：

- `throughput` 初始为 `1.0`
- `final_color` 初始为 `0.0`
- 命中镜面：更新反射方向与新起点，`throughput *= 0.8`，继续弹射
- 命中玻璃：按斯涅尔定律求折射方向，若无实解则触发全反射
- 命中漫反射：计算光照并累积到 `final_color`，随后终止该像素路径

反射向量实现：

$$
\mathbf{R} = \mathbf{L}_{in} - 2(\mathbf{L}_{in}\cdot\mathbf{N})\mathbf{N}
$$

## 4. 硬阴影与自相交修复（Shadow Acne）

在暗影射线和反射射线中均使用法线偏移：

$$
\mathbf{P}_{new} = \mathbf{P} + \epsilon \mathbf{N},\quad \epsilon=10^{-4}
$$

这样可避免射线与自身表面立即相交导致的黑斑噪点。

## 5. 选做：折射与全反射（已实现）

玻璃材质采用以下策略：

- 外部入射：`eta = eta_air / eta_glass`
- 内部出射：翻转法线并交换介质折射率
- 当 $1 - \eta^2(1-\cos^2\theta_i) < 0$ 时判定为全反射，退化为镜面反射方向

## 6. 选做：MSAA 抗锯齿（已实现）

在每个像素内进行 `spp` 次随机抖动采样：

- 子像素抖动：`jx,jy ~ U(-0.5,0.5)`
- 发射多条主光线
- 颜色求平均写回像素

程序中 `MSAA Samples` 滑块支持 `1~8`。

## 7. UI 交互项

通过 `ti.ui.Window` + `gui.sub_window` 提供实时滑动条：

- `Light X`
- `Light Y`
- `Light Z`
- `Max Bounces`（1~5，默认 3）
- `MSAA Samples`（1~8，默认 4）

你可以直接观察：

- 光源位置变化导致阴影实时移动
- `Max Bounces` 增加时镜面与玻璃内部路径更丰富
- `MSAA Samples` 增加时边缘锯齿明显减轻

## 8. 对应作业要求映射

- 任务 1（场景与材质 ID）：已完成
- 任务 2（迭代追踪 + throughput）：已完成
- 任务 3（硬阴影 + epsilon 偏移）：已完成
- 任务 4（UI 滑动条）：已完成
- 选做 1（玻璃折射 + 全反射）：已完成
- 选做 2（MSAA 抗锯齿）：已完成

## 9. 提交建议

- 提交目录：`CG_LAB/ray_tracing_lab/`
- 提交文件至少包含：
  - `ray_tracing_taichi.py`
  - `README.md`
  - `requirements.txt`
  - `assets/ray_tracing_demo.gif`
  - `assets/optional_features_demo.gif`

将仓库推送到 Git 平台（如 GitHub），在课程系统提交仓库链接即可。
