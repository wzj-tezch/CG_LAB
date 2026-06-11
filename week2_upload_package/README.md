# Week2 MVP 变换实验

**姓名**：武子杰 | **学号**：202411081003 | **专业**：计算机科学与技术（公费师范）

---

**课程对应**：Week2 / 课程 `work2`  
**目录**：`week2_upload_package/`

## 项目简介

基于 Taichi GUI 的 MVP 管线实验：实现模型、视图、投影矩阵，渲染线框三角形与旋转立方体。

## 项目结构

```
week2_upload_package/
├── pyproject.toml
├── week2/
│   ├── main.py        # GUI 入口
│   ├── mvp.py         # MVP 矩阵
│   ├── geometry.py    # 三角形与立方体顶点/边
│   └── make_gif.py    # 离线 GIF 导出
└── assets/week2/      # 演示 GIF（运行 make_gif 后生成）
```

## 安装与运行

```bash
cd week2_upload_package
uv sync
uv run -m week2.main
```

## 实现内容

- `get_model_matrix(angle)`：绕 Z 轴旋转的模型矩阵
- `get_view_matrix(eye_pos)`：将相机平移到原点的视图矩阵
- `get_projection_matrix(eye_fov, aspect_ratio, zNear, zFar)`：透视 + 正交投影
- 渲染内容：
  - 基础要求：线框三角形
  - 选做内容：线框立方体（8 顶点、12 边）

## 交互按键

- `A`：逆时针旋转
- `D`：顺时针旋转
- `Esc`：退出程序

## 验收标准

- 窗口分辨率 `700x700`
- 可见三角形与立方体线框
- 按 `A/D` 可连续旋转
- 存在透视效果（远小近大）

## 生成 GIF

```bash
uv run -m week2.make_gif
```

输出路径：`assets/week2/mvp_demo.gif`

效果预览（若已生成）：

![week2-mvp-demo](assets/week2/mvp_demo.gif)

## 依赖

- Python >= 3.12
- taichi >= 1.7.4
- imageio >= 2.37.3

## 相关实验

- 前置：[`work0_gravity_lab/`](../work0_gravity_lab/)（Taichi 入门与 GPU 并行）
