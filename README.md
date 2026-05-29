# CG_LAB

计算机图形学课程实验仓库（[课程主页](https://zhanghongwen.cn/cg)）。

## 各次作业目录

* `week2_upload_package/`：Week2 MVP
* `bezier_lab/`：贝塞尔曲线实验（Python + Taichi，De Casteljau 与 GPU 光栅化）
* `phong_lab/`：**Phong 光照模型**（Taichi 光线投射、球与圆锥隐式求交、深度竞争、`ti.ui` 滑动条实时调参）。完整说明与实验报告见 **[phong_lab/README.md](phong_lab/README.md)**。
* `mass_spring_lab/`：**质点弹簧模型**（布料仿真、三种数值积分器对比、剪切/弯曲弹簧、球体碰撞、动态风场）。完整说明与实验报告见 **[mass_spring_lab/README.md](mass_spring_lab/README.md)**。
* `work1_pytorch3d_lab/`：**PyTorch3D 可微渲染网格优化实验（球体 -> 奶牛）**。本仓库中 Work1 的唯一正式目录，完整实验报告见 **[work1_pytorch3d_lab/README.md](work1_pytorch3d_lab/README.md)**。
* `cloth_sim_taichi/`：**Taichi 布料仿真（质点-弹簧 + 三种积分器 + GGUI 交互）**。完整实验报告见 **[cloth_sim_taichi/README.md](cloth_sim_taichi/README.md)**。
* `smpl_lbs_lab/`：**SMPL 线性混合蒙皮 (LBS) 可视化**（四阶段正视图、手写 LBS 验证、姿态动画）。完整实验报告见 **[smpl_lbs_lab/README.md](smpl_lbs_lab/README.md)**。
* `theory_homework/`：**理论课四次作业**（基础知识 / 几何 / 渲染 / 动画），含完整答案 `.md` 与 `.docx`。见 **[theory_homework/README.md](theory_homework/README.md)**。

## 效果预览

### 布料仿真交互（`cloth_sim_taichi`）

![布料仿真交互预览](cloth_sim_taichi/assets/interactive_demo_20260521.gif)

### SMPL LBS 蒙皮可视化（`smpl_lbs_lab`）

![SMPL LBS 四阶段对比](smpl_lbs_lab/assets/comparison_preview.png)

### Phong 光照（`phong_lab`）

动图源文件：`D:\桌面\屏幕录制 2026-04-16 181624.gif`（仓库内副本见 `phong_lab/assets`）。

![Phong 预览](<phong_lab/assets/屏幕录制 2026-04-16 181624.gif>)

### Week2 MVP

动图路径：`week2_upload_package/assets/week2/mvp_demo.gif`（若存在）。

## 目录补充说明

* `week2_upload_package/week2/`：MVP 实验代码（含三角形与立方体线框渲染）
* `bezier_lab/`：贝塞尔相关代码与说明见该目录内文档
* `work1_pytorch3d_lab/`：包含 Work1 的源码、环境文件、实验结果与提交说明；评审与复现请以该目录为准
* `cloth_sim_taichi/`：包含布料仿真源码、依赖、实验报告与交互录屏素材
* `smpl_lbs_lab/`：包含 SMPL LBS 源码、实验报告、四阶段可视化图、数据图表与姿态动画（模型文件需自行下载）
