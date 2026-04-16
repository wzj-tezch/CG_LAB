# Phong 光照模型 — 计算机图形学实验

**课程主页：** [https://zhanghongwen.cn/cg](https://zhanghongwen.cn/cg)  

**授课教师：** 张鸿文　**助教：** 张怡冉  

**姓名：** 武子杰　**学号：** 202411081003　**日期：** 2026 年 4 月 16 日  

---

## 一、实验目的

1. **理论：** 理解局部光照模型，区分环境光（Ambient）、漫反射（Diffuse）与镜面高光（Specular）的物理含义与在画面中的作用。  
2. **数学：** 熟悉三维向量运算，包括法向量、光线方向、视线方向与反射向量 **R** 的构造与归一化。  
3. **工程：** 使用 Taichi 完成光线投射（Ray Casting）、隐式几何求交与深度竞争，并通过 `ti.ui.Window` 滑动条实时调节 Phong 系数，观察参数对渲染结果的影响。

---

## 二、实验原理

Phong 将表面亮度分解为三项并相加：

$$
I = I_{ambient} + I_{diffuse} + I_{specular}
$$

- **环境光：** $I_{ambient} = K_a \cdot C_{light} \cdot C_{object}$  
- **漫反射（Lambert）：** $I_{diffuse} = K_d \cdot \max(0, \mathbf{N}\cdot\mathbf{L}) \cdot C_{light} \cdot C_{object}$  
- **镜面高光：** $I_{specular} = K_s \cdot \max(0, \mathbf{R}\cdot\mathbf{V})^n \cdot C_{light}$，其中 $\mathbf{R} = 2(\mathbf{N}\cdot\mathbf{L})\mathbf{N}-\mathbf{L}$，$n$ 为高光指数 Shininess。

实现中 $\mathbf{N},\mathbf{L},\mathbf{V}$ 均取单位向量；点积为负时用 `max(0, …)` 截断；输出前对 RGB 做 **clamp 到 [0,1]**，避免过曝。

---

## 三、实验环境

| 项目 | 说明 |
|------|------|
| 操作系统 | Windows 10/11 |
| Python | 3.x（建议 3.10+） |
| 主要依赖 | `taichi`、`numpy`、`imageio` |
| 运行交互程序 | `python phong_raycast.py`（需支持 GGUI 的 GPU 后端，如 Vulkan） |
| 生成预览 GIF | `python export_preview_gif.py`（CPU 后端，无窗口） |

安装依赖：

```bash
cd phong_lab
pip install -r requirements.txt
```

---

## 四、实验内容与实现要点

### 4.1 场景（隐式几何）

- **红色球体：** 球心、半径与颜色在 `phong_raycast.py` 中常量定义，光线与球求二次方程正根。  
- **紫色圆锥：** 顶点、轴方向、高度与底面半径定义有限锥体；侧面为圆锥二次曲面与高度条带求交；**底面圆盘**单独求交，避免锥体「漏底」。  
- **相机与光源：** 相机位于 $(0,0,5)$，向 $z=0$ 附近成像平面投射透视射线；点光源位置与颜色为常量；背景为深青色。

### 4.2 深度测试（Z 竞争）

对每个像素，对球、锥侧面、锥底分别计算最近正交距离 **t**，取**最小的正 t**作为可见表面，等价于光线意义下的 Z-buffer，保证遮挡正确。

### 4.3 法向量与着色

- 球：**N** 为球心指向表面点的单位向量。  
- 锥面：由隐式梯度得到法向并归一化；若 **N·V < 0** 则翻转法向，减轻错误背面光照。  
- 在命中点计算 **L、V、R** 后按 Phong 三项叠加。

### 4.4 交互界面

使用 `ti.ui.Window` 与 `slider_float` 绑定 **Ka、Kd、Ks、Shininess**（ImGui 默认字体对中文支持有限，控件标签使用英文）。参数范围与作业要求一致。

---

## 五、效果预览（GIF）

下图由 `export_preview_gif.py` 在无窗口模式下循环改变 **Ka / Kd / Ks / Shininess** 并导出。若仓库中缺少该文件，在本目录执行 `python export_preview_gif.py` 重新生成。

![Phong 光照效果预览（参数随时间循环调节）](assets/preview.gif)

---

## 六、参数调节观察（简要）

| 参数 | 现象 |
|------|------|
| **Ka** 增大 | 整体变亮，背光面与阴影区仍可见基础亮度，画面更「平」。 |
| **Kd** 增大 | 漫反射增强，物体固有色更明显，明暗对比主要由 **N·L** 决定。 |
| **Ks** 增大 | 镜面项变强，高光更亮、更「油」。 |
| **Shininess** 增大 | 高光更集中、边缘更锐利；减小则高光发散、更柔和。 |

---

## 七、总结

本实验在 Taichi 中完成了从射线生成、多几何体求交与深度竞争，到 Phong 三项叠加与 UI 实时调参的完整链路。实现过程中需注意向量归一化、负点积截断与最终颜色 clamp。若需扩展，可在同框架下加入 **Blinn-Phong**（半程向量 **H**）或**硬阴影**（向光源发射阴影射线）。

---

## 附录：本目录文件

| 文件 | 作用 |
|------|------|
| `phong_raycast.py` | 主程序：渲染 + 交互窗口 |
| `export_preview_gif.py` | 导出 `assets/preview.gif` |
| `requirements.txt` | Python 依赖列表 |
| `assets/preview.gif` | 效果动图（可重新生成） |
