# 贝塞尔曲线实验（Python + Taichi）

- **授课教师**：张鸿文  
- **助教**：张怡冉  
- **课程主页**：https://zhanghongwen.cn/cg  

## 动图演示

![贝塞尔曲线演示：控制多边形（灰线）、曲线（绿色光栅化）、控制点（红圆）](assets/bezier_demo.gif)

动图由仓库内 [`export_demo_gif.py`](export_demo_gif.py) 生成。

## 静态参考效果图

（与实验说明中的参考效果一致；因飞书文档常无法外链展示，此处用脚本按同一视觉规范绘制。）

![参考效果：控制点、控制多边形、光栅化贝塞尔曲线](assets/reference_effect.png)

生成方式：`pip install pillow` 后执行 `python export_reference_figure.py`，输出 `assets/reference_effect.png`。

## 内容说明

使用 De Casteljau 算法在 CPU 端采样贝塞尔曲线，将采样点批量传入 GPU，通过 `@ti.kernel` 并行完成光栅化（点亮 `pixels` 帧缓冲）。控制点通过 GGUI 绘制，支持控制多边形（灰线）与曲线（绿色）叠加显示。

## 环境

- Python 3.8+（推荐 3.10+）
- Windows / Linux / macOS，需支持 Taichi 所选后端（GPU 不可用时自动回退 CPU）

## 安装与运行

```bash
pip install -r requirements.txt
python bezier_taichi.py
```

## 交互

| 操作 | 说明 |
|------|------|
| **鼠标左键** | 在画布上添加控制点（红色圆点） |
| **键盘 C** | 清空所有控制点并重置画面 |

当控制点不少于 2 个时，自动用灰色折线连接控制点（控制多边形），并实时绘制绿色贝塞尔曲线。

## 参数（与实验要求一致）

- 画布分辨率：800×800  
- 曲线采样段数：`NUM_SEGMENTS = 1000`（共 1001 个采样点）  
- 最大控制点数：100  
