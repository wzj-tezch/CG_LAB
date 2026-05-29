# SMPL LBS 蒙皮可视化实验

基于官方 **SMPL_NEUTRAL** 模型与 **smplx**，分阶段可视化线性混合蒙皮 (LBS) 全流程，并验证手写实现与官方前向一致性。

| 项目 | 内容 |
|------|------|
| 学生 | 武子杰（202411081003） |
| 完整报告 | [202411081003-武子杰-SMPL_LBS实验报告.md](202411081003-武子杰-SMPL_LBS实验报告.md) |
| 仓库 | [wzj-tezch/CG_LAB](https://github.com/wzj-tezch/CG_LAB/tree/main/smpl_lbs_lab) |

## 效果预览

### 四阶段 LBS 对比（正视图）

![四阶段对比](assets/comparison_preview.png)

### 姿态动画（选做）

![姿态动画](assets/pose_animation.gif)

## 快速开始

```bash
pip install -r requirements.txt
# 1. 从 SMPLify 官网下载 SMPL_NEUTRAL.pkl（需注册）
# 2. 转换格式
python scripts/convert_smpl_pkl.py path/to/SMPL_NEUTRAL.pkl models/smpl/SMPL_NEUTRAL.pkl
# 3. 运行实验
python run_experiment.py
python export_charts.py   # 可选：补充数据图表
```

> **注意**：`models/smpl/*.pkl` 受 SMPL 许可协议约束，**不随仓库分发**。请自行从 [smplify.is.tue.mpg.de](https://smplify.is.tue.mpg.de/) 或课程云盘获取。

## 输出目录

```
outputs/
├── stage_a_template_weights.png   # 模板 + 单关节权重
├── all_joint_weights.png          # 全关节主导权重
├── stage_b_shaped_joints.png      # 形状校正 + 关节
├── stage_c_pose_offsets.png       # 姿态偏移
├── stage_d_lbs_result.png         # 最终 LBS
├── comparison_grid.png            # 2×2 总对比
├── summary.txt                    # 误差与模型信息
└── charts/                        # 数据图表
assets/
└── pose_animation.gif             # 选做动画
```

## 目录结构

| 文件 | 说明 |
|------|------|
| `run_experiment.py` | 任务 1–7 + 动画主入口 |
| `manual_lbs.py` | 手写 LBS，暴露五阶段中间量 |
| `visualize.py` | 正视图网格渲染 |
| `export_charts.py` | 补充统计图表 |
| `scripts/convert_smpl_pkl.py` | chumpy → numpy 模型转换 |
