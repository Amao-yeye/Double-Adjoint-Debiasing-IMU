# PLATO

**PLATO: Privileged Learning from Accurate Trajectory Observations for IMU Bias Dynamics**

PLATO 是一个面向视觉惯性里程计研究的本地工作区。项目目标是利用视觉条件较好时得到的高质量轨迹观测，作为 privileged supervision 来学习或估计 IMU 偏置动力学，重点关注陀螺仪偏置 `b_g` 和加速度计偏置 `b_a`。学习到的偏置模型或校正后的 IMU 数据随后用于 OpenVINS 等 VIO 系统，评估其在视觉退化、光照变化或图像缺失场景下是否能改善 IMU 传播和整体轨迹精度。

当前代码已经覆盖以下研究链路：

1. 使用 ORB-SLAM3 在视觉条件较好的数据段生成 stereo VO / pseudo-groundtruth 候选轨迹。
2. 对轨迹进行时间、坐标系、重力方向和相机-IMU 外参处理，导出 Debias_IMU 可用的 IMU/body supervision。
3. 基于外部 Debias_IMU 代码训练和评估 Neural ODE / SE_2(3) / NLL 等偏置动力学变体。
4. 导出 raw、constant-bias、network-debiased 或 gated residual IMU 流。
5. 在 OpenVINS 中做 clean-IMU、visual-dropout 和自然光照退化评估。

## 顶层结构

```text
PLATO/
├── README.md                 # 本文件：项目入口和结构索引
├── .gitignore                # 忽略外部仓库、数据集、结果、构建产物等
├── .gitmodules               # paper/plato-overleaf Overleaf 子模块
├── Instructions/             # 迁移后的任务说明、历史路线图和 agent 说明
├── plato/                    # PLATO 自有代码、配置、文档和补丁
├── tools/                    # 少量仓库级预处理工具
├── paper/                    # 论文 PDF 和 Overleaf 子模块
├── external/                 # 外部研究代码，本地 nested git worktrees
├── baseline_code_original/   # 原始 Debias_IMU 基线代码快照
├── datasets/                 # 本地数据集占位目录，通常只放 symlink/说明
├── results/                  # 生成结果、pseudo-GT、训练输出和评估输出
└── experiments/              # 实验占位目录，输出应保持 ignored
```

## `plato/` 子结构

`plato/` 是本项目的主要维护边界。新写的 PLATO 专用脚本、配置和文档应优先放在这里，避免直接污染外部仓库。

```text
plato/
├── configs/
│   ├── debias_imu/           # Debias_IMU 训练与消融 YAML 配置
│   ├── openvins/             # LIASuw/corldata OpenVINS 配置
│   ├── orbslam3/             # ORB-SLAM3 数据集和调参配置
│   └── rviz/                 # RViz 可视化配置
├── docs/                     # 项目记忆、数据说明、实验状态和流程文档
├── patches/
│   └── external/             # 外部仓库本地修改补丁记录
└── scripts/
    ├── orbslam3_export/      # ORB-SLAM3 轨迹导出、对齐、评估和可视化
    ├── train_bias_dynamics/  # Debias_IMU 训练包装器、诊断和偏置分析
    ├── openvins_eval/        # OpenVINS 输入导出、bag 生成、运行和汇总
    └── *.py                  # 数据集导出、轨迹转换、标定和通用流程脚本
```

重点入口：

- `plato/docs/repository_map.md`：最完整的仓库结构和脚本用途说明。
- `plato/docs/project_goal.md`：研究目标和技术路线。
- `plato/docs/dataset_guide.md`：本机数据集路径、命名和注意事项。
- `plato/docs/current_experiment_state.md`：当前 Debias_IMU / OpenVINS 实验状态。
- `plato/docs/pseudo_groundtruth_registry.md`：pseudo-groundtruth 候选、接受和拒绝记录。
- `plato/docs/project_memory.md`：项目长期记忆和重要决策。

## 主要脚本分区

### ORB-SLAM3 轨迹与 pseudo-GT

路径：`plato/scripts/orbslam3_export/`

这一组脚本负责从 LIASuw、corldata 等数据集中导出 stereo image/IMU 序列，调用 ORB-SLAM3，搜索时间偏移，做 SE(3) no-scale 对齐，生成指标、图表和交互式 HTML 轨迹检查。

代表性脚本：

- `run_liasuw_orbslam3_trial.py`：LIASuw stereo / stereo-inertial ORB-SLAM3 运行与评估。
- `run_corldata_evo_pseudogt_workflow.py`：当前 corldata pseudo-GT 生成和 evo 对齐评估主流程。
- `diagnose_corldata_gt_quality.py`：corldata mocap GT 质量诊断。
- `detect_corldata_light_segments.py`：corldata 自然光照开关区间检测。
- `make_corldata_3d_html.py`：生成 GT 与 VO 对比的可拖拽 3D/投影视图。

### IMU/body supervision 导出

路径：`plato/scripts/`

这些脚本把 ORB-SLAM3 或 mocap 轨迹转换为 Debias_IMU 可训练的数据格式，重点处理相机、IMU/body、mocap marker、世界系和重力方向之间的关系。

代表性脚本：

- `export_liasuw_debias_body_gt.py`：导出 LIASuw 200 Hz body GT 和聚合 TUM-VI 风格数据树。
- `export_corldata_debias_body_pseudogt.py`：导出 corldata VO teacher 轨迹到 IMU/body pseudo-GT。
- `export_corldata_imu_aligned_gt_from_extrinsics.py`：基于 Body_0-to-IMU 外参导出 corldata IMU-aligned GT。
- `generate_imu_gt_from_mocap.py`、`convert_vo_to_imu.py`：通用 mocap/VO 到 IMU/body 轨迹转换工具。

### Debias_IMU 训练与诊断

路径：`plato/scripts/train_bias_dynamics/`

PLATO 通过这一层复用 `external/Debias_IMU/rss_debiasing_restart` 的核心模型和训练代码，同时把数据集 glue code、实验配置、NLL/SE_2(3) 扩展、可视化和诊断保持在 `plato/` 下。

代表性脚本：

- `run_liasuw_debias_ad_pipeline.py`：LIASuw 自动微分 Debias_IMU 训练/测试主流程。
- `run_corldata_gt_supervision_experiment.py`：corldata 配置驱动训练包装器。
- `run_euroc_debias_phys_delta_pipeline.py`：EuRoC 偏置动力学实验包装器。
- `m2_bias_networks.py`：PLATO 自有的 M2 no-derivative 偏置网络。
- `bias_trend_utils.py` 及 `compare_*_predicted_bias_*`：预测偏置与参考偏置趋势诊断。
- `check_corldata_preintegration.py`、`plot_liasuw_bias_preintegration_window.py`：短窗口预积分一致性检查。

### OpenVINS 评估

路径：`plato/scripts/openvins_eval/`

这一组脚本负责把 Debias_IMU 输出转换成 OpenVINS 可用的 IMU 流或 ROS bag，并汇总 clean-IMU、视觉 dropout、自然光照退化和 re-entry 相关结果。

代表性脚本：

- `export_liasuw_openvins_inputs.py`：导出 LIASuw raw / constant / network-debiased IMU CSV。
- `make_liasuw_openvins_bags.py`、`make_visual_dropout_bags.py`：生成 OpenVINS 输入 bag。
- `run_openvins_dropout_trial.sh`：运行视觉 dropout 场景下的 OpenVINS 试验。
- `summarize_liasuw_openvins_dropout.py`：汇总 LIASuw dropout 传播误差。
- `export_corldata_openvins_corrected_inputs.py`、`export_corldata_openvins_gated_residual_inputs.py`：导出 corldata 校正 IMU 流。
- `summarize_corldata_openvins_natural.py`、`compare_corldata_openvins_natural_runs.py`：汇总和比较 corldata 自然光照 OpenVINS 结果。

## 外部项目

`external/` 下保存本地外部研究代码。它们是 PLATO 依赖的执行后端，不是 PLATO 自有代码的默认修改位置。

```text
external/
├── ORB_SLAM3/                         # stereo / stereo-inertial SLAM 与 VO
├── Debias_IMU/rss_debiasing_restart/  # Neural ODE IMU bias dynamics 代码
└── OpenVINS/                          # 下游 VIO 评估系统
```

约定：

- 新增 PLATO glue code 时优先放入 `plato/scripts/`、`plato/configs/` 或 `plato/docs/`。
- 只有在任务确实需要修改外部仓库时才改 `external/`。
- 外部仓库本地修改应记录在 `plato/docs/project_memory.md`，并尽量同步补丁到 `plato/patches/external/`。

## 数据、结果和论文

- `datasets/`：仓库内占位目录。大型数据不应复制进 git；本机默认数据根目录是 `/home/amao/datasets`，详见 `plato/docs/dataset_guide.md`。
- `results/`：被 `.gitignore` 忽略的结果根目录，包含 `results/pseudo_gt/`、`results/debias_imu/`、`results/openvins_eval/` 等。
- `experiments/`：实验占位目录，具体输出应保持 ignored。
- `paper/`：论文相关文件。`paper/plato-overleaf` 是 Overleaf git 子模块，代码、数据和实验输出不要放入论文子模块。
- `baseline_code_original/`：原始 Debias_IMU 基线代码快照，用于对照和追溯。
- `Instructions/`：历史任务计划、路线图和迁移后的说明文件。

## 数据集约定

当前本机使用的数据根目录：

```text
/home/amao/datasets
```

当前主要数据集：

- EuRoC MAV：`/home/amao/datasets/EuRoC`
- LIASuw：`/home/amao/datasets/LIASuw`
- corldata：`/home/amao/datasets/corldata`

推荐脚本通过 CLI 参数或环境变量读取数据根目录：

```bash
export PLATO_DATASET_ROOT=/home/amao/datasets
```

不要把大型数据、ROS bag、训练输出、评估结果或 LaTeX 构建产物提交到仓库。

## 推荐阅读顺序

新任务开始前建议先看：

1. `plato/docs/project_goal.md`
2. `plato/docs/repository_map.md`
3. `plato/docs/dataset_guide.md`
4. `plato/docs/current_experiment_state.md`
5. `plato/docs/pseudo_groundtruth_registry.md`
6. 相关脚本子目录下的 `README.md`

如果任务涉及当前实验状态或历史决策，再查 `plato/docs/project_memory.md` 和 `Instructions/AGENTS.md`。

