# corldata Pseudo-GT Evaluation Workflow

本文档定义 corldata 后续统一使用的 VO pseudo-groundtruth 生成与评估流程。目标是让每次处理新序列都可复现、可审计，并明确区分“ORB-SLAM3 生成 VO”“坐标/时间补偿”“evo 精评”和“可视化”。

## 1. 输入与输出

输入数据：

- 数据根目录：`/home/amao/datasets/corldata/0511_data`
- 序列：以 `.bag` 和 `gt/<seq>/Body_0.tum` 同时存在为准，例如 `S07 S13 S15 S20 S21 S22 S23`
- rosbag：`/home/amao/datasets/corldata/0511_data/<seq>.bag`
- mocap GT：`/home/amao/datasets/corldata/0511_data/gt/<seq>/Body_0.tum`

当前主脚本：

```bash
/home/amao/anaconda3/bin/python plato/scripts/orbslam3_export/run_corldata_evo_pseudogt_workflow.py
```

新序列示例：

```bash
/home/amao/anaconda3/bin/python plato/scripts/orbslam3_export/run_corldata_evo_pseudogt_workflow.py \
  --sequences S07 S13 S15 S20 S21 S22 S23 \
  --summary-prefix S07_S23_evo_gt_aligned
```

当前 3D/投影视图脚本：

```bash
/home/amao/anaconda3/bin/python plato/scripts/orbslam3_export/make_corldata_3d_html.py
```

不传 `--sequences` 时，该脚本会自动发现已有 `evo_gt_aligned` 输出的序列并生成统一 index。

保留输出：

- `results/pseudo_gt/corldata/<seq>/raw/`
- `results/pseudo_gt/corldata/<seq>/sequence/`
- `results/pseudo_gt/corldata/<seq>/alignment_config/`
- `results/pseudo_gt/corldata/<seq>/gt_aligned/`
- `results/pseudo_gt/corldata/<seq>/evo/`
- `results/pseudo_gt/corldata/<seq>/evo_gt_aligned/`
- `results/pseudo_gt/corldata/<summary-prefix>_report.md`
- `results/pseudo_gt/corldata/<summary-prefix>_summary.csv`
- `results/pseudo_gt/corldata/<summary-prefix>_summary.json`
- `results/pseudo_gt/corldata/interactive_3d/`

## 2. 固定原则

- ORB-SLAM3 使用 stereo visual-only VO，保留 loop closure；这是高精度 teacher trajectory，不是严格 online odometry baseline。
- 双目 VO 有物理尺度，因此 evo 只允许 SE(3) 对齐，不使用 `--correct_scale`。
- 最终 world alignment 必须来自 evo 的 `--align`，可视化必须读取 evo 保存的 aligned trajectory。
- `T_C_B_fit` 是每个序列内部拟合的诊断性 camera-to-Body_0 右乘变换，不是跨序列通用物理外参。
- 用户决策：`S13` 及之后的 corldata 序列包含灯开/灯关造成的视觉退化，默认作为验证网络训练结果的 testing data，而不是 training pseudo-GT 候选。
- 不删除 `raw/`、`sequence/`、`logs/` 等原始或可复现实验输入。
- generated results 不进入 git。

## 3. 流程

### Step 1: 生成或复用 ORB-SLAM3 stereo VO

若以下文件已存在且未指定 `--force-raw`，主脚本会直接复用已有 VO：

```text
results/pseudo_gt/corldata/<seq>/raw/estimate_f_seconds.tum
```

若 raw VO 缺失，主脚本会先导出 DAVIS stereo 图像：

```bash
/usr/bin/python3 external/ORB_SLAM3/tools/export_liasuw_bag.py \
  /home/amao/datasets/corldata/0511_data/<seq>.bag \
  results/pseudo_gt/corldata/<seq>/sequence \
  --max-stereo-dt 0.02
```

导出内容：

- `sequence/left/`
- `sequence/right/`
- `sequence/timestamps.txt`

然后运行 ORB-SLAM3 stereo visual-only：

```bash
external/ORB_SLAM3/Examples/Stereo/stereo_tum_vi_batch \
  external/ORB_SLAM3/Vocabulary/ORBvoc.txt \
  plato/configs/orbslam3/corldata_davis_liasuw_0205_baseline.yaml \
  results/pseudo_gt/corldata/<seq>/sequence/left \
  results/pseudo_gt/corldata/<seq>/sequence/right \
  results/pseudo_gt/corldata/<seq>/sequence/timestamps.txt \
  plato_corldata_<seq>_stereo_vo
```

保存：

```text
results/pseudo_gt/corldata/<seq>/raw/estimate_f.txt
results/pseudo_gt/corldata/<seq>/raw/estimate_kf.txt
results/pseudo_gt/corldata/<seq>/raw/estimate_f_seconds.tum
results/pseudo_gt/corldata/<seq>/raw/export.log
results/pseudo_gt/corldata/<seq>/raw/orbslam3.log
```

`estimate_f_seconds.tum` 是后续评估统一使用的 VO 轨迹。

可选参数：

- `--force-raw`：即使 raw VO 已存在，也重新导出 bag 并重跑 ORB-SLAM3。
- `--skip-orb`：只评估已有 raw VO；若缺失则报错。

### Step 2: 读取轨迹

读取：

- VO：`raw/estimate_f_seconds.tum`
- GT：`gt/<seq>/Body_0.tum`

TUM 格式约定：

```text
timestamp_s x y z qx qy qz qw
```

### Step 3: 自动搜索时间偏移

对每个候选 `time_offset_s`，按时间关联 GT 和 VO：

```text
|t_gt - (t_vo + offset)| <= max_time_difference_s
```

默认：

```text
max_time_difference_s = 0.03
```

搜索策略：

- 优先从 `alignment_config/vo_gt_alignment.json` 读取旧 offset。
- 其次从 `evo_gt_aligned/metrics.json` 读取旧 offset。
- 若没有历史 offset，以 `1.0 s` 为中心搜索，默认半径 `0.35 s`。
- coarse 搜索后，在最佳值附近 fine 搜索。

每个 offset 的评分只用于时间对齐：

```text
p_est_se3 = R_align * p_vo + t_align
e_i = ||p_est_se3_i - p_gt_i||_2
score = sqrt((1/N) * sum_i e_i^2)
```

这里使用 SE(3) 位置对齐，不估计尺度。

输出：

```text
results/pseudo_gt/corldata/<seq>/alignment_config/time_offset_scan.csv
```

### Step 4: 拟合诊断性 camera/body 右乘变换

拟合模型：

```text
T_M_B_gt ~= T_M_V_fit * T_V_C_vo * T_C_B_fit
```

符号：

- `M`：mocap/world frame
- `V`：ORB-SLAM3 VO world frame
- `C`：ORB-SLAM3 left camera frame
- `B`：mocap `Body_0` frame
- `T_C_B_fit`：拟合出的 camera-to-Body_0 右乘刚体变换
- `T_M_V_fit`：诊断性的 world alignment

保存：

```text
results/pseudo_gt/corldata/<seq>/alignment_config/vo_gt_alignment.json
```

注意：

```text
camera_to_body_translation_norm_m = ||translation(T_C_B_fit)||
```

它不是轨迹误差，只是拟合出的右乘平移长度。若不同序列差异很大，说明它吸收了 frame convention、时间误差、GT 局部问题或 VO/GT 残差，不能当成真实标定外参。

### Step 5: 生成修正后的 GT

用拟合出的 `T_B_C_fit = inv(T_C_B_fit)` 把 mocap body GT 转成 camera-frame GT：

```text
T_M_C_gt = T_M_B_gt * T_B_C_fit
```

保存：

```text
results/pseudo_gt/corldata/<seq>/gt_aligned/groundtruth_camera_frame.tum
```

这一步只修正 GT 的 frame，使 GT 和 VO 都表达为 camera pose 语义；最终 world alignment 仍交给 evo。

### Step 6: evo 对齐与评估

调用 `evo_traj` 保存 evo 对齐后的轨迹：

```bash
evo_traj tum \
  results/pseudo_gt/corldata/<seq>/raw/estimate_f_seconds.tum \
  --ref results/pseudo_gt/corldata/<seq>/gt_aligned/groundtruth_camera_frame.tum \
  --sync \
  --t_offset <time_offset_s> \
  --t_max_diff 0.03 \
  --align \
  --plot_mode xyz \
  --save_as_tum
```

重命名并保存：

```text
results/pseudo_gt/corldata/<seq>/evo/estimate_evo_aligned.tum
results/pseudo_gt/corldata/<seq>/evo/groundtruth_camera_frame_evo.tum
```

调用 `evo_ape` 做平移和角度统计：

```bash
evo_ape tum <gt_camera_frame.tum> <estimate_f_seconds.tum> \
  -r trans_part \
  --align \
  --t_offset <time_offset_s> \
  --t_max_diff 0.03
```

```bash
evo_ape tum <gt_camera_frame.tum> <estimate_f_seconds.tum> \
  -r angle_deg \
  --align \
  --t_offset <time_offset_s> \
  --t_max_diff 0.03
```

禁止加入：

```bash
--correct_scale
```

### Step 7: 生成 aligned CSV 和 metrics

读取 evo 保存的 aligned estimate / aligned GT：

```text
estimate_evo_aligned.tum
groundtruth_camera_frame_evo.tum
```

重新按时间关联后，逐帧计算位置误差：

```text
e_pos_i = ||p_est_aligned_i - p_gt_aligned_i||_2
pos_rmse = sqrt((1/N) * sum_i e_pos_i^2)
```

角度误差：

```text
R_err_i = R_est_aligned_i^T * R_gt_aligned_i
e_rot_i = acos(clip((trace(R_err_i) - 1) / 2, -1, 1)) * 180 / pi
rot_rmse = sqrt((1/N) * sum_i e_rot_i^2)
```

同时记录：

- `position_mean_m`
- `position_median_m`
- `position_p90_m`
- `position_p99_m`
- `position_max_m`
- `orientation_mean_deg`
- `orientation_max_deg`
- `rpy_rmse_deg`

保存：

```text
results/pseudo_gt/corldata/<seq>/evo_gt_aligned/aligned_poses.csv
results/pseudo_gt/corldata/<seq>/evo_gt_aligned/metrics.json
```

解释原则：

- `pos_rmse` 是 evo SE(3) 对齐后的 3D 平均误差。
- `pos_rmse` 小不代表没有局部坏段；必须同时查看 `p90/p99/max`。
- 网页图会自动缩放轨迹，厘米级误差在小尺度轨迹上仍可能视觉明显。

## 4. 汇总报告

主脚本会生成：

```text
results/pseudo_gt/corldata/<summary-prefix>_report.md
results/pseudo_gt/corldata/<summary-prefix>_summary.csv
results/pseudo_gt/corldata/<summary-prefix>_summary.json
```

报告字段：

- `offset s`：自动搜索出的时间偏移。
- `raw VO`：`run` 表示本次运行了 ORB-SLAM3，`reuse` 表示复用了已有 VO。
- `assoc`：成功关联的 pose 数。
- `pos RMSE / p90 / p99 / max m`：evo aligned 后的位置误差统计。
- `rot RMSE / max deg`：evo aligned 后的角度误差统计。
- `fitted right t norm m`：`||translation(T_C_B_fit)||`，诊断量，不是误差。

## 5. 可视化

生成 HTML：

```bash
/home/amao/anaconda3/bin/python plato/scripts/orbslam3_export/make_corldata_3d_html.py
```

入口：

```text
results/pseudo_gt/corldata/interactive_3d/index.html
```

每个序列页面包含：

- 可拖动 3D 轨迹图
- XY 投影视图
- YZ 投影视图
- XZ 投影视图
- 米制坐标轴刻度
- 自动比例尺，候选为 `0.1 m / 0.2 m / 0.5 m / 1.0 m`

HTML 只使用：

```text
results/pseudo_gt/corldata/<seq>/evo_gt_aligned/aligned_poses.csv
results/pseudo_gt/corldata/<seq>/evo_gt_aligned/metrics.json
```

即可视化读取的是 evo 保存并由脚本整理后的 aligned trajectory。

## 6. 测试集关灯段标记

`S13+` testing data 可从导出的 DAVIS 灰度图自动识别开灯/关灯段，并把状态映射到完整的 converted GT 轨迹：

```bash
/home/amao/anaconda3/bin/python plato/scripts/orbslam3_export/detect_corldata_light_segments.py
```

默认处理：

```text
S13 S15 S20 S21 S22 S23
```

输出：

```text
results/pseudo_gt/corldata/light_segments/<seq>_gt_xy_light_segments.png
results/pseudo_gt/corldata/light_segments/<seq>_brightness_diagnostic.png
results/pseudo_gt/corldata/light_segments/light_segments_summary.md
results/pseudo_gt/corldata/light_segments/light_segments.csv
```

逻辑：

- 从 `results/pseudo_gt/corldata/<seq>/sequence/left/*.png` 计算平均亮度、暗像素比例和梯度能量。
- 用平均亮度的 Otsu 阈值初分割，再做短窗口平滑、短 gap 填充和短 off 段过滤。
- 读取 `evo_gt_aligned/metrics.json` 中的 `time_offset_s`，将图像时间映射到 GT 时间。
- XY 图读取 `gt_aligned/groundtruth_camera_frame.tum` 的完整 GT 轨迹，黑色表示开灯段，灰色表示关灯段。
- 绘图时按 GT 时间排序、去除重复时间戳，并且不跨时间/位置不连续处连线，避免可视化产生假的跳变线。

## 7. 质量检查清单

每次评估后至少检查：

- 时间偏移是否落在合理范围，且 `time_offset_scan.csv` 最小值附近平滑。
- `associated_pose_count` 是否接近 VO pose 总数。
- `position_rmse_m`、`position_p90_m`、`position_p99_m`、`position_max_m` 是否一致合理。
- `orientation_rmse_deg` 和 `orientation_max_deg` 是否存在局部突变。
- `camera_to_body_translation_norm_m` 是否跨序列异常变化。
- HTML 中 3D、XY、YZ、XZ 视图是否都显示 GT 和 VO。
- 对疑似 mocap 跳变序列，结合 `gt_quality/` 诊断，不直接接受为 pseudo-GT。
- 对 `S13+` testing data，检查 `light_segments/*_brightness_diagnostic.png` 是否显示清晰亮度阶跃，避免把普通暗纹理误判为关灯。

## 8. 接受标准

当前 corldata S00-S05 仍是评估和诊断结果，不自动等同于最终 pseudo-GT。接受某段为训练用 pseudo-GT 前，需要同时满足：

- stereo VO tracking 完整或局部窗口明确。
- GT 无明显未处理跳变。
- SE(3) 对齐后位置误差和角度误差稳定。
- `p99/max` 不显示严重局部失真。
- frame transform 解释合理，且不依赖明显非物理补偿。

`S13+` 例外：即使某些序列指标看起来较好，也默认保留为 testing data，用于评估 learned bias/network correction 在视觉退化条件下的表现。除非用户明确改变数据划分，不应把 `S13+` 加入训练 pseudo-GT 池。
