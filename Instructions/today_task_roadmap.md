# 今日任务路线图

## 0. 今日核心目标

今天的主线是先把 **ground-truth trajectory supervision** 跑通，并在此基础上完成 input 设计和 bias dynamics 模型的第一轮验证。

整体顺序为：

1. 先完成 GT 监督闭环；
2. 保持与 `Debias_IMU/BiasDy` 原项目一致的数据处理方式；
3. 对比原文 baseline cost function 和新的 \(SE_2(3)\) cost function；
4. 系统分析 bias dynamics 的输入设计，包括 \((b,u,\dot u)\)、\((b,u)\)、局部变化量和短时间窗口特征；
5. GT 监督稳定后，再扩展到高精度 visual odometry supervision。

今天不优先做 VO 监督。VO 监督作为下一阶段任务预留。

---

## 1. 总体实验路线

监督信号分为两类：

1. **Ground-truth trajectory supervision**
2. **High-precision visual odometry supervision**

当前阶段优先级：

```text
Priority 1: GT supervision pipeline
Priority 2: input ablation for bias dynamics
Priority 3: baseline loss vs SE_2(3) loss
Priority 4: prepare interface for future VO supervision
```

GT 监督的作用不是最终证明方法实用性，而是在最干净的监督信号下验证：

1. 数据处理是否正确；
2. 坐标系是否正确；
3. 训练 pipeline 是否正确；
4. bias dynamics input 设计是否合理；
5. 新的 \(SE_2(3)\) supervision loss 是否优于 baseline loss。

---

## 2. 第一阶段：完成 GT 监督数据处理

### 2.1 GT 对齐到 IMU 预积分坐标系

目标是把原始 GT trajectory 转换为 IMU/body frame 下的轨迹，使其能够直接监督 IMU preintegration。

需要完成：

1. 明确原始 GT 的坐标系定义，例如 mocap/world frame、marker frame、camera frame 等；
2. 明确 IMU 预积分使用的 body frame；
3. 计算并应用必要的外参变换；
4. 将 GT 转换为 IMU/body 在世界系下的位姿：

\[
T_{WI}(t)
\]

5. 对齐初始帧，使训练数据和 IMU preintegration pipeline 使用同一个初始坐标系；
6. 检查 gravity direction、roll/pitch/yaw 趋势、位置轨迹趋势是否一致。

建议输出检查图：

- 原始 GT trajectory；
- 对齐后的 GT trajectory；
- 对齐后 GT 与 IMU/VIO trajectory 的 \(x,y,z\) 对比；
- roll, pitch, yaw 对比；
- trajectory alignment error curve。

---

### 2.2 GT 插值到 IMU 频率

原始 GT 频率为 120 Hz，IMU 频率为 200 Hz，因此需要将 GT 插值到 IMU timestamps。

要求：

1. 保持和 `Debias_IMU/BiasDy` 原项目一致的数据处理风格；
2. 位置可以使用 linear interpolation 或 cubic interpolation；
3. 旋转必须在 \(SO(3)\) 上插值，不要直接对 rotation matrix 或 Euler angle 做普通线性插值；
4. 插值后得到与 IMU timestamp 对应的：

\[
R_{WI}(t_k), \quad p_{WI}(t_k), \quad v_{WI}(t_k)
\]

5. 如果训练 pipeline 需要 velocity，则从 GT 数据同步插值，或从插值后的 position 中估计 velocity；
6. 插值后必须检查轨迹是否存在跳变、丢失和异常段。

---

### 2.3 训练窗口构造

保持和原项目一致的训练数据组织方式。

建议设置：

```text
window_size = 64
imu_frequency = 200 Hz
window_duration = 64 / 200 = 0.32 s
```

每个 training window 至少包含：

1. IMU measurements；
2. 初始状态；
3. GT supervision trajectory；
4. bias state 或 bias label；
5. 可选的 GT quality flag。

数据筛选要求：

1. 删除 GT 跳变段；
2. 删除 GT 丢失段；
3. 删除 tracking 或 mocap 质量明显差的段；
4. 只保留稳定、连续、高质量的 trajectory segment 用于训练监督。

---

## 3. 第二阶段：接入训练 pipeline

### 3.1 保持原项目结构不被破坏

接入 GT supervision 时，不要大幅重写原始 BiasDy pipeline。

建议只做最小侵入式修改：

1. 在 dataset 中新增或扩展 GT supervision 字段；
2. 在 loss 中新增 \(SE_2(3)\) supervision loss；
3. 在 training script 中增加配置开关；
4. 在 evaluation 中增加 trajectory error 对比；
5. 保留原文 baseline 模型和 baseline loss，方便公平对比。

建议配置项：

```yaml
supervision_type: gt
cost_type: baseline  # options: baseline, se23
window_size: 64
imu_frequency: 200
use_gt_quality_filter: true
input_type: b_u_udot  # options: b_u_udot, b_u, b_u_delta, b_u_history
```

---

## 4. 第三阶段：复现原文 BiasDy baseline

原文 bias dynamics 采用 gyro bias 和 accelerometer bias 分离建模。

### 4.1 Gyro bias dynamics

原文形式为：

\[
\dot b^g = f^g_\theta(b^g,\omega,\dot\omega)
\]

对应输入：

```text
gyro input = [bg, w, w_dot]
```

输出：

```text
gyro output = bg_dot
```

### 4.2 Accelerometer bias dynamics

原文形式为：

\[
\dot b^a = f^a_\theta(b^a,a,\dot a)
\]

对应输入：

```text
acc input = [ba, a, a_dot]
```

输出：

```text
acc output = ba_dot
```

其中 \(\dot u\) 由 IMU signal 的 spline 连续化后求导得到。该 baseline 必须保留，用于和原文进行公平对比。

---

## 5. 第四阶段：测试 loss function

### 5.1 Baseline cost function

先跑通原项目已有 baseline loss。

目的：

1. 验证数据处理没有问题；
2. 验证 GT supervision 已正确进入 pipeline；
3. 验证训练能够稳定收敛；
4. 得到可比较的 baseline 结果。

需要记录：

1. training loss；
2. validation loss；
3. trajectory RMSE；
4. rotation error；
5. bias prediction curve；
6. preintegration trajectory before/after debiasing。

---

### 5.2 \(SE_2(3)\) cost function

在 baseline 跑通后，测试新的 \(SE_2(3)\) trajectory supervision loss。

状态定义为：

\[
X_k = (R_k, v_k, p_k) \in SE_2(3)
\]

可以使用 trajectory residual：

\[
r_k = \log\left(X_{k,\mathrm{gt}}^{-1}\hat X_k\right)
\]

loss 形式为：

\[
\mathcal{L}_{SE_2(3)}
=
\sum_k r_k^\top W r_k
\]

其中 \(r_k\) 同时包含 rotation、velocity 和 position residual。

重点比较：

1. baseline cost 是否能稳定训练；
2. \(SE_2(3)\) cost 是否带来更好的 trajectory consistency；
3. \(SE_2(3)\) cost 是否对 rotation、velocity、position 同时有效；
4. \(SE_2(3)\) cost 是否更适合 IMU preintegration 的状态结构。

---

## 6. 第五阶段：input ablation 和 \(f_\theta\) 结构分析

GT 监督阶段的主要目标之一是验证 bias dynamics 的输入设计。

核心问题：

> 原文使用 \((b,u,\dot u)\)，但 \(\dot u\) 需要 spline 连续化后求导，计算和实现都更复杂。我们需要验证是否可以用更高效、更鲁棒的输入形式替代 \(\dot u\)。

---

### 6.1 实验 A0：原文输入

这是必须保留的 baseline。

\[
\dot b^g = f^g_\theta(b^g,\omega,\dot\omega)
\]

\[
\dot b^a = f^a_\theta(b^a,a,\dot a)
\]

输入为：

```text
gyro input = [bg, w, w_dot]
acc input  = [ba, a, a_dot]
```

目的：

1. 复现原文；
2. 建立公平对比；
3. 评估 \(\dot u\) 的真实贡献。

---

### 6.2 实验 A1：去掉 \(\dot u\)

测试：

\[
\dot b = f_\theta(b,u)
\]

输入为：

```text
gyro input = [bg, w]
acc input  = [ba, a]
```

目的：

1. 验证是否真的需要 \(\dot u\)；
2. 提高计算效率；
3. 减少 spline derivative 对噪声的敏感性；
4. 测试更简单模型的泛化能力。

---

### 6.3 实验 A2：使用局部差分替代 spline derivative

测试：

\[
\dot b_k = f_\theta(b_k,u_k,\Delta u_k)
\]

输入为：

```text
gyro input = [bg, w, delta_w]
acc input  = [ba, a, delta_a]
```

其中：

\[
\Delta u_k = u_k - u_{k-1}
\]

目的：

1. 用低成本局部变化量替代 \(\dot u\)；
2. 避免 spline derivative；
3. 保留部分运动变化信息；
4. 比较其是否接近或优于原文 \(\dot u\) 方案。

---

### 6.4 实验 A3：使用短时间窗口特征

测试：

\[
\dot b_k = f_\theta(b_k, u_{k-m:k})
\]

输入可以是：

```text
[b_k, u_k, u_{k-1}, ..., u_{k-m}]
```

或者压缩为：

```text
[b_k, u_k, mean(u), std(u), delta_u]
```

目的：

1. 不显式使用 \(\dot u\)；
2. 利用一段时间内的 IMU 变化信息；
3. 提高对不同运动模式的泛化能力；
4. 测试短时间 history 是否比 spline derivative 更鲁棒。

---

### 6.5 实验 A4：可选的 gyro/acc coupling model

原文将 gyro bias 和 accelerometer bias 分开建模，但真实 IMU bias dynamics 可能存在耦合。

可选测试：

\[
\dot b = f_\theta(b^g,b^a,\omega,a)
\]

输入为：

```text
input  = [bg, ba, w, a]
output = [bg_dot, ba_dot]
```

该实验不是今天的第一优先级，可以放在 E0/E1/E2/E3 稳定后再做。

---

## 7. 今日建议实验顺序

今天优先完成以下顺序：

| 优先级 | 任务 | 目标 |
|---|---|---|
| P0 | 建立安全分支和配置开关 | 保证不破坏原项目 |
| P1 | 完成 GT 到 IMU/body frame 对齐 | 保证监督信号坐标系正确 |
| P2 | 完成 GT 到 IMU 频率插值 | 保持与 IMU timestamps 对齐 |
| P3 | 构造 64-step training windows | 接入训练数据 |
| P4 | 跑通原文 baseline input \([b,u,\dot u]\) | 得到公平对比 |
| P5 | 跑通 baseline cost function | 验证 pipeline |
| P6 | 跑通 \(SE_2(3)\) cost function | 验证新 supervision loss |
| P7 | 测试 \([b,u]\) 输入 | 判断 \(\dot u\) 是否必要 |
| P8 | 测试 \([b,u,\Delta u]\) 输入 | 替代 spline derivative |
| P9 | 整理结果和图像 | 输出今日实验总结 |

---

## 8. 今日最小可交付结果

今天至少应该产出以下结果：

### 8.1 数据处理结果

1. GT 对齐到 IMU/body frame 的脚本；
2. GT 插值到 IMU timestamps 的脚本；
3. 64-step training window 构造结果；
4. 坐标系和插值检查图。

### 8.2 训练结果

至少完成以下两组：

```text
E0: original input [b, u, u_dot] + baseline cost
E1: original input [b, u, u_dot] + SE_2(3) cost
```

如果时间允许，继续完成：

```text
E2: input [b, u] + baseline cost
E3: input [b, u] + SE_2(3) cost
E4: input [b, u, delta_u] + SE_2(3) cost
```

### 8.3 对比图

每组实验至少输出：

1. loss curve；
2. trajectory before/after debiasing；
3. GT vs predicted trajectory；
4. \(x,y,z\) over time；
5. roll, pitch, yaw over time；
6. bias prediction curve；
7. trajectory RMSE 和 rotation error。

---

## 9. 结果记录格式

建议每组实验保存如下信息：

```text
experiment_name
supervision_type
cost_type
input_type
window_size
training_dataset
testing_dataset
trajectory_rmse
rotation_rmse
velocity_rmse
final_train_loss
final_val_loss
notes
```

推荐实验命名：

```text
gt_baseline_b_u_udot
gt_se23_b_u_udot
gt_baseline_b_u
gt_se23_b_u
gt_se23_b_u_delta
gt_se23_b_u_history
```

---

## 10. 后续任务：VO supervision

GT supervision 跑通之后，再切换到 high-precision visual odometry supervision。

VO supervision 需要额外处理：

1. VO 轨迹质量筛选；
2. VO 坐标系到 IMU/body frame 的外参变换；
3. VO 与 IMU 时间同步；
4. tracking loss 和跳变段剔除；
5. VO supervision 的置信度建模。

当前不优先实现 VO supervision，但今天写代码时应预留接口：

```yaml
supervision_type: gt  # future options: gt, vo
```

并保证未来可以较容易切换为：

```yaml
supervision_type: vo
```

---

## 11. 给 Codex 的执行要求

请按照以下原则执行：

1. 优先跑通 GT supervision，不要一开始处理 VO supervision；
2. 保持和 `Debias_IMU/BiasDy` 原项目一致的数据处理方式；
3. 不要破坏原始 training pipeline；
4. 所有新增功能使用配置开关控制；
5. 原文 \([b,u,\dot u]\) input 必须保留为 baseline；
6. 新增 \([b,u]\)、\([b,u,\Delta u]\)、短时间窗口 input；
7. baseline cost 和 \(SE_2(3)\) cost 都要支持；
8. 训练窗口先统一使用 64 steps；
9. 每个实验必须保存配置、日志、曲线和评价结果；
10. 每次修改后检查全文 pipeline 的一致性、代码严谨性和可复现实验记录。

---

## 12. 今日判断标准

今天的任务是否成功，不以最终精度最高为唯一标准，而以以下标准判断：

1. GT 是否已经正确对齐到 IMU/body frame；
2. GT 是否已经正确插值到 IMU frequency；
3. 训练 pipeline 是否能够稳定运行；
4. baseline input \([b,u,\dot u]\) 是否能够复现；
5. \(SE_2(3)\) cost 是否能够接入并训练；
6. \([b,u]\) 是否能作为更高效输入进行初步比较；
7. 实验结果是否可记录、可复现、可继续扩展到 VO supervision。

核心原则：

> 先用 GT supervision 验证方法正确性，再用 VO supervision 验证方法实用性。
