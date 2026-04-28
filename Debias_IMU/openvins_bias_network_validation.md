# 用 Bias Network 验证 OpenVINS

## 论文和代码里的 bias 定义

baseline 论文把 IMU 观测建模为：

```text
w_meas = w_true + bg + noise
a_meas = a_true + ba + noise
```

网络学习连续 bias 动力学：

```text
dot(bg) = fg(bg, w_meas, w_meas_dot)
dot(ba) = fa(ba, a_meas, a_meas_dot)
```

当前代码在测试阶段已经积分出整条 bias 轨迹，并保存：

```text
FullTra_result.p
  sol_pred[:, 4:7]    -> predicted gyro bias bg
  sol_pred[:, 13:16]  -> predicted accel bias ba
  net_imu             -> raw_imu - [bg, ba]
```

因此最小验证路径不是先改 OpenVINS 内部状态，而是先把 `net_imu` 导出成 OpenVINS 可读的 EuRoC/TUM IMU CSV。

## 路线 A：离线去 bias IMU，最快验证

生成 OpenVINS 可读数据：

```bash
python3 Debias_IMU/export_openvins_imu.py \
  --results_dir Debias_IMU/results/vo_dense_noV203_fast_localad_right_Jl_w8_e1801 \
  --raw_data_dir Debias_IMU/data/EUROC \
  --output_dir Debias_IMU/data/EUROC_openvins_debiased \
  --tree_mode symlink
```

`--tree_mode symlink` 会把原始数据集里的相机、标定等文件逐文件链接过去，但跳过原始 `mav0/imu0/data.csv`，再写入网络去 bias 后的 IMU。这样 OpenVINS 可以读取同一套相机数据，但 IMU 已经变成：

```text
w_for_openvins = w_raw - bg_network
a_for_openvins = a_raw - ba_network
```

对比实验建议至少跑三组：

```text
rawimu:      原始 EuRoC/TUM 数据
constant:    raw IMU - 初始常量 bias
network:     raw IMU - bias network
```

为了避免双重补偿，OpenVINS 里 IMU bias 初值应设为接近 0。若 OpenVINS 仍估计 bias，网络 IMU 可以理解为“更好的输入”，不是严格意义上的“替换 OpenVINS bias 状态”。

## 路线 B：真正替换 OpenVINS 内部 bias

如果目标是严格替换 OpenVINS EKF/MSCKF 里的 bias，需要在 OpenVINS 源码侧改 propagation：

```text
原逻辑：
  w_corr = w_raw - state.bg
  a_corr = a_raw - state.ba

替换后：
  bg_net, ba_net = bias_network(t, imu_history, bias_prev)
  w_corr = w_raw - bg_net
  a_corr = a_raw - ba_net
```

同时要决定 bias 是否仍作为滤波状态：

```text
方案 1：bias 仍在状态里，但 propagation 使用网络 bias
  优点：改动小，可保留协方差维度。
  缺点：状态里的 bias 和实际补偿 bias 不一致，更新步骤可能解释困难。

方案 2：bias 从状态中移除或固定为 0
  优点：语义干净。
  缺点：OpenVINS 预积分、协方差、状态克隆、初始化都要跟着改，工作量更大。
```

建议先做路线 A。如果路线 A 在 OpenVINS 上有收益，再进入路线 B；否则直接替换滤波器内部 bias 会很难判断失败来自网络、时间戳、坐标系，还是滤波器协方差建模。

