# Debias_IMU LIASuw Supervision Check

Last checked: 2026-05-09

## GT Frame Used For Supervision

Frame convention: `T_AB` maps points from frame `B` into frame `A`.

The LIASuw 200 Hz Debias_IMU export uses the following chain:

```text
T_Wv_C    : ORB-SLAM3 stereo-only left-camera pose in VO world
T_C_I     : camera-to-IMU/body extrinsic
T_Wv_I    = T_Wv_C * T_C_I

T_Wm_M    : mocap marker pose in mocap/world frame
T_M_I     : estimated fixed mocap-marker to IMU/body transform
T_Wm_I_gt = T_Wm_M * T_M_I
```

Therefore the supervision pose is the IMU/body pose in the mocap/world frame,
not the camera pose and not the raw mocap marker pose.

The aggregate Debias_IMU training tree uses:

```text
results/debias_imu/LIASuw_training_candidates_200hz/<seq>/mav0/mocap0/data.csv
```

This file was checked against the per-sequence exported body GT:

```text
results/pseudo_gt/LIASuw/<seq>/debias_imu_body_gt_200hz/body_gt_world.csv
```

The check command is:

```bash
/home/amao/anaconda3/envs/debiasimu/bin/python \
  plato/scripts/train_bias_dynamics/verify_liasuw_supervision_frames.py
```

Result on 2026-05-09: all selected sequences matched exactly:

```text
0912s06_80   max_dt_ns=0 max_pose_abs_diff=0
0912s07_30   max_dt_ns=0 max_pose_abs_diff=0
0912s10_40   max_dt_ns=0 max_pose_abs_diff=0
0203s18_80   max_dt_ns=0 max_pose_abs_diff=0
0203s19_80   max_dt_ns=0 max_pose_abs_diff=0
0203s22_80   max_dt_ns=0 max_pose_abs_diff=0
0205s14_80   max_dt_ns=0 max_pose_abs_diff=0
```

The first-frame-normalized files exist only as diagnostics and are not the
default supervision tree.  The current Debias_IMU dynamics use
`g_W = [0, 0, -9.81]^T`, so the world-frame `body_gt_world.csv` variant is the
appropriate one unless gravity is explicitly transformed for first-frame
coordinates.

## IMU Propagation Model

Debias_IMU's state is:

```text
y = [xi, t, b_g, v_W, p_W, b_a]
R_hat_WI = R0_WI * Exp(xi)
```

The IMU measurements are in the IMU/body frame:

```text
omega_tilde_I, a_tilde_I
```

The continuous model implemented by `VFSE23vbiasBase` is:

```text
d R_hat_WI / dt = R_hat_WI * (omega_tilde_I - b_g)^
d v_hat_W / dt  = R_hat_WI * (a_tilde_I - b_a) + g_W
d p_hat_W / dt  = v_hat_W
d b_g / dt      = f_g(b_g, omega_tilde_I, d omega_tilde_I / dt)
d b_a / dt      = f_a(b_a, a_tilde_I, d a_tilde_I / dt)
```

With first-order Euler integration, this corresponds to:

```text
R_hat_{k+1} = R_hat_k * Exp((omega_tilde_k - b_g,k) * dt)
v_hat_{k+1} = v_hat_k + (R_hat_k * (a_tilde_k - b_a,k) + g_W) * dt
p_hat_{k+1} = p_hat_k + v_hat_k * dt
```

This is the current code path.  A second-order preintegration update would add
`0.5 * acceleration * dt^2` to position, but the Debias_IMU runner currently
integrates the ODE with `p_dot = v`.

## Consistency With GT Body State

The GT state is built as:

```text
X_gt = SE_2(3)(R_gt_WI, v_gt_W, p_gt_W)
```

with:

```text
X_gt[:, :3, :3] = R_gt_WI
X_gt[:, :3,  3] = v_gt_W
X_gt[:, :3,  4] = p_gt_W
```

The ODE initial state is taken directly from the same GT body state:

```text
R0 = X_gt[0, :3, :3]
v0 = X_gt[0, :3, 3]
p0 = X_gt[0, :3, 4]
```

The predicted state is compared to GT using the same layout:

```text
R_hat  = R_pred
v_hat  = sol_pred[:, 7:10]
p_hat  = sol_pred[:, 10:13]

AOE     compares R_pred              with X_gt[:, :3, :3]
ATE vel compares sol_pred[:, 7:10]   with X_gt[:, :3, 3]
ATE pos compares sol_pred[:, 10:13]  with X_gt[:, :3, 4]
```

So `R_hat`, `v_hat`, and `p_hat` are the predicted IMU/body pose, velocity, and
position in the same mocap/world frame as the LIASuw body GT.
