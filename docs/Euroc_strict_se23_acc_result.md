# EuRoC Strict SE23 Acc-Stage Result

Date: 2026-05-21

## Goal

Use standard SE23 right residual in acc-stage training and improve test-set
position RMSE by more than 5% relative to `Euroc_baseline_w64`.

Success criterion:

```text
baseline mean APE = 683.6750579834 m
5% target threshold = 649.4913050842 m
```

## Final Result

Best run:

```text
Train dir:
external/Debias_IMU/results/Euroc_PLATO_strict_right_ba_rate_tanh1e2_fromV1P10_lr1e6_w64_b1000_e101

Eval dir:
external/Debias_IMU/results/Euroc_PLATO_strict_right_ba_rate_tanh1e2_fromV1P10_lr1e6_w64_b1000_e101_eval_best

Checkpoint:
external/Debias_IMU/results/Euroc_PLATO_strict_right_ba_rate_tanh1e2_fromV1P10_lr1e6_w64_b1000_e101/Acc_weights/best_val_model.pt
```

Mean test metrics:

| Run | AOE deg | Velocity RMSE | Position RMSE / APE m |
| --- | ---: | ---: | ---: |
| Baseline w64 | 3.3927208662 | 16.3727520466 | 683.6750579834 |
| Strict SE23 acc FT | 3.0437938690 | 16.3848239899 | 644.7721817017 |

Relative to baseline:

```text
AOE improvement = 10.284577%
APE improvement = 5.690258%
Velocity change = -0.073732%  # slightly worse
```

Per-sequence test metrics:

| Sequence | AOE deg | Velocity RMSE | Position RMSE / APE m |
| --- | ---: | ---: | ---: |
| MH_02_easy | 1.1904010773 | 6.1456284523 | 136.3275146484 |
| MH_04_difficult | 1.3391394615 | 3.7568717003 | 108.7495193481 |
| V2_02_medium | 4.8970413208 | 21.7532768250 | 803.5936889648 |
| V1_03_difficult | 4.7066202164 | 25.5018577576 | 857.7445068359 |
| V1_01_easy | 3.0857672691 | 24.7664852142 | 1317.4456787109 |

## Loss

Strict SE23 right residual:

```text
r = Log(X_gt * X_pred^{-1})
```

Internal residual order follows the original project SE_2(3) state order:

```text
[phi, rho_v, rho_p]
```

Formula implemented by `se23_gt_right_residual`:

```text
R_err = R_gt * R_pred^T
phi = Log(R_err)
rho_v = J_l^{-1}(phi) * (v_gt - R_err * v_pred)
rho_p = J_l^{-1}(phi) * (p_gt - R_err * p_pred)
```

Acc-stage loss weights:

```text
mse_residual_mode = right
mse_acc_rot_weight = 0
mse_acc_vel_weight = 1
mse_acc_pos_weight = 10
```

Important interpretation:

```text
Rotation block is not directly penalized in acc-stage training, but rho_v and
rho_p are still computed from the strict SE23 group residual and keep the
rotation coupling through R_err and J_l^{-1}(phi).
```

Loss correctness check:

```text
BiasDy/plato/loss_smoke_test.py now verifies that se23_gt_right_residual equals
Log(X_gt * X_pred^{-1}) and recovers a known small right perturbation.
```

## Training Setup

```text
stage = acc only
BG frozen = true
window = w64
batch_size = 1000
epochs = 101
best epoch = 100
lr_Acc = 1e-6
weight_decay_Acc = 1e-6
train_sequence_strategy = cycle
integral_method = euler
```

Train/validation sequences:

```text
MH_01_easy
MH_03_medium
MH_05_difficult
V1_02_medium
V2_01_easy
V2_03_difficult
```

Test sequences:

```text
MH_02_easy
MH_04_difficult
V2_02_medium
V1_03_difficult
V1_01_easy
```

## Network Structure

### BG

Network type:

```text
bw_func_net_damped_145e3
```

Checkpoint:

```text
external/Debias_IMU/results/Euroc_baseline_w64/Gyro_weights/model_epoch_1800.pt
```

Input:

```text
[bw, gyro_meas, gyro_meas_dot]  # 9D
```

Base architecture:

```text
net:
9 -> 128 -> 128 -> 9
GELU, Tanh

residual:
x = net(x) + x

net2:
9 -> 512 -> 9
Tanh

residual:
x = net2(x) + x

output:
9 -> 3
```

Damped BG dynamics:

```text
dot bg = f_bg(bg, gyro, gyro_dot) - 0.145 * bg
```

### BA

Network type:

```text
ba_func_net_rate_tanh_1e2
```

Initialization checkpoint:

```text
external/Debias_IMU/results/Euroc_PLATO_acc_se23_hybrid_V1P10_w64_b1000/Acc_weights/best_val_model.pt
```

Input:

```text
[ba, acc_meas, acc_meas_dot]  # 9D
```

Base architecture:

```text
net:
9 -> 256 -> 512 -> 9
GELU, GELU

residual:
x = net(x) + x

net2:
9 -> 256 -> 256 -> 9
GELU, GELU

residual:
x = net2(x) + x

output:
9 -> 3
```

Rate-limited BA dynamics:

```text
dot ba = 0.01 * tanh(f_ba(ba, acc, acc_dot) / 0.01)
```

No BA gate or bias-state auxiliary supervision is used.

## Failed Strict-SE23 Attempts

| Run | Init | BA network | Mean APE m | Note |
| --- | --- | --- | ---: | --- |
| `Euroc_PLATO_strict_right_ba_rate_tanh1e2_w64_b1000_e401_eval_best` | scratch | `ba_func_net_rate_tanh_1e2` | 741.707629 | Does not beat baseline position RMSE. |
| `Euroc_PLATO_strict_right_ba_scaled1e1_w64_b1000_e401_eval_best` | scratch | `ba_func_net_scaled_1e1` | 2392.161011 | Strict validation improves but test velocity/position diverge. |

