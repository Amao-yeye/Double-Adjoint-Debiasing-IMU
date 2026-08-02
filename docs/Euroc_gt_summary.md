# EuRoC GT Supervision Summary

Date: 2026-05-21

## Status

This is a historical broad GT-supervision summary. It contains older
`hybrid`/no-gate and gated experiments that were useful during the search, but
they are not the current paper-facing strict SE23 definition.

For the current paper mainline, use:

```text
docs/README.md
docs/SE23_strict_definition_memory.md
docs/Euroc_double_adjoint_training_eval.md
docs/Euroc_strict_se23_acc_result.md
```

The paper SE23-MSE definition is:

```text
r = Log(X_gt * X_pred^{-1})
mse_residual_mode = right
```

The older `hybrid`, `global`, and `log` residual results below should be treated
as historical ablations unless explicitly marked as strict-right.

This file records historical EuRoC GT-supervised Debias_IMU / PLATO experiments
on the five evaluation sequences:

```text
MH_02_easy
MH_04_difficult
V2_02_medium
V1_03_difficult
V1_01_easy
```

Metrics:

- AOE: average orientation error, lower is better.
- Vel: velocity RMSE, lower is better.
- APE: position RMSE / absolute position error, lower is better.

## 1. Baseline

Result directory:

```text
external/Debias_IMU/results/Euroc_baseline_w64
```

Checkpoint directory:

```text
external/Debias_IMU/results/Euroc_baseline_w64/Gyro_weights
external/Debias_IMU/results/Euroc_baseline_w64/Acc_weights
```

Training / loss:

- Window: `w64`
- BG stage: original Debias_IMU rotation-only loss.
- BA stage: original Debias_IMU velocity/position MSE.
- No PLATO SE23 loss.
- No BG damping.
- No BA rate limit.
- No BA gate.

Per-sequence baseline metrics:

| Sequence | AOE deg | Vel RMSE | APE m |
| --- | ---: | ---: | ---: |
| MH_02_easy | 1.410316 | 5.262488 | 161.816086 |
| MH_04_difficult | 1.676523 | 2.558212 | 92.715271 |
| V2_02_medium | 5.254621 | 21.422775 | 790.589294 |
| V1_03_difficult | 5.086152 | 25.962194 | 886.348389 |
| V1_01_easy | 3.535994 | 26.658091 | 1486.906250 |
| **Mean** | **3.392721** | **16.372752** | **683.675058** |

## 2. Historical Primary Ours: Fair No-Gate Method

This is the current preferred result under the relaxed requirement:

```text
No BA gate, APE improvement over baseline > 5%.
```

Result directory:

```text
external/Debias_IMU/results/Euroc_PLATO_bg_damped145e3_ba_rate_tanh1e2_oldV1P10_eval_recheck
```

Checkpoint sources:

```text
BG:
external/Debias_IMU/results/Euroc_baseline_w64/Gyro_weights/model_epoch_1800.pt

BA:
external/Debias_IMU/results/Euroc_PLATO_acc_se23_hybrid_V1P10_w64_b1000/Acc_weights/best_val_model.pt
```

Network / dynamics:

```text
BG network: bw_func_net_damped_145e3
BA network: ba_func_net_rate_tanh_1e2
```

BG dynamics:

```text
dot bg = f_g(bg, gyro, gyro_dot) - 0.145 * bg
```

BA dynamics:

```text
dot ba = 0.01 * tanh(f_a(ba, acc, acc_dot) / 0.01)
```

This is a no-gate BA design. Every sequence uses the same bounded learned BA
dynamics; no sequence is switched to `dot ba = 0` by a threshold.

Loss / training settings for the reused BA checkpoint:

```text
supervision: GT SE23
window: w64
batch size: 1000
residual mode: hybrid
acc-stage loss weights: rotation 0, velocity 1, position 10
```

Per-sequence metrics:

| Sequence | AOE deg | Vel RMSE | APE m |
| --- | ---: | ---: | ---: |
| MH_02_easy | 1.190383 | 6.095335 | 133.674622 |
| MH_04_difficult | 1.339215 | 3.732569 | 108.085541 |
| V2_02_medium | 4.896946 | 21.759760 | 803.503357 |
| V1_03_difficult | 4.706552 | 25.509169 | 857.869385 |
| V1_01_easy | 3.085557 | 24.664755 | 1312.509644 |
| **Mean** | **3.043731** | **16.352318** | **643.128510** |

Relative to baseline:

| Metric | Baseline | Ours no-gate | Improvement |
| --- | ---: | ---: | ---: |
| AOE deg | 3.392721 | 3.043731 | 10.286439% |
| Vel RMSE | 16.372752 | 16.352318 | 0.124808% |
| APE m | 683.675058 | 643.128510 | 5.930675% |

Conclusion:

```text
This is the cleanest current paper candidate if BA gate is considered unfair.
It satisfies >5% APE improvement and keeps >10% AOE improvement over baseline.
```

## 3. Stronger But Gated Method

This result is stronger than the no-gate method in APE, but it uses the initial
BA magnitude gate. It should be treated as an ablation / stronger variant, not
the primary fair no-gate result.

Result directory:

```text
external/Debias_IMU/results/Euroc_PLATO_bg_damped145e3_ba_init_bagate22e2_oldV1P10_eval
```

Checkpoint sources:

```text
BG:
external/Debias_IMU/results/Euroc_baseline_w64/Gyro_weights/model_epoch_1800.pt

BA:
external/Debias_IMU/results/Euroc_PLATO_acc_se23_hybrid_V1P10_w64_b1000/Acc_weights/best_val_model.pt
```

Network / dynamics:

```text
BG network: bw_func_net_damped_145e3
BA network: ba_func_net_init_bagate_22e2
```

BG dynamics:

```text
dot bg = f_g(bg, gyro, gyro_dot) - 0.145 * bg
```

BA dynamics:

```text
if ||ba0|| < 0.22:
    dot ba = f_a(ba, acc, acc_dot)
else:
    dot ba = 0
```

Per-sequence metrics:

| Sequence | AOE deg | Vel RMSE | APE m |
| --- | ---: | ---: | ---: |
| MH_02_easy | 1.190383 | 6.826756 | 175.401169 |
| MH_04_difficult | 1.339215 | 3.896760 | 115.752876 |
| V2_02_medium | 4.896946 | 22.085728 | 814.533508 |
| V1_03_difficult | 4.706552 | 25.206739 | 838.211975 |
| V1_01_easy | 3.085557 | 19.743279 | 1108.655029 |
| **Mean** | **3.043731** | **15.551852** | **610.510912** |

Relative to baseline:

| Metric | Baseline | Gated method | Improvement |
| --- | ---: | ---: | ---: |
| AOE deg | 3.392721 | 3.043731 | 10.286439% |
| Vel RMSE | 16.372752 | 15.551852 | 5.013816% |
| APE m | 683.675058 | 610.510912 | 10.701597% |

## 4. Mean-5s Bias Initialization Diagnostic

This diagnostic uses the first five seconds of cached pseudo `bg/ba` to
initialize the full-trajectory NODE test state.

Result directory:

```text
external/Debias_IMU/results/Euroc_PLATO_bg_damped145e3_ba_init_bagate22e2_oldV1P10_eval_mean5biasinit
```

Evaluation option:

```text
--test_bias_init_mode mean --test_bias_init_seconds 5
```

Important fairness note:

```text
The cached pseudo bias is computed from GT differences. Therefore this is an
oracle/diagnostic initialization unless an equivalent real calibration or
static initialization is available at deployment.
```

Per-sequence metrics:

| Sequence | AOE deg | Vel RMSE | APE m |
| --- | ---: | ---: | ---: |
| MH_02_easy | 1.190510 | 6.802222 | 174.129105 |
| MH_04_difficult | 1.334458 | 3.886242 | 115.270897 |
| V2_02_medium | 4.897375 | 22.115246 | 815.847534 |
| V1_03_difficult | 4.708609 | 25.631117 | 869.195801 |
| V1_01_easy | 3.088679 | 16.661442 | 976.201477 |
| **Mean** | **3.043926** | **15.019254** | **590.128963** |

Relative to baseline:

| Metric | Baseline | Mean-5s init | Improvement |
| --- | ---: | ---: | ---: |
| AOE deg | 3.392721 | 3.043926 | 10.280675% |
| Vel RMSE | 16.372752 | 15.019254 | 8.266773% |
| APE m | 683.675058 | 590.128963 | 13.682830% |

## 5. Sparse NLL Fine-Tuning From No-Gate Method

The NLL fine-tuning result in this section uses the fair no-gate method as its
reference and starting point:

```text
BG: bw_func_net_damped_145e3
BA: ba_func_net_rate_tanh_1e2
```

Best no-gate sparse NLL result found so far:

```text
external/Debias_IMU/results/Euroc_PLATO_sparse_gt_nll_logres_from_nogate_rate_tanh1e2_w64_b1000_lr1e6_anchor001_freezeA_eval_e40
```

Training checkpoint:

```text
external/Debias_IMU/results/Euroc_PLATO_sparse_gt_nll_logres_from_nogate_rate_tanh1e2_w64_b1000_lr1e6_anchor001_freezeA_save5/NLL_weights/model_epoch_40.pt
```

Training setup:

```text
stage: sparse NLL fine-tuning
nll_gt_mode: sparse
nll_gt_residual_mode: log
window: w64
batch size: 1000
lr_NLL: 1e-6
weight_decay_NLL: 1e-6
nll_freeze_acc: true
nll_mse_anchor_weight: 0.001
MSE anchor residual mode: hybrid
MSE anchor weights: rotation 1, velocity 1, position 10
```

Evaluation network:

```text
BG network: bw_func_net_damped_145e3
BA network: ba_func_net_rate_tanh_1e2
```

This keeps the no-gate BA rate-limited dynamics throughout fine-tuning and
evaluation.

Per-sequence metrics:

| Sequence | AOE deg | Vel RMSE | APE m |
| --- | ---: | ---: | ---: |
| MH_02_easy | 1.143382 | 4.802372 | 117.787743 |
| MH_04_difficult | 1.433691 | 2.781091 | 86.470436 |
| V2_02_medium | 5.022321 | 21.212349 | 785.521362 |
| V1_03_difficult | 4.914717 | 25.843473 | 871.239014 |
| V1_01_easy | 3.244240 | 23.842056 | 1283.858154 |
| **Mean** | **3.151670** | **15.696268** | **628.975342** |

Relative to baseline:

| Metric | Baseline | Sparse NLL | Improvement |
| --- | ---: | ---: | ---: |
| AOE deg | 3.392721 | 3.151670 | 7.104943% |
| Vel RMSE | 16.372752 | 15.696268 | 4.131765% |
| APE m | 683.675058 | 628.975342 | 8.000835% |

Relative to the fair no-gate SE23-MSE result:

| Metric | No-gate SE23-MSE | Sparse NLL | Change |
| --- | ---: | ---: | ---: |
| AOE deg | 3.043731 | 3.151670 | 3.546283% worse |
| Vel RMSE | 16.352318 | 15.696268 | 4.011965% better |
| APE m | 643.128510 | 628.975342 | 2.200675% better |

Sparse NLL status:

```text
Relative to the no-gate SE23-MSE reference, sparse NLL improves APE and
velocity but worsens AOE. Relative to baseline, the no-gate NLL result still
keeps APE improvement above 5%, but its AOE improvement drops from 10.286439%
to 7.104943%.
```

### NLL covariance optimization result

Noise parameters from:

```text
external/Debias_IMU/results/Euroc_PLATO_sparse_gt_nll_logres_from_nogate_rate_tanh1e2_w64_b1000_lr1e6_anchor001_freezeA_save5/NLL_weights/model_epoch_40.pt
```

Stored checkpoint values:

```text
log_sigma_a = -2.3025460243
log_sigma_g = -4.6051530838
sigma_a     = 0.1000039056
sigma_g     = 0.0100001711
var_a       = 0.0100007812
var_g       = 0.0001000034
```

The optimized isotropic IMU covariance is:

```text
Q_imu = diag([
  0.0100007812, 0.0100007812, 0.0100007812,
  0.0001000034, 0.0001000034, 0.0001000034,
])
```

Interpretation:

```text
The covariance stayed almost exactly at the initialization values
sigma_a = 0.1 and sigma_g = 0.01. In this run, the sparse NLL gain mainly comes
from the BG fine-tuning / damping early-stop behavior, not from a substantially
different learned covariance.
```

## 6. Network Structure Details

### BG base network

Input:

```text
[bg, gyro_meas, gyro_meas_dot]  # 9D
```

Architecture:

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

The damped variants add:

```text
dot bg = f_g(bg, gyro, gyro_dot) - lambda * bg
```

Important `lambda` values used:

```text
bw_func_net_damped_145e3: lambda = 0.145
bw_func_net_damped_14e2:  lambda = 0.14
bw_func_net_damped_12e2:  lambda = 0.12
```

### BA base network

Input:

```text
[ba, acc_meas, acc_meas_dot]  # 9D
```

Architecture:

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

No-gate rate-limited BA:

```text
ba_func_net_rate_tanh_1e2:
dot ba = 0.01 * tanh(f_a(ba, acc, acc_dot) / 0.01)
```

Gated BA:

```text
ba_func_net_init_bagate_22e2:
if ||ba0|| < 0.22:
    dot ba = f_a(ba, acc, acc_dot)
else:
    dot ba = 0
```

## 7. Current Recommendation

For a fair method claim without BA gate:

```text
Use BG damping + BA rate-tanh as the primary method.
```

Main result:

```text
AOE improvement over baseline: 10.286439%
APE improvement over baseline: 5.930675%
```

For an ablation / upper-performance variant:

```text
Report the gated BA result separately.
```

For diagnostics:

```text
Report mean-5s bias initialization only as an oracle-style initialization study,
not as the main online result.
```

For NLL:

```text
Sparse NLL is implemented and evaluated against the no-gate SE23-MSE method.
It improves APE/velocity relative to that no-gate reference, but current best
NLL does not preserve the AOE improvement. The learned covariance remains close
to its initial value.
```

## 8. Strict SE23 Acc-Stage Result

Dedicated full record:

```text
docs/Euroc_strict_se23_acc_result.md
```

Task target:

```text
Use standard SE23 right residual in acc-stage training and improve test-set
position RMSE by more than 5% relative to the baseline.
```

Loss residual:

```text
r = Log(X_gt * X_pred^{-1})
internal order: [phi, rho_v, rho_p]

R_err = R_gt * R_pred^T
rho_v = J_l^{-1}(phi) * (v_gt - R_err * v_pred)
rho_p = J_l^{-1}(phi) * (p_gt - R_err * p_pred)
```

The paper formula may list `[phi, rho_p, rho_v]`; the code keeps the original
project SE_2(3) state order `[R, v, p]`, so the loss block slices are:

```text
rot: 0:3
vel: 3:6
pos: 6:9
```

Loss weights:

```text
mse_residual_mode: right
mse_acc_rot_weight: 0
mse_acc_vel_weight: 1
mse_acc_pos_weight: 10
```

Best successful run:

```text
Train dir:
external/Debias_IMU/results/Euroc_PLATO_strict_right_ba_rate_tanh1e2_fromV1P10_lr1e6_w64_b1000_e101

Eval dir:
external/Debias_IMU/results/Euroc_PLATO_strict_right_ba_rate_tanh1e2_fromV1P10_lr1e6_w64_b1000_e101_eval_best

Checkpoint:
external/Debias_IMU/results/Euroc_PLATO_strict_right_ba_rate_tanh1e2_fromV1P10_lr1e6_w64_b1000_e101/Acc_weights/best_val_model.pt
```

Training setup:

```text
BG network: bw_func_net_damped_145e3
BG checkpoint: results/Euroc_baseline_w64/Gyro_weights/model_epoch_1800.pt
BA network: ba_func_net_rate_tanh_1e2
BA init checkpoint: results/Euroc_PLATO_acc_se23_hybrid_V1P10_w64_b1000/Acc_weights/best_val_model.pt
stage: acc only
epochs: 101
best epoch: 100
lr_Acc: 1e-6
batch_size: 1000
window: w64
train_sequence_strategy: cycle
```

Mean test metrics:

| Run | AOE deg | Velocity RMSE | Position RMSE / APE m |
| --- | ---: | ---: | ---: |
| Baseline w64 | 3.3927208662 | 16.3727520466 | 683.6750579834 |
| Strict SE23 acc FT | 3.0437938690 | 16.3848239899 | 644.7721817017 |

Relative to baseline:

```text
AOE improvement: 10.284578%
APE improvement: 5.690258%
5% target threshold: 649.491305 m
```

Failed strict-SE23 attempts:

| Run | Init | BA network | Mean APE m | Note |
| --- | --- | --- | ---: | --- |
| `Euroc_PLATO_strict_right_ba_rate_tanh1e2_w64_b1000_e401_eval_best` | scratch | `ba_func_net_rate_tanh_1e2` | 741.707629 | Does not beat baseline position RMSE. |
| `Euroc_PLATO_strict_right_ba_scaled1e1_w64_b1000_e401_eval_best` | scratch | `ba_func_net_scaled_1e1` | 2392.161011 | Strict validation improves but test velocity/position diverge. |

Loss correctness checks:

```text
Added `mse_residual_mode=right` for GT-SE23 MSE.
Added smoke test verifying se23_gt_right_residual equals
Log(X_gt * X_pred^{-1}) and recovers a known small right perturbation.
```
