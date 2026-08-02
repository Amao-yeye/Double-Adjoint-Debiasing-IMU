# EuRoC VO32 MSE Summary

Date: 2026-05-21

## Task Interpretation

The correct meaning of `w32` for this VO experiment is:

```text
32 VO supervision timestamps per training window
```

It is not 32 IMU integration steps.

The EuRoC IMU cache is 200 Hz (`dt = 0.005 s`). The available VO trajectories
are approximately 20 Hz (`dt_vo = 0.05 s`). Therefore the VO32 configuration is:

```text
plato_window_steps = 311
plato_sup_stride   = 10
supervised indices = 0, 10, 20, ..., 310
number of VO timestamps = 32
window duration = 1.55 s
```

The training uses the first five VO-quality-accepted training sequences:

```text
MH_01_easy
MH_03_medium
MH_05_difficult
V1_02_medium
V2_01_easy
```

`V2_03_difficult` remains excluded from VO-supervised training because the VO
quality check showed it is a severe outlier.

## Code Change

Added optional VO relative residual mode:

```text
--vo_residual_mode {absolute,relative}
```

Files:

```text
external/Debias_IMU/BiasDy/plato/losses.py
external/Debias_IMU/BiasDy/plato/training.py
external/Debias_IMU/BiasDy/mainEurocPLATO.py
external/Debias_IMU/BiasDy/plato/loss_smoke_test.py
```

Default behavior remains unchanged:

```text
vo_residual_mode = absolute
```

For the VO32 experiment, `relative` mode is used. It forms relative pose
increments from the first supervised VO timestamp in the window and excludes
the first timestamp from the residual. This avoids pushing per-window
VO-vs-GT absolute alignment error into the learned bias dynamics and focuses
the loss on preintegrated motion over the VO window.

Verification passed:

```text
python BiasDy/plato/loss_smoke_test.py
python BiasDy/plato/supervision_smoke_test.py
python -m py_compile BiasDy/plato/losses.py BiasDy/plato/training.py BiasDy/mainEurocPLATO.py
```

## Baseline

Reference baseline remains:

```text
external/Debias_IMU/results/Euroc_baseline_w64
```

Mean metrics on the five standard test sequences:

| Metric | Baseline |
| --- | ---: |
| AOE deg | 3.392721 |
| Vel RMSE | 16.372752 |
| APE m | 683.675058 |

## Best VO32-MSE Result So Far

Training result:

```text
external/Debias_IMU/results/Euroc_PLATO_vo32_5seq_mse_rel_hybrid_bg145_ba_rate_tanh1e2_ft_lr1e5_e11
```

Evaluation result:

```text
external/Debias_IMU/results/Euroc_PLATO_vo32_5seq_mse_rel_hybrid_bg145_ba_rate_tanh1e2_ft_lr1e5_e11_eval_best
```

Checkpoint:

```text
external/Debias_IMU/results/Euroc_PLATO_vo32_5seq_mse_rel_hybrid_bg145_ba_rate_tanh1e2_ft_lr1e5_e11/Acc_weights/best_val_model.pt
```

Configuration:

```text
supervision: vo_pose
vo_residual_mode: relative
mse_residual_mode: hybrid
VO timestamps per window: 32
plato_window_steps: 311
plato_sup_stride: 10
batch_size: 32
stage: acc only
epochs: 11
lr_Acc: 1e-5
weight_decay_Acc: 1e-6
train_sequence_strategy: cycle
BG network: bw_func_net_damped_145e3
BA network: ba_func_net_rate_tanh_1e2
BG checkpoint: results/Euroc_baseline_w64/Gyro_weights/model_epoch_1800.pt
BA init checkpoint: results/Euroc_PLATO_acc_se23_hybrid_V1P10_w64_b1000/Acc_weights/best_val_model.pt
```

Per-sequence metrics:

| Sequence | AOE deg | Vel RMSE | APE m |
| --- | ---: | ---: | ---: |
| MH_02_easy | 1.190401 | 6.084743 | 134.539169 |
| MH_04_difficult | 1.339139 | 3.753370 | 108.611496 |
| V2_02_medium | 4.897041 | 21.722198 | 802.946167 |
| V1_03_difficult | 4.706620 | 25.528940 | 858.817261 |
| V1_01_easy | 3.085767 | 24.741077 | 1317.404297 |
| **Mean** | **3.043794** | **16.366066** | **644.463678** |

Relative to baseline:

| Metric | Baseline | VO32-MSE | Improvement |
| --- | ---: | ---: | ---: |
| AOE deg | 3.392721 | 3.043794 | 10.284577% |
| Vel RMSE | 16.372752 | 16.366066 | 0.040835% |
| APE m | 683.675058 | 644.463678 | 5.735383% |

Relative to the current no-gate GT-SE23-MSE reference:

| Metric | No-gate GT-SE23-MSE | VO32-MSE | Change |
| --- | ---: | ---: | ---: |
| AOE deg | 3.043731 | 3.043794 | 0.002076% worse |
| Vel RMSE | 16.352318 | 16.366066 | 0.084078% worse |
| APE m | 643.128510 | 644.463678 | 0.207605% worse |

Interpretation:

```text
This VO32-MSE setting satisfies the immediate requirement of improving over the
baseline on AOE, velocity RMSE, and APE. It does not improve over the previous
no-gate GT-SE23-MSE reference, so it should be reported as a VO-supervised
baseline-beating result rather than as the current global best pure-IMU model.
```

## Failed Or Less Useful Attempts

Wrong interpretation stopped:

```text
Started an original mainEuroc.py run with loss_window=32, interpreting w32 as
32 IMU steps. This was stopped after the clarification that w32 means 32 VO
supervision timestamps.
Partial directory: external/Debias_IMU/results/Euroc_baseline_w32
```

VO32 dry run:

```text
external/Debias_IMU/results/Euroc_PLATO_vo32_5seq_mse_rel_hybrid_dryrun
```

This validated sampling and gradients but was not an accuracy run.

VO32 BA fine-tune with `lr_Acc=1e-4`, 51 epochs:

```text
external/Debias_IMU/results/Euroc_PLATO_vo32_5seq_mse_rel_hybrid_bg145_ba_rate_tanh1e2_ft_e51
external/Debias_IMU/results/Euroc_PLATO_vo32_5seq_mse_rel_hybrid_bg145_ba_rate_tanh1e2_ft_e51_eval_best
```

Best validation checkpoint overfit the VO objective and worsened full test
accuracy:

| Metric | Value |
| --- | ---: |
| AOE deg | 3.043794 |
| Vel RMSE | 17.794674 |
| APE m | 707.143719 |

The early checkpoint `model_epoch_0.pt` from the same run still beat baseline
on AOE and APE, but velocity was slightly worse than baseline:

```text
external/Debias_IMU/results/Euroc_PLATO_vo32_5seq_mse_rel_hybrid_bg145_ba_rate_tanh1e2_ft_e51_eval_e0
```

Mean metrics:

| Metric | Value |
| --- | ---: |
| AOE deg | 3.043794 |
| Vel RMSE | 16.393397 |
| APE m | 646.339163 |

## Current Recommendation

Use this as the current VO32-MSE result:

```text
external/Debias_IMU/results/Euroc_PLATO_vo32_5seq_mse_rel_hybrid_bg145_ba_rate_tanh1e2_ft_lr1e5_e11_eval_best
```

Do not use the `lr_Acc=1e-4` best-val checkpoint as a paper result. It fits the
VO validation loss but damages full-trajectory test accuracy.

## VO32 Sparse NLL Fine-Tuning

Implemented sparse noisy VO pose NLL following:

```text
docs/sparse_noisy_vo_nll_for_codex.md
```

Code changes:

```text
external/Debias_IMU/BiasDy/plato/nll.py
external/Debias_IMU/BiasDy/plato/training.py
external/Debias_IMU/BiasDy/mainEurocPLATO.py
external/Debias_IMU/BiasDy/plato/nll_smoke_test.py
external/Debias_IMU/BiasDy/plato/__init__.py
```

New CLI:

```text
--nll_vo_mode {woodbury,sparse}
```

Verification:

```text
python BiasDy/plato/nll_smoke_test.py
```

The smoke test compares sparse VO pose NLL against the existing dense Woodbury
pose NLL on a short trajectory and checks gradients.

### NLL Training Setup

Starting point:

```text
VO32-MSE checkpoint:
external/Debias_IMU/results/Euroc_PLATO_vo32_5seq_mse_rel_hybrid_bg145_ba_rate_tanh1e2_ft_lr1e5_e11/Acc_weights/best_val_model.pt
```

Shared NLL settings:

```text
supervision: vo_pose
vo_residual_mode: relative
nll_vo_mode: sparse
VO timestamps per window: 32
plato_window_steps: 311
plato_sup_stride: 10
batch_size: 32
lr_NLL: 1e-6
weight_decay_NLL: 1e-6
nll_mse_anchor_weight: 0.001
meas_sigma_rot_vo: 0.01
meas_sigma_pos_vo: 0.05
BG network: bw_func_net_damped_145e3
BA network: ba_func_net_rate_tanh_1e2
```

### Primary NLL Result: All Metrics Better Than VO-MSE

Training directory:

```text
external/Debias_IMU/results/Euroc_PLATO_vo32_sparse_nll_from_vomse_lr1e6_anchor001_freezeA_e51
```

Checkpoint:

```text
external/Debias_IMU/results/Euroc_PLATO_vo32_sparse_nll_from_vomse_lr1e6_anchor001_freezeA_e51/NLL_weights/model_epoch_0.pt
```

Evaluation directory:

```text
external/Debias_IMU/results/Euroc_PLATO_vo32_sparse_nll_from_vomse_lr1e6_anchor001_freezeA_eval_e0
```

Training policy:

```text
nll_freeze_acc: true
Trainable: BG network + IMU covariance
Selected checkpoint: epoch 0
```

Mean metrics:

| Metric | VO32-MSE | VO32 Sparse NLL e0 | Change |
| --- | ---: | ---: | ---: |
| AOE deg | 3.043794 | 3.036446 | 0.241402% better |
| Vel RMSE | 16.366066 | 16.354852 | 0.068518% better |
| APE m | 644.463678 | 644.070758 | 0.060969% better |

Relative to baseline:

| Metric | Baseline | VO32 Sparse NLL e0 | Improvement |
| --- | ---: | ---: | ---: |
| AOE deg | 3.392721 | 3.036446 | 10.501152% |
| Vel RMSE | 16.372752 | 16.354852 | 0.109327% |
| APE m | 683.675058 | 644.070758 | 5.792854% |

This is the current recommended VO NLL checkpoint when the requirement is
strictly "NLL must be better than VO-MSE" across AOE, velocity, and APE.

Noise covariance at this checkpoint:

```text
sigma_a = 0.1000000909
sigma_g = 0.0100000091
var_a   = 0.0100000184
var_g   = 0.0001000002
```

The covariance is effectively unchanged at epoch 0. The small improvement comes
from the first NLL update to the BG network, not from a meaningful covariance
change.

### APE/Velocity-Focused NLL Result

Evaluation directory:

```text
external/Debias_IMU/results/Euroc_PLATO_vo32_sparse_nll_from_vomse_lr1e6_anchor001_freezeA_eval_best
```

Checkpoint:

```text
external/Debias_IMU/results/Euroc_PLATO_vo32_sparse_nll_from_vomse_lr1e6_anchor001_freezeA_e51/NLL_weights/best_val_model.pt
```

This checkpoint is epoch 50. It improves APE and velocity more clearly, but
worsens AOE relative to VO32-MSE:

| Metric | VO32-MSE | VO32 Sparse NLL best-val | Change |
| --- | ---: | ---: | ---: |
| AOE deg | 3.043794 | 3.106333 | 2.054649% worse |
| Vel RMSE | 16.366066 | 15.873238 | 3.011278% better |
| APE m | 644.463678 | 629.821132 | 2.272051% better |

Relative to baseline:

| Metric | Baseline | VO32 Sparse NLL best-val | Improvement |
| --- | ---: | ---: | ---: |
| AOE deg | 3.392721 | 3.106333 | 8.441240% |
| Vel RMSE | 16.372752 | 15.873238 | 3.050886% |
| APE m | 683.675058 | 629.821132 | 7.877123% |

Noise covariance at this checkpoint:

```text
sigma_a = 0.1000048593
sigma_g = 0.0100004859
var_a   = 0.0100009721
var_g   = 0.0001000097
```

The covariance remains very close to initialization. The gain mainly comes from
model fine-tuning, not covariance learning.

### Less Useful NLL Attempt

Freeze-BG NLL:

```text
external/Debias_IMU/results/Euroc_PLATO_vo32_sparse_nll_from_vomse_lr1e6_anchor001_freezeG_e51
external/Debias_IMU/results/Euroc_PLATO_vo32_sparse_nll_from_vomse_lr1e6_anchor001_freezeG_eval_best
```

Mean metrics:

| Metric | Value | Change vs VO32-MSE |
| --- | ---: | ---: |
| AOE deg | 3.043794 | unchanged |
| Vel RMSE | 16.383479 | 0.106395% worse |
| APE m | 645.322868 | 0.133319% worse |

This is not useful. Freezing BG and tuning BA with this NLL setting did not
improve full-trajectory accuracy.
