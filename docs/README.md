# Experiment Memory Index

Date: 2026-05-21

This is the entry point for experiment memory in `formalPLATO/docs`.

## Current Paper Mainline

The paper-facing method should be described with the strict right-invariant
SE_2(3) residual:

```text
r = Log(X_gt * X_pred^{-1})
```

Do not use older `hybrid`, `global`, or `log` residual experiments as the main
SE23-MSE definition. Those records are historical ablations unless explicitly
re-run with `mse_residual_mode=right`.

Canonical files:

| File | Role |
| --- | --- |
| `SE23_strict_definition_memory.md` | Canonical loss definition, strict MSE/NLL result table, BCH Jacobian convention. |
| `Euroc_double_adjoint_training_eval.md` | Controlled AD vs double-adjoint training/evaluation and memory results. |
| `PLATO_double_adjoint_notes.md` | Implementation notes for analytical outer adjoint and inner NODE adjoint. |
| `Euroc_strict_se23_acc_result.md` | Detailed strict SE23 acc-stage result and network/loss setup. |

## Canonical Metrics

Five-sequence EuRoC test set:

```text
MH_02_easy
MH_04_difficult
V2_02_medium
V1_03_difficult
V1_01_easy
```

Mean metrics:

| Run | AOE deg | Vel RMSE | APE m | Main note |
| --- | ---: | ---: | ---: | --- |
| Baseline w64 | 3.392720866 | 16.372752047 | 683.675057983 | Original Debias_IMU baseline. |
| Strict SE23-MSE AD | 3.043793869 | 16.384823990 | 644.772181702 | Paper strict MSE reference. |
| Strict SE23-MSE double-adjoint | 3.043793869 | 16.389042807 | 644.928808594 | Same loss/optimizer, analytical gradients. |
| Strict-right NLL double-adjoint e40 | 3.075663114 | 15.752899933 | 631.278599548 | NLL fine-tune from strict MSE, selected by APE. |

Relative to baseline:

| Run | AOE improvement | APE improvement |
| --- | ---: | ---: |
| Strict SE23-MSE AD | 10.284578% | 5.690258% |
| Strict SE23-MSE double-adjoint | 10.284577% | 5.667349% |
| Strict-right NLL double-adjoint e40 | 9.345235% | 7.663942% |

Relative to strict SE23-MSE AD, strict-right NLL e40 improves APE by
`2.092767%` and velocity RMSE by `3.856764%`, while AOE is `1.047024%` worse.

## Current Checkpoints

Strict SE23-MSE AD:

```text
external/Debias_IMU/results/Euroc_PLATO_strict_right_ba_rate_tanh1e2_fromV1P10_lr1e6_w64_b1000_e101/Acc_weights/best_val_model.pt
```

Strict SE23-MSE double-adjoint:

```text
external/Debias_IMU/results/Euroc_PLATO_strict_right_double_adjoint_ba_rate_tanh1e2_fromV1P10_lr1e6_w64_b1000_e101/Acc_weights/best_val_model.pt
```

Strict-right NLL double-adjoint e40:

```text
external/Debias_IMU/results/Euroc_PLATO_sparse_gt_nll_rightres_double_adjoint_from_strict_right_mse_w64_b1000_lr1e6_anchor001_freezeA_e51/NLL_weights/model_epoch_40.pt
```

## Main Network/Loss Setup

BG:

```text
network: bw_func_net_damped_145e3
checkpoint: external/Debias_IMU/results/Euroc_baseline_w64/Gyro_weights/model_epoch_1800.pt
dynamic: dot bg = f_bg(bg, gyro, gyro_dot) - 0.145 * bg
```

BA:

```text
network: ba_func_net_rate_tanh_1e2
dynamic: dot ba = 0.01 * tanh(f_ba(ba, acc, acc_dot) / 0.01)
no BA gate
no bias-state auxiliary supervision
```

Strict acc-stage MSE:

```text
window: w64
batch size: 1000
mse_residual_mode: right
rot weight: 0
vel weight: 1
pos weight: 10
```

Strict sparse GT NLL:

```text
nll_gt_mode: sparse
nll_gt_residual_mode: right
mse_residual_mode: right
nll_mse_anchor_weight: 0.001
nll_freeze_acc: true
```

## Double-Adjoint Memory Claim

Use `Euroc_double_adjoint_training_eval.md` for the current controlled memory
benchmark:

| Stage | AD peak MB | Double-adjoint peak MB | Reduction |
| --- | ---: | ---: | ---: |
| MSE acc w64 b1000 | 1183.894 | 371.603 | about 68.6% lower |
| Sparse NLL w64 b1000 | 915.869 | 383.070 | about 58.2% lower |

The accuracy claim should stay separate from the efficiency claim. Double
adjoint is an implementation of the same loss/optimizer, not a new loss.

## Historical Or Ablation Records

These files are useful for tracing decisions, but should not be cited as the
current strict SE23 paper mainline without the status notes inside them:

| File | Status |
| --- | --- |
| `Euroc_gt_summary.md` | Historical broad GT summary. Starts from older hybrid/no-gate experiments; strict mainline supersedes it. |
| `sparse_gt_nll_experiment_2026-05-21.md` | Historical sparse NLL search before strict-right NLL became the paper branch. |
| `debias_imu_failed_attempts_cleanup_2026-05-21.md` | Failed or discarded experiment log. |
| `Euroc_vo32_mse_summary.md` | VO32 supervision record; useful for VO branch, not GT strict mainline. |
| `openvins_debiased_imu_exports.md` | Export manifest notes. Some exports were generated from older fair no-gate/VO branches. |
| `debias_imu_paper_code_notes.md` | Code organization notes. |
| `sparse_gt_se23_nll_for_codex.md` | NLL design specification. |
| `sparse_noisy_vo_nll_for_codex.md` | VO NLL design specification. |
| `instruction519.md` | Older instruction/context note. |

## Future Record Rules

Every new experiment record should include:

```text
1. supervision source: GT / VO
2. residual mode: right / log / hybrid / global
3. window definition and batch size
4. BG and BA network types
5. initialization checkpoints
6. loss weights and NLL covariance settings
7. gradient mode: autograd / double_adjoint
8. exact train and eval result directories
9. five-sequence mean AOE, velocity RMSE, and APE
10. whether the result is paper mainline, ablation, diagnostic, or failed
```

