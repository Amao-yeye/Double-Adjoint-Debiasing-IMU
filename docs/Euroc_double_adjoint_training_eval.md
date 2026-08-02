# EuRoC Double-Adjoint Training/Evaluation

Date: 2026-05-21

Goal: compare the existing AD training path against analytical/double-adjoint
training while keeping the optimizer and training setup fixed. The intended
controlled variable is only the gradient computation method.

## Gradient Setting

Implemented mode:

```text
--gradient_mode double_adjoint
```

MSE uses the strict right-invariant SE23 residual:

```text
r = Log(X_gt X_hat^{-1})
```

The analytical residual Jacobian uses strict BCH:

```text
H = -dexp^{-1}_{-r}
```

## MSE Experiment

Reference AD result:

```text
external/Debias_IMU/results/Euroc_PLATO_strict_right_ba_rate_tanh1e2_fromV1P10_lr1e6_w64_b1000_e101_eval_best
```

Double-adjoint result:

```text
external/Debias_IMU/results/Euroc_PLATO_strict_right_double_adjoint_ba_rate_tanh1e2_fromV1P10_lr1e6_w64_b1000_e101
external/Debias_IMU/results/Euroc_PLATO_strict_right_double_adjoint_ba_rate_tanh1e2_fromV1P10_lr1e6_w64_b1000_e101_eval_best
```

Shared training setup:

```text
stage: acc
BG network: bw_func_net_damped_145e3
BA network: ba_func_net_rate_tanh_1e2
BG checkpoint: results/Euroc_baseline_w64/Gyro_weights/model_epoch_1800.pt
BA init checkpoint: results/Euroc_PLATO_acc_se23_hybrid_V1P10_w64_b1000/Acc_weights/best_val_model.pt
window: w64
batch size: 1000
epochs: 101
lr_Acc: 1e-6
weight_decay_Acc: 1e-6
optimizer: Adam
scheduler: StepLR(step_size=300, gamma=0.5)
mse_residual_mode: right
acc loss weights: rot 0, vel 1, pos 10
```

Mean test metrics:

| Run | AOE deg | Vel RMSE | APE m |
| --- | ---: | ---: | ---: |
| Baseline w64 | 3.392720866 | 16.372752047 | 683.675057983 |
| MSE AD strict-right | 3.043793869 | 16.384823990 | 644.772181702 |
| MSE double-adjoint strict-right | 3.043793869 | 16.389042807 | 644.928808594 |

Relative to AD:

```text
AOE diff: 0.000000 deg
Vel RMSE diff: +0.004219
APE diff: +0.156627 m
APE relative diff: +0.024292% worse
```

This is effectively the same precision for the full-trajectory EuRoC test
metric.

## NLL Experiment

Reference AD result from the previous best no-gate sparse GT NLL:

```text
external/Debias_IMU/results/Euroc_PLATO_sparse_gt_nll_logres_from_nogate_rate_tanh1e2_w64_b1000_lr1e6_anchor001_freezeA_eval_e40
```

Double-adjoint result:

```text
external/Debias_IMU/results/Euroc_PLATO_sparse_gt_nll_logres_double_adjoint_from_nogate_rate_tanh1e2_w64_b1000_lr1e6_anchor001_rightmse_freezeA_e51
external/Debias_IMU/results/Euroc_PLATO_sparse_gt_nll_logres_double_adjoint_from_nogate_rate_tanh1e2_w64_b1000_lr1e6_anchor001_rightmse_freezeA_eval_e40
external/Debias_IMU/results/Euroc_PLATO_sparse_gt_nll_logres_double_adjoint_from_nogate_rate_tanh1e2_w64_b1000_lr1e6_anchor001_rightmse_freezeA_eval_best
```

Training setup:

```text
stage: nll
nll_gt_mode: sparse
nll_gt_residual_mode: log
nll_mse_anchor_weight: 0.001
nll_freeze_acc: true
MSE anchor residual mode: right
MSE anchor weights: rot 1, vel 1, pos 10
window: w64
batch size: 1000
epochs: 51
lr_NLL: 1e-6
weight_decay_NLL: 1e-6
optimizer: Adam
```

Note: the previous AD NLL note recorded a hybrid MSE anchor. This double-adjoint
run uses the strict right SE23-MSE anchor per the later correction that SE23-MSE
should use `Log(X_gt X_hat^{-1})`.

Mean test metrics:

| Run | AOE deg | Vel RMSE | APE m |
| --- | ---: | ---: | ---: |
| NLL AD previous e40 | 3.151669979 | 15.696268368 | 628.975341797 |
| NLL double-adjoint e40 | 3.134353447 | 15.704946995 | 628.781622314 |
| NLL double-adjoint best-val/e50 | 3.139185405 | 15.638582277 | 630.186361694 |

The e40 checkpoint is the recommended analytical NLL checkpoint by full-trajectory
APE. It is close to, and slightly better than, the previous AD NLL test APE.

## Strict-Right NLL From Strict SE23-MSE

A fully strict branch was also run after fixing the paper definition:

```text
init checkpoint:
external/Debias_IMU/results/Euroc_PLATO_strict_right_ba_rate_tanh1e2_fromV1P10_lr1e6_w64_b1000_e101/Acc_weights/best_val_model.pt

training:
external/Debias_IMU/results/Euroc_PLATO_sparse_gt_nll_rightres_double_adjoint_from_strict_right_mse_w64_b1000_lr1e6_anchor001_freezeA_e51

recommended eval:
external/Debias_IMU/results/Euroc_PLATO_sparse_gt_nll_rightres_double_adjoint_from_strict_right_mse_w64_b1000_lr1e6_anchor001_freezeA_eval_e40
```

Strict settings:

```text
nll_gt_mode: sparse
nll_gt_residual_mode: right
mse_residual_mode: right
nll_mse_anchor_weight: 0.001
nll_freeze_acc: true
gradient_mode: double_adjoint
```

Mean test metrics:

| Run | AOE deg | Vel RMSE | APE m |
| --- | ---: | ---: | ---: |
| Strict SE23-MSE AD | 3.043793869 | 16.384823990 | 644.772181702 |
| Strict-right NLL double-adjoint e40 | 3.075663114 | 15.752899933 | 631.278599548 |
| Strict-right NLL double-adjoint best-val/e50 | 3.077915144 | 15.676734161 | 634.265779114 |

The e40 checkpoint is recommended by APE. Relative to strict SE23-MSE AD, it
improves APE by `2.092767%` and velocity RMSE by `3.856764%`, while AOE is
`1.047024%` worse. Relative to baseline, it keeps `9.345235%` AOE improvement
and `7.663942%` APE improvement.

## Memory

Full analytical training runs wrote:

```text
MSE double-adjoint max_memory_allocated: 374.049805 MB
NLL double-adjoint max_memory_allocated: 384.061035 MB
```

A same-configuration one-step benchmark measured forward/backward/optimizer.step
with the same model, loss, batch size, optimizer, and checkpoints:

| Stage | Method | Loss | Time s | Peak allocated MB |
| --- | --- | ---: | ---: | ---: |
| MSE acc w64 b1000 | AD | 42162.9219 | 1.341860 | 1183.894 |
| MSE acc w64 b1000 | double-adjoint | 42162.9219 | 1.000344 | 371.603 |
| Sparse NLL w64 b1000 | AD | 1304.05298 | 0.996880 | 915.869 |
| Sparse NLL w64 b1000 | double-adjoint | 1304.05298 | 1.091313 | 383.070 |

Memory ratios:

```text
MSE acc: 0.3139x AD memory, about 68.6% lower.
Sparse NLL: 0.4183x AD memory, about 58.2% lower.
```

## Conclusion

The analytical/double-adjoint path preserves EuRoC test precision for the
strict-right SE23-MSE experiment and for the sparse GT NLL e40 checkpoint, while
substantially reducing peak GPU memory. This supports using double-adjoint as
an efficiency implementation rather than an accuracy-changing method.
