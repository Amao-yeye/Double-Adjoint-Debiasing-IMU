# EuRoC Train-Global Prior Screens - 2026-05-19

## Goal

Check whether a fair train-only bias prior can replace the rejected test-GT
median initialization while preserving the current research direction:

- no `u_dot`;
- constant bias must remain a valid solution;
- select by held-out long-horizon pure IMU AOE / velocity ATE / position ATE;
- do not optimize or select by bias-trend similarity.

Baseline references:

| Method | AOE deg | ATE vel | ATE pos m |
| --- | ---: | ---: | ---: |
| RSS/original `Euroc_baseline_w64` | 3.494 | 16.295 | 678.010 |
| `Euroc_SE23_w64` | 3.494 | 16.070 | 672.283 |

## Diagnostics

Train-only global bias was computed from five training sequences, excluding
`V2_03_difficult` because the retained cache does not contain that sequence:

```text
bw = [-0.00250848, 0.02143016, 0.07812129]
ba = [-0.01726828, 0.12701823, 0.06936392]
```

| Diagnostic | AOE deg | ATE vel | ATE pos m | Interpretation |
| --- | ---: | ---: | ---: | --- |
| train-global median constant | 4.758 | 28.755 | 1129.517 | Better than first-bias constant, still not RSS-beating |
| train-global mean constant | 4.987 | 29.881 | 1187.134 | Worse than median |
| first bias exponentially attracted to train-global, lambda `0.005/s` | 12.195 | 78.606 | 3462.599 | Bad, especially MH_02 |
| first bias exponentially attracted to train-global, lambda `0.02/s` | 7.482 | 56.077 | 2594.081 | Bad |
| first bias exponentially attracted to train-global, lambda `0.1/s` | 4.463 | 31.646 | 1407.902 | Orientation improves, position still bad |
| 5 s raw-IMU gyro mean + train-global Acc | 49.553 | 195.682 | 7666.997 | Initial segment is not a safe static gyro estimator |
| 5 s raw-IMU static gyro/Acc estimate | 49.553 | 181.335 | 6857.013 | Also unusable |

## Checkpoint Re-Tests

| Run | Initialization | AOE deg | ATE vel | ATE pos m | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| `EUROC_unified_no_udot_mse_checkpoint_train_global_median_onlytest` | train-global median | 4.758 | 29.901 | 1176.710 | Existing oracle-initialized MSE checkpoint does not transfer to train-global init |
| `EUROC_conservative_residual_lr1e4_train_global_median_onlytest` | train-global median | 5.321 | 36.538 | 1371.041 | Low-LR conservative model is worse than constant |
| `EUROC_conservative_residual_lr5e3_train_global_median_onlytest` | train-global median | 86.109 | 304.776 | 11055.171 | Overfit dynamics remain catastrophic |

The large `bw/ba_func_net_no_udot` train-global checkpoint re-tests were
stopped after about 10 minutes without producing sequence outputs. They were
not needed because their first-bias tests had already failed badly and they are
not the selected conservative architecture.

## Train-Global Y0 Training Screen

`prepare_euroc_train_global_bias_cache.py` created
`results/debias_imu/cache/EUROC_train_global_median_y0_noV203/`, replacing only
training/validation `w_bias/a_bias` arrays with the train-global median. IMU
and pose GT were unchanged. The intent was to make the short-window ODE initial
bias state match the train-only test prior.

Run:

```text
results/debias_imu/EUROC_conservative_residual_train_global_y0_w64_mse_e101/
```

Configuration:

- network: `bw_func_conservative_residual` / `ba_func_conservative_residual`;
- no `u_dot`;
- `loss_window=64` source poses;
- MSE/SE23 only;
- test initialization: `train_global_median`;
- skip plots, but still write numeric `evaluation_summary.txt`.

Training:

| Stage | Best epoch | Best val |
| --- | ---: | ---: |
| Gyro | 90 | 28.3701 |
| Acc | 60 | 23979.0901 |

Held-out test:

| Sequence | AOE deg | ATE vel | ATE pos m |
| --- | ---: | ---: | ---: |
| MH_02_easy | 75.071 | 300.470 | 11874.987 |
| MH_04_difficult | 32.827 | 107.272 | 2499.695 |
| V1_01_easy | 63.740 | 270.991 | 8553.733 |
| V1_03_difficult | 30.372 | 60.469 | 1920.083 |
| V2_02_medium | 38.820 | 91.441 | 2520.624 |
| mean | 48.166 | 166.129 | 5473.824 |

Conclusion: this train-global-y0 MSE training is a negative result. It confirms
that short-window validation can improve while full-sequence rollout is
destroyed.

## Current Conclusion

- A single train-only global constant prior is fair but not accurate enough.
- Simple attraction from first bias to the train-global prior is not enough.
- Raw IMU initial-window bias estimation is unsafe on the current EuRoC test
  sequences.
- Re-training conservative residual dynamics with train-global y0 worsens
  full-trajectory rollout badly.

Next useful direction is not freer residual dynamics. It should keep the
constant solution but add explicit full-rollout or much longer-horizon
selection/regularization, because short-window SE23 validation is repeatedly
misleading on EuRoC.
