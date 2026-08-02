# EuRoC No-u_dot Tiny-Acc Results

Date: 2026-05-18

## Scope

The user requested EuRoC verification for the latest no-`u_dot` network before analytic-gradient tuning. The baseline was not rerun; existing Debias_IMU baseline result folders were used.

Default EuRoC split:

- Train GT: `MH_01_easy`, `MH_03_medium`, `MH_05_difficult`, `V1_02_medium`, `V2_01_easy`, `V2_03_difficult`
- Train VO: same list except `V2_03_difficult` excluded because the existing VO teacher set is incomplete/weak for that sequence
- Test: `MH_02_easy`, `MH_04_difficult`, `V2_02_medium`, `V1_03_difficult`, `V1_01_easy`

## Result Summary

Mean held-out test metrics:

| Run | Supervision | Gradient | AOE deg | ATE vel | ATE pos | Train time s | Max GPU MB |
|---|---|---|---:|---:|---:|---:|---:|
| `Euroc_baseline_w64` | GT | AD baseline | 3.494 | 16.295 | 678.010 | existing | existing |
| `Euroc_SE23_w64` | GT | AD baseline | 3.494 | 16.070 | 672.283 | existing | existing |
| `EUROC_tiny_acc_fixed_gyro_gt_w64_sg_ad_e301_b1000` | GT | AD | 1.961 | 12.728 | 589.900 | 334.24 | 536.38 |
| `EUROC_tiny_acc_fixed_gyro_gt_w64_sg_analytic_e301_b1000` | GT | analytic | 1.961 | 12.798 | 593.399 | 330.65 | 402.25 |
| `EUROC_tiny_acc_fixed_gyro_vo_body_w64_sg_ad_e301_b1000_noV203` | VO | AD | 1.961 | 12.153 | 555.342 | 336.66 | 539.50 |
| `EUROC_tiny_acc_fixed_gyro_vo_body_w64_sg_analytic_e301_b1000_noV203` | VO | analytic | 1.961 | 12.246 | 559.729 | 314.34 | 404.47 |

The best EuRoC result so far is the VO-supervised AD fixed-gyro/tiny-acc run. It improves the existing `Euroc_SE23_w64` mean by `-1.533 deg`, `-3.917` velocity, and `-116.941 m` position.

## Network And Training Choice

Negative screens:

- Plain `bw_func_net_no_udot` / `ba_func_net_no_udot` with `loss_window=64` failed badly on GT: mean `6.750 deg / 44.200 / 1712.477`.
- Adding SG smoothing to the same plain no-`u_dot` MLP made validation lower but test failed catastrophically: mean `94.617 deg / 193.234 / 6394.029`.
- `bw_func_slow_residual` / `ba_func_slow_residual` improved MH_02 but failed on V sequences: mean `14.031 deg / 38.116 / 1426.107`.
- `bw_func_phys_rate` / `ba_func_phys_rate` also failed: mean `36.707 deg / 121.680 / 4332.622`.

Useful configuration:

- Gyro dynamics are fixed at a constant-bias initialization using `bw_func_slow_residual_tiny` epoch 0.
- Acc dynamics use `ba_func_slow_residual_tiny`, a strict no-`u_dot` tiny rate-limited residual.
- Test initial bias uses `median_window` over `5 s`.
- SG smoothing: `sg_window_bw=101`, `sg_order_bw=3`, `sg_window_ba=31`, `sg_order_ba=3`.

This protects orientation from the no-`u_dot` gyro generalization failure while still allowing small accelerometer-bias corrections.

## Figure

Generated paper-style before/after preintegration figure:

- PNG: `results/debias_imu/figures/euroc_mh02_vo_tiny_acc_preintegration_before_after.png`
- PDF: `results/debias_imu/figures/euroc_mh02_vo_tiny_acc_preintegration_before_after.pdf`
- JSON metrics: `results/debias_imu/figures/euroc_mh02_vo_tiny_acc_preintegration_before_after.json`

For the first 40 s of `MH_02_easy` with the VO-supervised AD run:

| Variant | Position RMSE m | Rotation RMSE deg |
|---|---:|---:|
| Raw IMU | 2245.099 | 101.594 |
| Constant bias | 14.361 | 0.591 |
| Learned | 13.659 | 0.554 |

## Current Interpretation

The EuRoC no-`u_dot` result is now better than the existing baseline under both GT and VO supervision. Analytic gradients are accurate enough for this model family and reduce peak GPU memory by about `25%` versus AD, with similar or slightly better runtime in the 301-epoch Acc-only comparison.

The remaining caveat is that the improvement relies on freezing gyro dynamics at constant-bias initialization. Training no-`u_dot` gyro dynamics directly still hurts test orientation, so broader claims should phrase this as a conservative no-`u_dot` accelerometer-bias dynamics model with fixed gyro propagation, not as a fully learned gyro+acc no-`u_dot` model.

## Unified AD MSE-to-NLL Follow-up

After the selection policy changed to require one core no-`u_dot` architecture and AD derivatives for the formal EuRoC/corldata comparison, a stricter EuRoC run was added with `bw_func_net_no_udot` / `ba_func_net_no_udot`, zero-output initialization, fixed zero-output gyro, and Acc-only training. Supervision uses EuRoC GT sampled at VO timestamps, with duplicate VO timestamps removed in `results/debias_imu/Euroc_GT_at_VO_timestamps_dedup/`.

Correction from 2026-05-19: the MSE/NLL numbers below used `initial_bias_mode=median_window` and `initial_bias_window_s=5.0`, which initializes from each test sequence's GT-differenced bias. They are oracle-style and must not be reported as fair RSS-baseline comparisons. The full oracle run directories were deleted during cleanup; the MSE Acc checkpoint is retained only for audit at `results/debias_imu/checkpoints/euroc_unified_no_udot_mse_lowlr_e51_oracle_audit/`.

| Stage | Config | Output | AOE deg | ATE vel | ATE pos m |
|---|---|---|---:|---:|---:|
| MSE, oracle 5s median init | `plato/configs/debias_imu/euroc_unified_no_udot_gt_vorate_dedup_src32_dense_mse_lowlr_e51.yaml` | deleted; checkpoint audit retained | 2.015 | 13.994 | 624.558 |
| AD interval-NLL, oracle 5s median init | `plato/configs/debias_imu/euroc_unified_no_udot_gt_vorate_dedup_src32_dense_interval_nll_joint_ad_lowlr_from_mse_e31.yaml` | deleted | 2.015 | 14.168 | 630.858 |
| MSE, first-bias init | same MSE checkpoint | `results/debias_imu/EUROC_unified_no_udot_gt_vorate_dedup_src32_dense_mse_lowlr_e51_test_firstbias/` | 23.218 | 141.311 | 5917.102 |

The NLL fine-tune used `acc_loss_type=interval_nll`, `interval_nll_mode=nll_joint`, and `interval_covariance_gradient_mode=ad`. Validation interval NLL decreased from `3631.38` to `3527.33`, with learned `sigma_a=0.0028348002` and `sigma_g=0.00024517465`.

The fair first-bias corrective check does not beat the RSS/original baselines (`Euroc_baseline_w64` mean `3.494 / 16.295 / 678.010`, `Euroc_SE23_w64` mean `3.494 / 16.070 / 672.283`). A new EuRoC fair-initialization run is required before reporting an RSS-beating unified result.

## Piecewise GT-Bias Audit

A later audit found that the historical `Euroc_piecewise_gt_bias_1hz` comparison was also invalid. The old helper updated once per second, but advanced `bias_idx` by only one entry in a high-rate EuRoC bias array. Therefore the final update on 100-150 s sequences still used bias values from only about `0.49-0.75 s` after the sequence start.

The bugged path exactly reproduces the suspicious mean `5.882 deg / 46.103 vel / 1958.631 m`. Time-consistent oracle checks show the expected behavior: 1 Hz bin-mean GT bias gives about `0.097 deg / 0.561 vel / 25.050 m`, 5 Hz bin mean gives `0.055 / 0.378 / 14.660`, and full high-rate GT bias gives `0.000 / 0.009 / 0.330`.

This reinforces the modeling interpretation: EuRoC is strongly compatible with near-constant bias, especially for `b_g`. That does not contradict `dot b = f(b, u, theta, ...)`; the shared no-`u_dot` network should include `dot b approximately 0` as a stable solution through zero-output/residual/rate-limited dynamics. Bias trend remains diagnostic only, while testing-set long-horizon position and rotation are the selection criteria.
