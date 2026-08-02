# EuRoC Fair First-Bias Screens - 2026-05-19

## Goal

Test whether a no-`u_dot` network can improve EuRoC long-horizon pure IMU rollout under a fairer test protocol than the rejected oracle 5 s median initialization.

Protocol:

- test initialization: `initial_bias_mode=first`;
- no test-window median/mean GT-differenced bias;
- source-timing GT sampled at VO timestamps from `results/debias_imu/Euroc_GT_at_VO_timestamps_dedup`;
- test split: `MH_02_easy`, `MH_04_difficult`, `V1_01_easy`, `V1_03_difficult`, `V2_02_medium`;
- primary metric: full-sequence testing AOE / ATE velocity / ATE position.

Baseline references:

| Method | AOE deg | ATE vel | ATE pos m |
|---|---:|---:|---:|
| RSS/original `Euroc_baseline_w64` | 3.494 | 16.295 | 678.010 |
| `Euroc_SE23_w64` | 3.494 | 16.070 | 672.283 |

## Code Changes

- Added shared conservative residual no-`u_dot` network names:
  - `bw_func_conservative_residual`
  - `ba_func_conservative_residual`
- Added EuRoC `--val-seqs` support so checkpoint selection can use sequence-level validation instead of only the original same-sequence tail split.
- Added config records under `plato/configs/debias_imu/` for conservative residual, large no-`u_dot`, and sequence-level validation screens.

## Results

| Run | Network | LR | Validation policy | AOE deg | ATE vel | ATE pos m | Gyro best epoch | Acc best epoch |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `EUROC_conservative_residual_gt_vorate_src32_mse_first_e101` | conservative residual | 1e-4 | same-sequence tail | 15.956 | 91.609 | 3888.399 | 0 | 100 |
| `EUROC_conservative_residual_gt_vorate_src32_mse_first_lr5e3_e101` | conservative residual | 5e-3 | same-sequence tail | 86.154 | 312.634 | 11157.326 | 100 | 80 |
| `EUROC_no_udot_gt_vorate_src32_mse_first_lr1e3_e101` | large no-`u_dot` | 1e-3 | same-sequence tail | 50.703 | 272.202 | 9533.365 | 100 | 100 |
| `EUROC_no_udot_gt_vorate_src32_mse_first_lr1e3_seqvalV201_e101` | large no-`u_dot` | 1e-3 | held-out `V2_01_easy` | 66.882 | 380.405 | 13819.146 | 90 | 90 |

For the low-LR conservative residual run, first-bias constant on the same output directory is:

```text
constant mean 15.450 deg / 90.117 vel / 3866.079 m
```

So that run is essentially constant-bias behavior and does not improve it.

## Interpretation

- The conservative residual structure correctly preserves constant-bias behavior at low learning rate, but it does not improve EuRoC full-sequence test rollout.
- Raising learning rate lets the same conservative structure reduce same-sequence validation loss, but it catastrophically fails on held-out test sequences.
- The larger `bw/ba_func_net_no_udot` network, which is closer to the current corldata best family, reduces validation loss even more but also fails on EuRoC held-out test.
- Sequence-level validation support works technically, but using only `V2_01_easy` as validation still did not prevent test failure.

Current conclusion:

- These fair-first EuRoC screens do not beat RSS/original Debias_IMU and should not be reported as positive results.
- EuRoC still appears to need either a better fair initial-bias policy or a more constrained bias-attractor model that learns a stable train-set bias/equilibrium without arbitrary sequence-specific dynamics.
- The modeling principle remains valid: the network must allow `dot b approximately 0`. The failed high-LR runs show the complementary risk: if the model is allowed to move too freely, it can learn validation-improving dynamics that destroy cross-sequence rollout.

Recommended next EuRoC direction:

1. Keep no-`u_dot`, MSE -> NLL, and testing pos/rot as primary metrics.
2. Replace free MLP gyro dynamics with a constrained train-set bias-attractor or global-equilibrium residual model.
3. Use multi-sequence validation or leave-one-sequence validation across the EuRoC training set for checkpoint selection.
4. Report oracle median/piecewise GT-bias diagnostics separately from fair first-bias or train-only-initialized runs.
