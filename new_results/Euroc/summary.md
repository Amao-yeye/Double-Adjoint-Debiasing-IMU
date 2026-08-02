# Euroc Rollout Samples

Ours best source:

`results/debias_imu/EUROC_conservative_residual_train_global_y0_fullrollout_v201_w256_mse_e31`

Sequences sampled: `MH_02_easy`, `MH_04_difficult`, `V1_01_easy`, `V1_03_difficult`, `V2_02_medium`.

## Formal non-overlapping window evaluation

This is the Altawaitan-style EuRoC comparison: non-overlapping IMU windows, GT initialization at each window start, and mean error over propagated timestamps in all test windows. The initialized first state is excluded from the primary mean.

Formal outputs:

- [formal_windows/summary.md](formal_windows/summary.md)
- [formal_windows/summary.csv](formal_windows/summary.csv)
- [formal_windows/per_sequence_summary.csv](formal_windows/per_sequence_summary.csv)
- [formal_windows/per_window_summary.csv](formal_windows/per_window_summary.csv)

| L steps | method | windows | propagated states | mean rot rad | mean rot deg | mean vel m/s | mean pos m |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | raw | 2447 | 122350 | 0.010256 | 0.5877 | 0.0398 | 0.0033 |
| 50 | ours | 2447 | 122350 | 0.000781 | 0.0447 | 0.0225 | 0.0022 |
| 100 | raw | 1222 | 122200 | 0.020283 | 1.1621 | 0.0902 | 0.0141 |
| 100 | ours | 1222 | 122200 | 0.001308 | 0.0750 | 0.0419 | 0.0075 |
| 150 | raw | 813 | 121950 | 0.030283 | 1.7351 | 0.1549 | 0.0350 |
| 150 | ours | 813 | 121950 | 0.001766 | 0.1012 | 0.0610 | 0.0160 |
| 200 | raw | 609 | 121800 | 0.040253 | 2.3063 | 0.2338 | 0.0685 |
| 200 | ours | 609 | 121800 | 0.002220 | 0.1272 | 0.0798 | 0.0276 |

## Formal V102/V103/V202 subset

This is the user-specified subset `V1_02_medium`, `V1_03_difficult`, and `V2_02_medium`, with all non-overlapping windows pooled before averaging. `V1_02_medium` was first densified from 10 Hz GT to 200 Hz IMU timestamps in an isolated cache.

Subset outputs:

- [formal_windows_v102_v103_v202/summary.md](formal_windows_v102_v103_v202/summary.md)
- [formal_windows_v102_v103_v202/summary.csv](formal_windows_v102_v103_v202/summary.csv)
- [formal_windows_v102_v103_v202/per_sequence_summary.csv](formal_windows_v102_v103_v202/per_sequence_summary.csv)
- [formal_windows_v102_v103_v202/per_window_summary.csv](formal_windows_v102_v103_v202/per_window_summary.csv)

| L steps | method | windows | propagated states | mean rot rad | mean rot deg | mean vel m/s | mean pos m |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | raw | 1213 | 60650 | 0.010339 | 0.5924 | 0.0270 | 0.0023 |
| 50 | ours | 1213 | 60650 | 0.001071 | 0.0614 | 0.0150 | 0.0015 |
| 100 | raw | 606 | 60600 | 0.020385 | 1.1680 | 0.0631 | 0.0097 |
| 100 | ours | 606 | 60600 | 0.001659 | 0.0951 | 0.0265 | 0.0048 |
| 150 | raw | 403 | 60450 | 0.030395 | 1.7415 | 0.1129 | 0.0246 |
| 150 | ours | 403 | 60450 | 0.002137 | 0.1224 | 0.0380 | 0.0100 |
| 200 | raw | 302 | 60400 | 0.040348 | 2.3118 | 0.1768 | 0.0496 |
| 200 | ours | 302 | 60400 | 0.002616 | 0.1499 | 0.0495 | 0.0172 |

## Sampled visualization outputs

Each rollout length uses 3 high-motion samples per sequence, 15 samples total. Plots show only ground truth, raw IMU, and ours.

| rollout | states | mean raw end err m | mean ours end err m | report |
| --- | ---: | ---: | ---: | --- |
| 50 steps | 51 | 0.020 | 0.017 | [HTML](rollout50steps/html/index.html) |
| 100 steps | 101 | 0.053 | 0.033 | [HTML](rollout100steps/html/index.html) |
| 150 steps | 151 | 0.112 | 0.056 | [HTML](rollout150steps/html/index.html) |
| 200 steps | 201 | 0.222 | 0.086 | [HTML](rollout200steps/html/index.html) |

Endpoint rotation/velocity/position error summary:

- [rollout_error_summary.md](rollout_error_summary.md)
- [rollout_error_summary.csv](rollout_error_summary.csv)
- [rollout_error_per_segment.csv](rollout_error_per_segment.csv)

Per-step outputs:

- `rollout50steps/`
- `rollout100steps/`
- `rollout150steps/`
- `rollout200steps/`
