# Euroc Formal Rollout Window Evaluation

Errors are averaged over all non-overlapping test windows and propagated state timestamps inside each window.
Rotation error is `||Log(R_gt^T R_hat)||`. The initialized first state is excluded from the primary mean columns; including-initial diagnostics are saved in CSV.

Ours source: `/home/amao/workspace/PLATO/results/debias_imu/EUROC_conservative_residual_train_global_y0_fullrollout_v201_w256_mse_e31`

| L steps | method | windows | states | mean rot rad | mean rot deg | mean vel m/s | mean pos m |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | raw | 2447 | 122350 | 0.010256 | 0.5877 | 0.0398 | 0.0033 |
| 50 | ours | 2447 | 122350 | 0.000781 | 0.0447 | 0.0225 | 0.0022 |
| 100 | raw | 1222 | 122200 | 0.020283 | 1.1621 | 0.0902 | 0.0141 |
| 100 | ours | 1222 | 122200 | 0.001308 | 0.0750 | 0.0419 | 0.0075 |
| 150 | raw | 813 | 121950 | 0.030283 | 1.7351 | 0.1549 | 0.0350 |
| 150 | ours | 813 | 121950 | 0.001766 | 0.1012 | 0.0610 | 0.0160 |
| 200 | raw | 609 | 121800 | 0.040253 | 2.3063 | 0.2338 | 0.0685 |
| 200 | ours | 609 | 121800 | 0.002220 | 0.1272 | 0.0798 | 0.0276 |

Files:

- `summary.csv`: aggregate over all test sequences.
- `per_sequence_summary.csv`: aggregate per sequence.
- `per_window_summary.csv`: one row per sampled window and method.
- `metadata.json`: protocol details.
