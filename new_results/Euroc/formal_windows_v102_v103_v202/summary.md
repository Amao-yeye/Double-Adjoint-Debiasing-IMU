# Euroc Formal Rollout Window Evaluation

Errors are averaged over all non-overlapping test windows and propagated state timestamps inside each window.
Rotation error is `||Log(R_gt^T R_hat)||`. The initialized first state is excluded from the primary mean columns; including-initial diagnostics are saved in CSV.

Ours source: `new_results/Euroc/formal_windows_v102_v103_v202/model_rollouts`

Sequences: `V1_02_medium`, `V1_03_difficult`, `V2_02_medium`.

Cache note: `V1_02_medium` in the existing `EUROC_test_gt` cache had 10 Hz GT/bias samples. For this report, its GT was interpolated to the 200 Hz IMU timestamps in `test_cache_dense` before model rollout and window evaluation.

| L steps | method | windows | states | mean rot rad | mean rot deg | mean vel m/s | mean pos m |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | raw | 1213 | 60650 | 0.010339 | 0.5924 | 0.0270 | 0.0023 |
| 50 | ours | 1213 | 60650 | 0.001071 | 0.0614 | 0.0150 | 0.0015 |
| 100 | raw | 606 | 60600 | 0.020385 | 1.1680 | 0.0631 | 0.0097 |
| 100 | ours | 606 | 60600 | 0.001659 | 0.0951 | 0.0265 | 0.0048 |
| 150 | raw | 403 | 60450 | 0.030395 | 1.7415 | 0.1129 | 0.0246 |
| 150 | ours | 403 | 60450 | 0.002137 | 0.1224 | 0.0380 | 0.0100 |
| 200 | raw | 302 | 60400 | 0.040348 | 2.3118 | 0.1768 | 0.0496 |
| 200 | ours | 302 | 60400 | 0.002616 | 0.1499 | 0.0495 | 0.0172 |

Files:

- `summary.csv`: aggregate over all test sequences.
- `per_sequence_summary.csv`: aggregate per sequence.
- `per_window_summary.csv`: one row per sampled window and method.
- `metadata.json`: protocol details.
