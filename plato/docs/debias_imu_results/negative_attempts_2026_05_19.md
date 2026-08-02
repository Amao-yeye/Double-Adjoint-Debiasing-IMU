# Negative And Misleading Debias_IMU Attempts - 2026-05-19

## EuRoC

Misleading result:

- The unified no-`u_dot` EuRoC MSE/NLL runs with `initial_bias_mode=median_window`, `initial_bias_window_s=5.0` used test-sequence GT-differenced bias to initialize `bg0/ba0`.
- Under this oracle-style initialization, constant bias reached about `2.015 deg / 12.043 vel / 555.868 m`, and the network reached about `2.015 deg / 13.994 vel / 624.558 m`.
- This should not be compared directly to RSS/original Debias_IMU baselines.

Corrective check:

- Same MSE checkpoint, `initial_bias_mode=first`, output `EUROC_unified_no_udot_gt_vorate_dedup_src32_dense_mse_lowlr_e51_test_firstbias/`.
- Mean network result: about `23.218 deg / 141.311 vel / 5917.102 m`.
- Per-sequence first-bias network result:
  - MH_02 `57.954 deg / 476.742 / 22019.121`
  - MH_04 `16.952 / 55.464 / 1635.169`
  - V1_01 `19.592 / 67.532 / 2316.691`
  - V1_03 `7.592 / 10.190 / 379.929`
  - V2_02 `14.001 / 96.626 / 3234.602`

Other EuRoC negative screens:

- Plain no-`u_dot` MLP with `loss_window=64` failed on GT.
- SG-smoothed plain no-`u_dot` reduced validation but failed badly on test.
- Full slow-residual and phys-rate variants hurt held-out generalization.
- Trajectory-NLL pilot runs were either not AD-compatible with the user's latest constraint or worsened long-horizon drift.
- The old `Euroc_piecewise_gt_bias_1hz` diagnostic result `5.882 deg / 46.103 vel / 1958.631 m` is invalid. Its helper advanced `bias_idx += 1` once per 1 s update while `bias_w_gt/a_gt` were high-rate arrays, so the final update still used bias samples from only about `0.49-0.75 s` into the test sequence. Time-consistent oracle bin-mean piecewise checks are much stronger, e.g. 1 Hz bin mean about `0.097 deg / 0.561 vel / 25.050 m`, but they are oracle diagnostics because they use test GT-differenced bias.
- Fair first-bias no-`u_dot` screens on 2026-05-19 did not beat RSS. Results: conservative residual low LR `15.956 / 91.609 / 3888.399`; conservative residual high LR `86.154 / 312.634 / 11157.326`; large no-`u_dot` LR `1e-3` with same-sequence validation `50.703 / 272.202 / 9533.365`; large no-`u_dot` LR `1e-3` with held-out `V2_01_easy` validation `66.882 / 380.405 / 13819.146`. These runs are recorded in `euroc_fair_first_bias_screens_2026_05_19.md`.
- Train-only global bias prior screens also did not beat RSS. The train-global median constant diagnostic reached `4.758 / 28.755 / 1129.517`, and the existing MSE checkpoint retested with train-global initialization reached `4.758 / 29.901 / 1176.710`. Low-LR conservative residual retested with train-global initialization was worse at `5.321 / 36.538 / 1371.041`; high-LR conservative residual remained catastrophic at `86.109 / 304.776 / 11055.171`.
- Re-training `bw_func_conservative_residual` / `ba_func_conservative_residual` with train/validation bias labels replaced by the train-global median initial condition also failed: `EUROC_conservative_residual_train_global_y0_w64_mse_e101` reached held-out mean `48.166 / 166.129 / 5473.824` despite modest short-window validation improvement. This is recorded in `euroc_train_global_prior_screens_2026_05_19.md`.

Conclusion:

- EuRoC needs a fair initialization policy before it can be reported as RSS-beating.
- Any result using test GT-differenced median bias initialization must be labeled oracle-style.
- EuRoC near-constant bias behavior is not a failure mode by itself. The next fair model should allow `dot b approximately 0` as a stable learned solution rather than forcing nonzero dynamics.
- Same-sequence validation loss can be misleading on EuRoC: high-LR conservative residual and large no-`u_dot` both improved validation while destroying held-out full-sequence rollout.
- Train-global initialization is fairer than test-GT median initialization, but it is not sufficient by itself. Future EuRoC work needs full-rollout-aware selection or regularization, not just another short-window MSE screen.

## corldata

Useful result retained:

- `corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_interval_nll_joint_lr1e5_from_covonly_continue/`
- S13/S15 improved over the previous SE23 checkpoint:
  - S13 `7.166 deg / 15.995 vel / 181.865 m`
  - S15 `6.534 / 18.101 / 290.986`
- S20-S23 only-test showed modest but consistent gains over SE23.

Negative or superseded corldata attempts:

- Original-style 1000 Hz baseline `w128/e1801` was fair but weak on full-trajectory velocity/position; retained as baseline evidence, not selected model.
- `w64/e300` original baseline is superseded by `w128/e1801`.
- Endpoint NLL, q-only endpoint NLL, anchored endpoint NLL, and Acc-only endpoint NLL did not replace interval-NLL.
- `nll_train_gyro` endpoint variants degraded S13/S15 in tested settings.
- M2/no-derivative, phys-rate, phys-delta, gyro-screen, and no-velocity VO-supervision ablations were worse or not selected.
- The later six-sequence corldata full-test under `corldata_sparse_endpoint_no_udot_w256_interval_nll_joint_fulltest/` was stopped by the user and contains only partial outputs; do not report it as a complete test.

## LIASuw

Retained references:

- First AD run, M2 no-derivative run, strict baseline, and selected phys-delta/VO-pseudoGT runs.

Negative or process attempts deleted:

- smoke runs;
- routeA gravity/world-align smoke and full attempts;
- phys-rate/phys-eq probes;
- short probe runs that only verified plumbing or produced worse drift.

Conclusion:

- LIASuw remains useful as pipeline evidence and OpenVINS/dropout evidence, but many intermediate Debias_IMU training probes were process artifacts and are not needed as full result directories.
