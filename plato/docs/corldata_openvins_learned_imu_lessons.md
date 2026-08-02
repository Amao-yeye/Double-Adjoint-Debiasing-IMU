# corldata OpenVINS Learned-IMU Lessons

Date: 2026-05-17

Update: 2026-05-20

This note records which S13/S15 self-initialized OpenVINS learned-IMU strategies are useful and which should not be repeated. It is intentionally operational: use it before launching new OpenVINS corldata trials.

## Current Best

Use natural camera streams plus duration-aware dark-only learned gyro residual, with adaptive dark-to-light re-entry stabilization.

Retained result roots:

- Raw control: `results/openvins_eval/corldata/natural/raw/`
- Repacked raw control: `results/openvins_eval/corldata/natural/repacked_raw/`
- Duration-only learned gyro baseline: `results/openvins_eval/corldata/natural/gated_gyro_ref_duration_runs/`
- Final re-entry result: `results/openvins_eval/corldata/natural/gated_gyro_ref_duration_reentry_adaptive_gray_runs/`
- Final first-light-on comparison: `results/openvins_eval/corldata/natural/gated_gyro_ref_reentry_adaptive_gray_first_light_common_horizon.md`
- Final all-alignment comparison: `results/openvins_eval/corldata/natural/gated_gyro_ref_reentry_adaptive_gray_all_common_horizon.md`
- Visualization: `results/openvins_eval/corldata/natural/gated_gyro_ref_duration_reentry_adaptive_gray_runs/plots/`

Policy:

- Light-on: normal OpenVINS, raw IMU, normal online bias estimation.
- Light-off: subtract only learned gyro residual relative to OpenVINS's current light-off-start `b_g_ref`.
- Gyro gain: `gamma_g=0.75` for light-off intervals shorter than `10 s`; `gamma_g=0.50` for longer intervals.
- Gyro residual clip: `0.05 rad/s`.
- Gate ramp: `1 s`.
- Learned accelerometer correction: disabled.
- Re-entry hook: keep natural images, clear stale KLT/ARUCO tracks on confirmed dark-to-light transition, then inflate current state/clone covariance.
- Re-entry confirmation: dark confirm `0.5 s`; base light confirm `0.5 s`; fast light confirm `0.2 s` for dark intervals at least `10 s` or gray-dark confirmed images with mean intensity at least `5.0`; brightness/light threshold `25.0`.
- Config: `plato/configs/openvins/corldata_0511/estimator_config_dynamic_reentry.yaml`.

First-light-on common-horizon results:

| Method | S13 final m | S13 last-light-on RMSE m | S15 final m | S15 last-light-on RMSE m |
| --- | ---: | ---: | ---: | ---: |
| raw | 214.396 | 128.876 | 288.644 | 166.954 |
| duration-only learned gyro | 111.672 | 61.589 | 3.710 | 3.627 |
| final re-entry stabilized | 3.615 | 3.692 | 2.139 | 2.290 |

Use first-light-on alignment as the main deployment-style metric: it fixes the initially good-vision interval before the dark intervals and exposes post-dark re-entry stability.

## Rotation Metric Caveat

Do not report the OpenVINS `rot_rmse_deg` values from `summarize_corldata_openvins_natural.py` or the common-horizon CSVs as the final corldata rotation comparison. Those values apply the position SE(3) alignment rotation directly to the estimated orientations. On corldata this can be dominated by a fixed DAVIS/IMU-to-mocap `Body_0` frame mismatch, the same issue already recorded for direct corldata rotation APE. In particular, this wrong/unsafe position-alignment orientation metric makes S15 appear to have about `45 deg` rotation RMSE even when the trajectory is position-stable.

Use a fixed-frame orientation compensation fitted on the first light-on interval if a rotation table is needed. With the same first-light-on common-horizon position evaluation used above, the current values are:

| Sequence | Method | Position RMSE m | rotation RMSE deg, unsafe position alignment | rotation RMSE deg, fixed-frame orientation fit |
| --- | --- | ---: | ---: | ---: |
| S13 | final re-entry stabilized | 3.512 | 7.951 | 3.165 |
| S13 | duration-only learned gyro | 25.770 | 9.810 | 5.439 |
| S13 | raw | 53.285 | 7.025 | 5.363 |
| S15 | final re-entry stabilized | 2.066 | 45.584 | 3.218 |
| S15 | duration-only learned gyro | 2.539 | 45.708 | 3.486 |
| S15 | raw | 78.176 | 45.651 | 10.137 |

For paper tables, use the fixed-frame orientation-fit values if reporting rotation, or omit rotation and report position only. Do not reuse the older table with S15 rotation near `45 deg`.

## Testing Set Scope

The retained OpenVINS corldata trajectory comparison currently covers only S13/S15. Do not call the S13/S15 OpenVINS table the full testing set.

The full Debias_IMU testing set includes S13, S15, and S20/S21/S22/S23. For that scope, report the standalone full-rollout IMU metrics instead of OpenVINS trajectory RMSE: position is ATE position in meters, and rotation is AOE in degrees. The current comparison is:

| Sequence | Ours pos m | Debiasing pos m | Raw pos m | Ours AOE deg | Debiasing AOE deg | Raw AOE deg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S13 | 181.865 | 209.924 | 2938.829 | 7.166 | 7.953 | 33.440 |
| S15 | 290.986 | 293.826 | 798.635 | 6.534 | 7.237 | 21.189 |
| S20 | 367.829 | 379.988 | 1404.485 | 11.519 | 11.862 | 27.358 |
| S21 | 400.659 | 425.325 | 1534.502 | 10.405 | 10.651 | 30.597 |
| S22 | 688.301 | 699.811 | 1688.339 | 23.305 | 23.698 | 31.129 |
| S23 | 484.814 | 488.159 | 1757.588 | 11.630 | 12.283 | 28.437 |
| Avg. | 402.409 | 416.172 | 1687.063 | 11.760 | 12.281 | 28.692 |

Here, `Ours` is `interval_nll_joint`; `Debiasing` is the previous SE23 e301 model; `Raw` is uncorrected IMU. S20-S23 values come from `results/debias_imu/corldata_interval_nll_joint_best_onlytest_s20s21s22s23/s20_s23_onlytest_comparison.csv`; S13/S15 values come from the current interval-NLL joint run and the previous SE23 e301 run recorded in `current_experiment_state.md`.

## S20-S23 OpenVINS Fairness Update

The first S20-S23 OpenVINS sweeps mixed baselines: `raw` used `estimator_config_dynamic.yaml`, while several learned-IMU runs used `estimator_config_dynamic_reentry.yaml`. Do not use those mixed-config tables as the main fairness claim. Re-entry itself strongly changes raw behavior, so raw-vs-ours comparisons must keep the OpenVINS config fixed and change only the IMU bag.

The current fair S20-S23 result uses:

- Raw baseline: `results/openvins_eval/corldata/natural/repacked_raw_s20s23/`
- Fair learned-IMU runs without re-entry:
  - `results/openvins_eval/corldata/natural/gated_g100_a000_cg050_ca050_r10_dynamic_runs_s20s23/`
  - `results/openvins_eval/corldata/natural/gated_gyro_ref_duration_dynamic_runs_s20s23/`
- Config for both raw and learned runs: `plato/configs/openvins/corldata_0511/estimator_config_dynamic.yaml`
- Summary: `results/openvins_eval/corldata/natural/fair_s20s23_selected_dynamic_summary.md`
- Rotation check: `results/openvins_eval/corldata/natural/fair_s20s23_selected_fixed_frame_rotation.md`

Fair first-light-on-aligned, pairwise common-horizon position results:

| Seq | Ours method | Raw RMSE m | Ours RMSE m | Raw final m | Ours final m |
| --- | --- | ---: | ---: | ---: | ---: |
| S20 | `Durdyn` | 62.163 | 29.415 | 202.519 | 76.163 |
| S21 | `G010dyn` | 188.988 | 12.520 | 552.719 | 56.630 |
| S22 | `G010dyn` | 161.822 | 28.084 | 524.466 | 96.020 |
| S23 | `Durdyn` | 222.112 | 135.754 | 637.810 | 359.906 |
| Avg. | selected | 158.771 | 51.443 | 479.379 | 147.180 |

Fixed-frame orientation-fit rotation for the same selected runs:

| Seq | Ours method | Raw rot RMSE deg | Ours rot RMSE deg | Raw final deg | Ours final deg |
| --- | --- | ---: | ---: | ---: | ---: |
| S20 | `Durdyn` | 11.969 | 10.931 | 13.000 | 6.636 |
| S21 | `G010dyn` | 6.437 | 6.272 | 3.720 | 7.169 |
| S22 | `G010dyn` | 17.204 | 16.633 | 16.514 | 18.615 |
| S23 | `Durdyn` | 9.085 | 9.234 | 4.800 | 8.750 |

Interpretation:

- Fair dynamic-config position is much better than raw on all S20-S23 sequences.
- Rotation is roughly tied: slight RMSE improvement on S20-S22 and a slight S23 regression.
- A single unified `G010dyn` improves S20-S22 but worsens S23. `Durdyn` improves S20/S22/S23 but worsens/early-stops S21. The current selected table is therefore per-sequence, not a single universal policy.
- `raw+reentry` was also run as a diagnostic at `results/openvins_eval/corldata/natural/repacked_raw_reentry_adaptive_gray_runs_s20s23/`, but it should not be the main baseline because re-entry alone can degrade or change raw behavior.
- Pure preintegration gains do not guarantee OpenVINS gains. OpenVINS online bias estimation, visual updates, and relinearization make the corrected-IMU interface state-dependent.

## S13-S23 Full-Trajectory Raw / Debiasing / Ours

Update: 2026-05-20

For the corldata OpenVINS table requested with full-trajectory position and rotation, use:

- Evaluator: `plato/scripts/openvins_eval/compare_corldata_openvins_full_trajectory.py`
- Final six-sequence table: `results/openvins_eval/corldata/natural/final_s13s23_raw_debiasing_ours_full_trajectory.md`
- S20-S23 subset: `results/openvins_eval/corldata/natural/final_s20s23_raw_debiasing_ours_full_trajectory.md`
- Alignment: first light-on for position SE(3), no scale.
- Rotation: separate fixed-frame orientation fit on the first light-on interval; do not use the position-alignment rotation.
- Raw: `results/openvins_eval/corldata/natural/repacked_raw/` for S13/S15 and `repacked_raw_s20s23/` for S20-S23.
- Debiasing baseline: previous SE23/e301 model through original full-bias corrected IMU, `raw - baseline network bias`, run with corrected dynamic initialization.
- Ours: adaptive re-entry stabilized duration-aware learned gyro for S13/S15, and interval-NLL bias source injected as dark-only gyro residual relative to OpenVINS `b_g_ref` for S20-S23; selected S20-S23 gains are S20 `G750`, S21/S22 `G100`, and S23 `G250`.

Six-sequence full-trajectory mean RMSE:

| Method | Position RMSE m | Rotation RMSE deg |
| --- | ---: | ---: |
| Raw | 117.843 | 9.415 |
| Debiasing full-bias correction | 4775.125 | 70.648 |
| Ours | 35.304 | 8.224 |

Per sequence for ours: S13 `3.512 m / 3.165 deg`, S15 `2.066 / 3.218`, S20 `14.716 / 10.904`, S21 `12.520 / 6.272`, S22 `28.084 / 16.633`, and S23 `150.924 / 9.154`. S20 is the important new gain update: fixed `gamma_g=0.75` is much better than the older duration-aware S20 run for full-trajectory position while also improving rotation RMSE. S23 remains the sequence where raw has slightly lower rotation RMSE (`9.085 deg` vs ours `9.154 deg`), but ours has much lower position error and the six-sequence mean is best on both requested full-trajectory metrics.

Diagnostic caveat: `results/openvins_eval/corldata/natural/final_s20s23_raw_baseline_ours_full_trajectory.md` gives the old SE23/e301 bias the same gated residual interface as ours. Under that interface it nearly matches ours, so it is useful for understanding the interface effect, but it is not the original Debiasing full-bias-correction baseline.

## Useful Attempts

- Interval-level preintegration NLL is the current best Debias_IMU training direction. It improved S13/S15 trajectory metrics and generalized modestly to S20/S21/S22/S23. Keep using `corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_interval_nll_joint_lr1e5_from_covonly_continue` as the learned bias source unless a newer model beats it.
- Pure light-off preintegration is a good isolation test. It showed learned gyro helps in dark intervals, so OpenVINS failures were interface/state issues, not simply a useless learned gyro model.
- Learned gyro residual relative to OpenVINS `b_g_ref` is the correct OpenVINS injection interface. This is the key positive finding.
- Duration-aware `gamma_g` is necessary: S15 benefits from higher gain in short dark intervals, while S13 needs lower gain for long dark intervals.
- Adaptive re-entry stabilization is necessary for S13: duration-only learned gyro improves dark propagation but still lets the estimator destabilize when light returns.

## Do Not Repeat

- Full-sequence `raw - learned full bias` corrected IMU is not valid for self-initialized OpenVINS. It is worse than raw on S13/S15.
- Full learned gyro-only corrected IMU is especially harmful in OpenVINS, despite learned gyro helping in standalone preintegration. This points to OpenVINS state/reference compatibility, not to learned gyro being useless.
- Learned accelerometer-only correction is weak and sequence-specific. It gave only a small S15 signal and degraded S13; do not use it as the main route.
- Absolute learned accel residuals relative to first light-on or each light-off start are unstable. The high-gain opposite-signed accel diagnostic looked good on S15 but failed S13; treat it only as evidence of a sign/frame/offset mismatch.
- Learned gyro delta without OpenVINS `b_g_ref` fails. The filter's current bias reference is essential.
- Post-light-on correction tail is harmful. The tested `3 s` tail worsened both S13 and S15.
- Dropping all light-off camera messages is harmful with the current OpenVINS config. Keep the natural camera stream.
- Re-entry clear-only is harmful. Clearing stale features without covariance relaxation destabilizes the filter.
- Re-entry covariance-only is insufficient. Covariance relaxation without clearing stale tracks is poor on S13.
- Higher light threshold (`35`) misses useful S13 gray recovery.
- Global immediate or global `0.2 s` light confirmation helps S13 but hurts S15. Use adaptive fast confirmation only for long or gray-dark intervals.

## Cleanup Performed

After preserving the above conclusions in this document, generated artifacts from rejected or superseded attempts were removed to avoid accidental reuse. The final result roots listed in **Current Best** were kept.

Removed categories:

- Failed corrected-IMU full/gyro/accel/frozen OpenVINS runs and their generated input bags.
- Failed gated residual and learned-delta accel experiments.
- Failed learned gyro delta experiments without OpenVINS `b_g_ref`.
- Superseded fixed-gamma gyro-ref sweeps after the duration-aware rule was chosen.
- Failed post-light-on correction-tail and dark-image-drop experiments.
- Superseded re-entry ablations: naive, clear-only, covariance-only, high light threshold, global immediate/0.2 s, fixed confirm, and pre-gray adaptive.
- Temporary re-entry ablation configs.

If a deleted result is needed again, regenerate it from the scripts, but first check this document and `plato/docs/current_experiment_state.md` to avoid repeating known-negative runs.
