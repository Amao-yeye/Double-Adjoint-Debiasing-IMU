# corldata OpenVINS Learned-IMU Lessons

Date: 2026-05-17

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
