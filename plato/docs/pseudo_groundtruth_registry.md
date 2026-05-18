# Pseudo-Groundtruth Registry

This registry records which ORB-SLAM3 trajectory results are currently accepted as pseudo-groundtruth for PLATO, and which generated results were rejected and removed to keep the workspace clean.

## Usable Pseudo-GT Windows

The first three entries are user-confirmed usable LIASuw 0912 stereo-only windows. The 0203/0205 entries are newly tested usable-quality candidates and should be confirmed before changing the Debias_IMU training set.

| Sequence | Status | Result Path | Note | Action |
|---|---|---|---|---|
| `0912s06_80` | usable | `results/pseudo_gt/LIASuw/0912s06_80/` | Selected first-80-second S06 stereo-only result. Final metrics are position RMSE about `0.047 m` and orientation RMSE about `3.50 deg` after fitted camera-to-GT frame rotation. Details are in `results/pseudo_gt/LIASuw/0912s06_80/training_selection.txt`. | Keep and use as accepted pseudo-GT. |
| `0912s07_30` | usable | `results/pseudo_gt/LIASuw/0912s07_30/` | Selected first-30-second S07 stereo-only local window. Metrics are position RMSE about `0.0337 m`, orientation RMSE about `1.98 deg`, pitch RMSE about `1.60 deg`, and orientation max about `5.00 deg`. Details are in `results/pseudo_gt/LIASuw/0912s07_30/window_selection_note.md`. | Keep and use as accepted local-window pseudo-GT. |
| `0912s10_40` | usable | `results/pseudo_gt/LIASuw/0912s10_40/` | Selected first-40-second S10 stereo-only local window. Metrics at `time_offset=-1.56` are position RMSE about `0.0278 m`, orientation RMSE about `2.26 deg`, pitch RMSE about `1.83 deg`, orientation p99 about `4.09 deg`, and orientation max about `9.40 deg`. Details are in `results/pseudo_gt/LIASuw/0912s10_40/window_selection_note.md`. | Keep and use as accepted local-window pseudo-GT. |
| `0203s18_80` | usable candidate | `results/pseudo_gt/LIASuw/0203s18_80/` | Newly tested LIASuw Training 0203 S18 stereo-only candidate. The bag is shorter than 80 s, so the actual image span is about `26.15 s`. Metrics at `time_offset=2.2` are position RMSE about `0.0354 m`, orientation RMSE about `1.32 deg`, pitch RMSE about `0.92 deg`, orientation max about `3.50 deg`, 524 associated poses, and no dropped-frame proxy count. | Keep as usable-quality 0203 candidate; confirm before adding to Debias_IMU training. |
| `0203s19_80` | usable candidate | `results/pseudo_gt/LIASuw/0203s19_80/` | Newly tested LIASuw Training 0203 S19 stereo-only candidate. The bag is shorter than 80 s, so the actual image span is about `23.95 s`. Metrics at `time_offset=2.2` are position RMSE about `0.0437 m`, orientation RMSE about `2.70 deg`, pitch RMSE about `1.75 deg`, orientation max about `9.52 deg`, 480 associated poses, and no dropped-frame proxy count. | Keep as usable-quality 0203 candidate; confirm before adding to Debias_IMU training. |
| `0203s22_80` | usable candidate | `results/pseudo_gt/LIASuw/0203s22_80/` | Newly tested LIASuw Training 0203 S22 stereo-only candidate. The bag is shorter than 80 s, so the actual image span is about `60.60 s`. Metrics at `time_offset=2.2` are position RMSE about `0.0329 m`, orientation RMSE about `2.97 deg`, pitch RMSE about `1.44 deg`, orientation max about `9.09 deg`, 1213 associated poses, and no dropped-frame proxy count. | Keep as usable-quality 0203 candidate; confirm before adding to Debias_IMU training. |
| `0205s14_80` | usable candidate | `results/pseudo_gt/LIASuw/0205s14_80/` | Newly tested first-80-second LIASuw Training 0205 S14 stereo-only candidate. It uses the 0205/0203 DAVIS APS baseline config `plato/configs/orbslam3/LIASuw_0205_image_baseline.yaml`, not the 0912 image config. Final metrics at `time_offset=0.34` are position RMSE about `0.0427 m`, orientation RMSE about `2.41 deg`, pitch RMSE about `1.75 deg`, orientation max about `6.14 deg`, 1562 associated poses, and 38 dropped-frame proxy count. | Keep as usable-quality 0205 candidate; confirm before adding to Debias_IMU training. |

## Testing Data, Not Training Pseudo-GT

| Sequence | Status | Result Path | Note | Action |
|---|---|---|---|---|
| `corldata S13+` | testing data | `results/pseudo_gt/corldata/S13/`, `S15/`, `S20/`, `S21/`, `S22/`, `S23/` | User decision on 2026-05-14: corldata sequences from `S13` onward have visually degraded intervals, with lights on for one interval and off for another. They are intended for validating network training results under degraded visual conditions. | Keep VO/evaluation artifacts for testing; do not add to training pseudo-GT by metric quality alone unless the user explicitly changes the data split. |

## Debias_IMU Body GT Exports

On 2026-05-08, Debias_IMU/TUM-VI-style 200 Hz IMU/body GT was generated for the current usable and usable-quality candidate windows using `plato/scripts/export_liasuw_debias_body_gt.py`.

Conversion convention:

```text
T_Wv_C       = ORB-SLAM3 stereo-only left-camera VO pose
T_Wv_I       = T_Wv_C * T_C_I
T_Wm_I_gt    = T_Wm_M * T_M_I
```

`T_C_I` comes from the appropriate camera-IMU calibration (`0912event/Calibration files/cam-imu.yaml` for 0912, `LIASuw_Training/esio.yaml` for 0203/0205). The default exported GT remains in mocap/world coordinates with the pose frame converted to IMU/body; first-frame-normalized copies are also written but require consistent gravity rotation before training.

Aggregate Debias_IMU dataset root:

`results/debias_imu/LIASuw_training_candidates_200hz/`

Each aggregate sequence contains:

- `mav0/imu0/data.csv`: original LIASuw IMU resampled to the exact 200 Hz GT timestamps.
- `mav0/mocap0/data.csv`: mocap-derived IMU/body GT in TUM-VI pose format.

Per-sequence source artifacts are stored at:

`results/pseudo_gt/LIASuw/<sequence>/debias_imu_body_gt_200hz/`

| Sequence | Status | Body-GT residual translation RMSE | Body-GT residual rotation RMSE | Output samples |
|---|---|---:|---:|---:|
| `0912s06_80` | usable | `0.0467 m` | `3.52 deg` | `15989` |
| `0912s07_30` | usable | `0.0305 m` | `1.91 deg` | `5990` |
| `0912s10_40` | usable | `0.0212 m` | `2.19 deg` | `7988` |
| `0203s18_80` | usable candidate | `0.0217 m` | `1.25 deg` | `5230` |
| `0203s19_80` | usable candidate | `0.0322 m` | `2.67 deg` | `4790` |
| `0203s22_80` | usable candidate | `0.0318 m` | `2.98 deg` | `12120` |
| `0205s14_80` | usable candidate | `0.0352 m` | `2.41 deg` | `15990` |

Checks completed: aggregate `imu0` and `mocap0` timestamps match row-by-row, median dt is about `0.004999936 s`, quaternions are unit norm to about `1e-12`, and first-frame-normalized files start at identity. A Debias_IMU `TUMDataset` smoke test was blocked because the current Python environments do not have `torch` installed.

## 0203 Candidate Evaluation

LIASuw Training 0203 candidates `s01`, `s02`, `s03`, `s05`, `s06`, `s07`, `s15`, `s17`, `s18`, `s19`, `s20`, `s21`, `s22`, `s23`, and `s24` were tested on 2026-05-08 with the 0203/0205 DAVIS APS baseline config `plato/configs/orbslam3/LIASuw_0205_image_baseline.yaml`. The initial evaluation used the dataset-order `time_offset=2.2`; promising sequences were also checked with local offset diagnostics and short-window scans. Final kept metrics use the position-focused `time_offset=2.2`.

| Sequence | Decision | Best/Final Result | Reason |
|---|---|---|---|
| `0203s01_80` | unusable, removed | Position RMSE about `0.144 m`, orientation RMSE about `10.68 deg`, and 597 dropped-frame proxy count. | Poor tracking and accuracy. |
| `0203s02_80` | unusable, removed | Position RMSE about `0.074 m`, orientation RMSE about `3.72 deg`, and 26 dropped-frame proxy count. | Close, but position is worse than accepted windows. |
| `0203s03_80` | unusable, removed | Position RMSE about `0.097 m`, orientation RMSE about `7.67 deg`, and 517 dropped-frame proxy count. | Poor tracking and orientation. |
| `0203s05_80` | unusable, removed | Position RMSE about `0.064 m`, orientation RMSE about `5.79 deg`, and 245 dropped-frame proxy count. | Orientation and tracking are not good enough. |
| `0203s06_80` | unusable, removed | Position RMSE about `0.091 m`, orientation RMSE about `6.90 deg`, and 826 dropped-frame proxy count. | Poor tracking and accuracy. |
| `0203s07_80` | unusable, removed | Position RMSE about `0.046 m`, orientation RMSE about `3.34 deg`, but orientation max was about `25.0 deg` with 65 dropped-frame proxy count. | Has a large local orientation spike; not kept as a full-window candidate. |
| `0203s15_full` | unusable, removed | Full-window fallback after first-80-second truncation failed due an anomalous early timestamp. Position RMSE about `0.151 m`, orientation RMSE about `13.61 deg`, and 549 dropped-frame proxy count. | Poor tracking and accuracy. |
| `0203s17_80` | unusable, removed | Position RMSE about `0.117 m`, orientation RMSE about `112.24 deg`. | Frame/orientation result is unusable. |
| `0203s18_80` | usable candidate, kept | Position RMSE about `0.0354 m`, orientation RMSE about `1.32 deg`, pitch RMSE about `0.92 deg`, orientation max about `3.50 deg`, and no dropped-frame proxy count. | Best 0203 candidate found in this pass. |
| `0203s19_80` | usable candidate, kept | Position RMSE about `0.0437 m`, orientation RMSE about `2.70 deg`, pitch RMSE about `1.75 deg`, orientation max about `9.52 deg`, and no dropped-frame proxy count. | Usable short candidate, but maximum errors are higher than S18. |
| `0203s20_80` | unusable, removed | Position RMSE about `0.081 m`, orientation RMSE about `10.91 deg`, and 397 dropped-frame proxy count. | Orientation is poor. |
| `0203s21_80` | unusable, removed | Position RMSE about `0.108 m`, orientation RMSE about `7.80 deg`, and 85 dropped-frame proxy count. | Accuracy is poor. |
| `0203s22_80` | usable candidate, kept | Position RMSE about `0.0329 m`, orientation RMSE about `2.97 deg`, pitch RMSE about `1.44 deg`, orientation max about `9.09 deg`, and no dropped-frame proxy count. | Longest usable-quality 0203 candidate found in this pass. |
| `0203s23_80` | unusable, removed | Position RMSE about `0.619 m`, orientation RMSE about `20.94 deg`. | Unusable trajectory accuracy. |
| `0203s24_80` | unusable, removed | Position RMSE about `0.111 m`, orientation RMSE about `8.23 deg`, and 202 dropped-frame proxy count. | Poor tracking and accuracy. |

## 0205 Candidate Evaluation

LIASuw Training 0205 candidates `s13` through `s17` were tested on 2026-05-08. The 0205 session uses a different DAVIS APS calibration from the accepted 0912 sequences; early runs with `plato/configs/orbslam3/LIASuw_image_nf2000_far8_depth30.yaml` were invalid for 0205 and were not used for decisions.

| Sequence | Decision | Best/Final Result | Reason |
|---|---|---|---|
| `0205s13_80` | unusable, removed | Correct 0205 baseline config gave position RMSE about `0.163 m`, orientation RMSE about `6.80 deg`, and 36 dropped-frame proxy count. The 2000-feature 0205 tuning variant crashed. | Accuracy is worse than accepted windows. |
| `0205s14_80` | usable candidate, kept | Correct 0205 baseline config gave position RMSE about `0.0427 m`, orientation RMSE about `2.41 deg`, pitch RMSE about `1.75 deg`, and orientation max about `6.14 deg` at `time_offset=0.34`. | Best 0205 candidate found in this pass. |
| `0205s15_80` | unusable, removed | Correct 0205 baseline config gave position RMSE about `0.152 m`, orientation RMSE about `10.79 deg`, pitch RMSE about `9.13 deg`, and 874 dropped-frame proxy count. The 2000-feature 0205 tuning variant crashed. | Tracking and orientation are poor. |
| `0205s16_80` | unusable, removed | Correct 0205 baseline config gave position RMSE about `0.083 m`, orientation RMSE about `4.95 deg`, and only 414 associated poses with 1186 dropped-frame proxy count. | Too many tracking losses and accuracy is below the accepted windows. |
| `0205s17_80` | unusable, removed | `GT/17/mc.tum` is absent; `mc1.tum` covers only about 11 s. With correct 0205 calibration, ORB-SLAM3 output began after the available GT interval, so no timestamp associations were available. | Cannot be reliably evaluated for pseudo-GT selection. |

## Unusable Results Removed

The following generated results are marked unusable for pseudo-GT and should not be used for Debias_IMU supervision/training. Their result files were removed from `results/` on 2026-05-08, except where a path was already absent.

| Removed Path | Status | Reason |
|---|---|---|
| `results/pseudo_gt/LIASuw/0912s06_80_stereo_inertial_camimu/` | comparison only, removed | Stereo-inertial S06 comparison. Position RMSE was slightly worse than the selected stereo-only S06 window; not selected as pseudo-GT. |
| `results/pseudo_gt/LIASuw/0912s07/` | unusable, removed | Stale/intermediate S07 output, superseded by the accepted `0912s07_30` local window. |
| `results/pseudo_gt/LIASuw/0912s07_80/` | unusable, removed | First-80-second S07 stereo-only result had rotation RMSE about `5.05 deg`; full S07 window is not accepted. |
| `results/pseudo_gt/LIASuw/0912s07_80_stereo_inertial_camimu/` | comparison only, removed | Stereo-inertial S07 diagnostic did not solve the pitch issue and dropped about `153` poses. |
| `results/pseudo_gt/LIASuw/0912s07_80_stereo_inertial_camimu_imu2x/` | comparison only, removed | S07 stereo-inertial IMU-weight diagnostic; not accepted. |
| `results/pseudo_gt/LIASuw/0912s07_80_stereo_inertial_camimu_imu4x/` | comparison only, removed | Best S07 stereo-inertial IMU-weight diagnostic, but still not accepted. |
| `results/pseudo_gt/LIASuw/0912s09/` | unusable, removed | Stale/intermediate S09 output; not accepted as pseudo-GT. |
| `results/pseudo_gt/LIASuw/0912s09_80/` | unusable, removed | S09 stereo-only result was worse than accepted windows. |
| `results/pseudo_gt/LIASuw/0912s10/` | unusable, removed | Stale/intermediate S10 output, superseded by the accepted `0912s10_40` local window. |
| `results/pseudo_gt/LIASuw/0912s10_80/` | unusable, removed | First-80-second S10 stereo-only result had position RMSE about `0.132 m`; full S10 window is not accepted. |
| `results/pseudo_gt/LIASuw/0912s11_80/` | unusable, removed | S11 stereo-only result was worse than accepted windows. |
| `results/pseudo_gt/LIASuw/0912s12_80/` | unusable, removed | Stale/intermediate S12 output; not accepted as pseudo-GT. |
| `results/pseudo_gt/LIASuw/0912s12_full/` | unusable, removed | S12 full-window result was worse than accepted windows. |
| `results/pseudo_gt/LIASuw/0912s13_80/` | unusable, removed | S13 stereo-only result had position RMSE about `0.0707 m` and rotation RMSE about `4.59 deg`; not accepted. |
| `results/pseudo_gt/LIASuw/0912s14_80/` | unusable, removed | S14 stereo-only result was worse than accepted windows. |
| `results/pseudo_gt/LIASuw/0203s01_80/` | unusable, removed | Re-evaluated with correct 0203/0205 calibration; tracking and accuracy remained poor. |
| `results/pseudo_gt/LIASuw/0203s02_80/` | unusable, removed | Close but position RMSE remained worse than accepted windows. |
| `results/pseudo_gt/LIASuw/0203s03_80/` | unusable, removed | Poor tracking and orientation. |
| `results/pseudo_gt/LIASuw/0203s05_80/` | unusable, removed | Orientation and tracking were not good enough. |
| `results/pseudo_gt/LIASuw/0203s06_80/` | unusable, removed | Poor tracking and accuracy. |
| `results/pseudo_gt/LIASuw/0203s07_80/` | unusable, removed | Large local orientation spike and tracking losses. |
| `results/pseudo_gt/LIASuw/0203s15_80/` | invalid/intermediate, removed | First-80-second truncation failed because an anomalous early timestamp kept only one frame. |
| `results/pseudo_gt/LIASuw/0203s15_full/` | unusable, removed | Full-window fallback was inaccurate and had many tracking losses. |
| `results/pseudo_gt/LIASuw/0203s17_80/` | unusable, removed | Orientation result was unusable. |
| `results/pseudo_gt/LIASuw/0203s20_80/` | unusable, removed | Orientation was poor. |
| `results/pseudo_gt/LIASuw/0203s21_80/` | unusable, removed | Accuracy was poor. |
| `results/pseudo_gt/LIASuw/0203s23_80/` | unusable, removed | Trajectory accuracy was unusable. |
| `results/pseudo_gt/LIASuw/0203s24_80/` | unusable, removed | Poor tracking and accuracy. |
| `results/pseudo_gt/LIASuw/0203_training_candidate_initial_summary.json` | obsolete summary, removed | Temporary batch summary; decisions are preserved here. |
| `results/pseudo_gt/LIASuw/0205s13_80/` | invalid/intermediate, removed | Initial 0205 S13 run used the 0912 image calibration and then crashed; not a valid 0205 pseudo-GT result. |
| `results/pseudo_gt/LIASuw/0205s13_80_0205calib/` | unusable, removed | Correct 0205 2000-feature tuning variant crashed. |
| `results/pseudo_gt/LIASuw/0205s13_80_baseline/` | unusable, removed | Correct 0205 baseline fallback was stable but inaccurate. |
| `results/pseudo_gt/LIASuw/0205s13_80_robust/` | invalid/intermediate, removed | Stable fallback using the 0912 robust image calibration; not valid for 0205 decisions. |
| `results/pseudo_gt/LIASuw/0205s14_80_0205calib/` | tuning comparison, removed | Correct 0205 2000-feature tuning variant was worse than the kept 0205 baseline result. |
| `results/pseudo_gt/LIASuw/0205s14_80_baseline/` | tuning comparison, removed | Reused exported images for tuning; final self-contained result was rerun and kept as `results/pseudo_gt/LIASuw/0205s14_80/`. |
| `results/pseudo_gt/LIASuw/0205s15_80/` | invalid/intermediate, removed | Initial 0205 S15 run used the 0912 image calibration; not a valid 0205 pseudo-GT result. |
| `results/pseudo_gt/LIASuw/0205s15_80_0205calib/` | unusable, removed | Correct 0205 2000-feature tuning variant crashed. |
| `results/pseudo_gt/LIASuw/0205s15_80_baseline/` | unusable, removed | Correct 0205 baseline fallback had poor tracking and orientation. |
| `results/pseudo_gt/LIASuw/0205s16_80/` | invalid/intermediate, removed | Initial 0205 S16 run used the 0912 image calibration; not a valid 0205 pseudo-GT result. |
| `results/pseudo_gt/LIASuw/0205s16_80_0205calib/` | unusable, removed | Correct 0205 2000-feature tuning variant had many tracking losses and poor orientation. |
| `results/pseudo_gt/LIASuw/0205s16_80_baseline/` | unusable, removed | Correct 0205 baseline fallback had too few associated poses and too many tracking losses. |
| `results/pseudo_gt/LIASuw/0205s17_80/` | invalid/intermediate, removed | Initial 0205 S17 run used the 0912 image calibration; not a valid 0205 pseudo-GT result. |
| `results/pseudo_gt/LIASuw/0205s17_80_0205calib/` | unusable, removed | Correct 0205 2000-feature tuning variant produced no associations with the short available `mc1.tum` GT interval. |
| `results/pseudo_gt/LIASuw/0205s17_80_baseline/` | unusable, removed | Correct 0205 baseline fallback produced no associations with the short available `mc1.tum` GT interval. |
| `results/pseudo_gt/LIASuw/0912_training_untested_summary.csv` | obsolete summary, removed | Summary of now-removed rejected full-window runs. Decisions are preserved here. |
| `results/pseudo_gt/LIASuw/0912_training_untested_summary.md` | obsolete summary, removed | Summary of now-removed rejected full-window runs. Decisions are preserved here. |
| `results/pseudo_gt/LIASuw/0912s07_stereo_inertial_imu_weight_summary.csv` | obsolete summary, removed | Summary of now-removed S07 stereo-inertial diagnostics. Decisions are preserved here. |
| `results/pseudo_gt/LIASuw/0912s07_stereo_inertial_imu_weight_summary.md` | obsolete summary, removed | Summary of now-removed S07 stereo-inertial diagnostics. Decisions are preserved here. |
| `results/0203_eval/` | unusable, removed | 0203 evaluation results are not accepted for Debias_IMU pseudo-GT. |
| `results/0203_imu_gt/` | unusable, removed | 0203 IMU-GT generated outputs are not accepted for Debias_IMU pseudo-GT. |
| `results/0203_stereo_vo/` | unusable, removed | 0203 stereo VO outputs are not accepted for Debias_IMU pseudo-GT. |
| `results/pseudo_gt/LIASuw/0912s15/` | unusable, already absent | Rejected by user decision. |

## Training Requirement

Future Debias_IMU training should sample only from user-confirmed usable windows and any usable candidates explicitly admitted into the training set. It should still apply local window quality filtering before training; candidate window lengths may include `32` and `64`, and maximum window length should not exceed `256`.
