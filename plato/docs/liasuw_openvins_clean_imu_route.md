# LIASuw OpenVINS Clean-IMU Route

Date: 2026-05-11

## Goal

Implement the Debiasing 6-DOF IMU paper's OpenVINS route on LIASuw without
modifying OpenVINS propagation. The learned bias trajectory is used offline to
produce a clean IMU stream:

```text
w_openvins = w_raw - bg_pred
a_openvins = a_raw - ba_pred
```

OpenVINS should then be run unchanged with the same stereo images and a replaced
IMU stream.

## Inputs

- Debias_IMU result:
  `results/debias_imu/LIASuw_phys_delta_vo_pseudogt_framefixed_e1800_w16pose_b64/`
- Test sequence:
  `0205s14_80`
- Raw LIASuw sequence export:
  `results/pseudo_gt/LIASuw/0205s14_80/sequence/`
- Body GT for optional evaluation/initialization:
  `results/pseudo_gt/LIASuw/0205s14_80/debias_imu_body_gt_200hz/body_gt_world.csv`

## PLATO Files Added

- `plato/scripts/openvins_eval/export_liasuw_openvins_inputs.py`
- `plato/scripts/openvins_eval/make_liasuw_openvins_bags.py`
- `plato/scripts/openvins_eval/summarize_liasuw_openvins_runs.py`
- `plato/scripts/openvins_eval/make_visual_dropout_bags.py`
- `plato/scripts/openvins_eval/run_openvins_dropout_trial.sh`
- `plato/scripts/openvins_eval/summarize_liasuw_openvins_dropout.py`
- `plato/configs/openvins/liasuw_0205/estimator_config.yaml`
- `plato/configs/openvins/liasuw_0205/kalibr_imucam_chain.yaml`
- `plato/configs/openvins/liasuw_0205/kalibr_imu_chain.yaml`

The initial clean-IMU route used unmodified OpenVINS propagation. A later
optional learned-bias propagation hook was added to external OpenVINS after
explicit user request; see the section below.

## Commands

Export raw, constant-bias-corrected, and network-debiased IMU CSVs:

```bash
/home/amao/anaconda3/envs/debiasimu/bin/python \
  plato/scripts/openvins_eval/export_liasuw_openvins_inputs.py
```

Build OpenVINS-readable ROS bags:

```bash
python3 plato/scripts/openvins_eval/make_liasuw_openvins_bags.py --compression bz2
```

## Outputs

Main output directory:

```text
results/openvins_eval/LIASuw/phys_delta_vo_framefixed_clean_imu/0205s14_80/
```

IMU CSVs:

```text
methods/raw/mav0/imu0/data.csv
methods/constant/mav0/imu0/data.csv
methods/network/mav0/imu0/data.csv
```

Bias exports:

```text
biases/network_bias_prediction_grid.csv
biases/network_bias_interpolated_to_raw_imu.csv
biases/constant_bias.json
```

ROS bags:

```text
bags/raw.bag
bags/constant.bag
bags/network.bag
```

Ground truth export:

```text
groundtruth/body_gt_openvins_asl.csv
groundtruth/body_gt_openvins_asl_image_times.csv
groundtruth/body_gt_openvins_asl_gtinit.csv
```

`body_gt_openvins_asl.csv` keeps the 200 Hz body-GT grid and is the normal
evaluation reference. `body_gt_openvins_asl_image_times.csv` is interpolated to
image timestamps for timestamp diagnostics. `body_gt_openvins_asl_gtinit.csv` is
only for OpenVINS GT-initialization diagnostics: it uses each bagged image
timestamp minus 1 microsecond so OpenVINS initializes just before the first image
callback and avoids floating-point out-of-order rejection.

## Validation

CSV validation is recorded in:

```text
results/openvins_eval/LIASuw/phys_delta_vo_framefixed_clean_imu/0205s14_80/validation.json
```

Key checks:

- `net_imu = imu_raw - predicted_bias` max absolute error: `2.384186e-07`.
- High-rate raw IMU interpolation versus saved 200 Hz IMU max absolute difference:
  `0.0118571`.
- Network-corrected high-rate IMU interpolation versus saved 200 Hz `net_imu`
  max absolute difference: `0.0118571`.
- Raw export preserves the original overlapping raw IMU rows.
- Raw and predicted-bias timestamps are strictly increasing.
- Raw, constant, and network exports share identical timestamps.

Bag validation is recorded in:

```text
results/openvins_eval/LIASuw/phys_delta_vo_framefixed_clean_imu/0205s14_80/bags/bag_validation.json
```

Each full bag contains:

- `/imu0`: `79845` messages.
- `/cam0/image_raw`: `1599` messages.
- `/cam1/image_raw`: `1599` messages.

The original image export had `1600` stereo pairs. The bag builder drops one
last stereo pair because `t_cam + timeshift_cam_imu` falls outside the corrected
IMU span. The configured `timeshift_cam_imu` is `0.00777437 s`, from the LIASuw
0203/0205 `esio.yaml`.

## OpenVINS Config

Use:

```text
plato/configs/openvins/liasuw_0205/estimator_config.yaml
```

The config fixes camera intrinsics, camera-IMU extrinsics, and time offset so the
raw/constant/network comparison changes only the IMU stream. It uses:

- IMU topic: `/imu0`
- Left image topic: `/cam0/image_raw`
- Right image topic: `/cam1/image_raw`
- Gravity magnitude: `9.7803`
- Camera-to-IMU timeshift: `0.00777437 s`

## Runtime Note

After rebuilding OpenVINS in the current PLATO path, `rospack find ov_msckf`
resolves to:

```text
/home/amao/workspace/PLATO/external/OpenVINS/src/open_vins/ov_msckf
```

`ldd external/OpenVINS/devel/lib/ov_msckf/ros1_serial_msckf` now links OpenVINS
libraries from `external/OpenVINS/devel/.private/...`, so the moved-workspace
linkage issue is repaired.

Self-initialization on the first 15 seconds of `0205s14_80` is not yet usable
with the current LIASuw config. Static init reports insufficient IMU excitation,
and dynamic init enters but fails because the 2 second window has either too few
valid features for the dynamic initializer or only about `0.6-1.3 deg` gyroscope
rotation versus the tested `init_dyn_min_deg=5.0`.

A 10 second GT-initialized diagnostic with `network.bag` and
`body_gt_openvins_asl_gtinit.csv` successfully wrote an OpenVINS estimate:

```text
results/openvins_eval/LIASuw/phys_delta_vo_framefixed_clean_imu/0205s14_80/
  runs/gtinit_minus1us_smoke_network/estimate.txt
```

The estimate file has 196 lines including the header and reaches timestamp
`1770303704.19843`, consistent with the 10 second bag slice. The node still
throws `class_loader::LibraryUnloadException` during ROS/plugin shutdown after
processing; this is separate from the data/config validation.

## Full GT-init Comparison

Full-bag GT-initialized runs were generated under:

```text
results/openvins_eval/LIASuw/phys_delta_vo_framefixed_clean_imu/0205s14_80/runs/
```

Summary files:

```text
runs/gtinit_full_summary.md
runs/gtinit_full_summary.json
```

Direct comparison against the 200 Hz IMU/body GT:

| method | duration_s | position RMSE m | rotation RMSE deg | velocity RMSE m/s | note |
| --- | ---: | ---: | ---: | ---: | --- |
| raw | 28.300 | 1215.3060 | 16.657 | 110.6548 | zero-bias GT-init; diverged early |
| raw_constbias | 79.650 | 0.0981 | 3.613 | 0.0379 | raw IMU, constant-bias initial state |
| constant | 79.650 | 0.0981 | 3.613 | 0.0379 | constant-bias-corrected IMU |
| network | 79.650 | 0.1021 | 4.187 | 0.0338 | network-debiased IMU |

Interpretation: raw with zero initial bias is not a fair baseline and diverges.
`raw_constbias` matches `constant`, as expected, which validates that OpenVINS'
state bias and offline constant correction are being applied consistently. In
this GT-init OpenVINS test, the network-debiased IMU slightly improves velocity
RMSE but is worse than constant bias on position and orientation.

## Optional Learned Bias Dynamics Hook

OpenVINS now has an optional learned nominal bias propagation path. The default
is off:

```yaml
use_learned_bias_dynamics: false
learned_bias_dynamics_model: "phys_delta"
learned_bias_dynamics_weights: "/home/amao/workspace/PLATO/results/debias_imu/LIASuw_phys_delta_e1800_w16_b64/openvins/phys_delta_openvins.yaml"
```

When disabled, OpenVINS keeps its original behavior exactly at the model level:
`b_g` and `b_a` are constant in the nominal propagation, and the original
random-walk process noise `sigma_wb` / `sigma_ab` is still injected.

When enabled, only the nominal bias dynamics are changed. Two model paths are
available:

- `affine`: a lightweight debug model,

```text
dbg_dt = c_g + Wbg * b_g + Wwg * w_m + Wag * a_m
dba_dt = c_a + Wba * b_a + Wwa * w_m + Waa * a_m
```

- `phys_delta`: the exported PLATO/Debias_IMU network. The current exported
  LIASuw model is:

```text
results/debias_imu/LIASuw_phys_delta_e1800_w16_b64/openvins/phys_delta_openvins.yaml
```

The `phys_delta` C++ path mirrors the trained feature construction: current
bias, corrected angular velocity/specific force, body-frame gravity, inertial
invariants, and discrete raw-IMU change `u_k-u_{k-1}` feed the exported
linear+GELU MLP. The model is integrated with first-order Euler over each IMU
interval. The filter covariance keeps the original additive bias process noise
as residual model uncertainty. The affine path uses the analytic self-transition
below; the `phys_delta` path uses a numerical derivative of `[f_g; f_a]` with
respect to `[b_g; b_a]`.

```text
F_bg_bg = I + dt * Wbg
F_ba_ba = I + dt * Wba
```

Measurement update code is unchanged.

External OpenVINS source files changed:

```text
external/OpenVINS/src/open_vins/ov_msckf/src/state/StateOptions.h
external/OpenVINS/src/open_vins/ov_msckf/src/state/Propagator.h
external/OpenVINS/src/open_vins/ov_msckf/src/state/Propagator.cpp
```

The PLATO-owned exporter is:

```text
plato/scripts/openvins_eval/export_phys_delta_to_openvins_yaml.py
```

Build/test status:

- `catkin build ov_msckf` succeeded for `ov_core`, `ov_init`, and `ov_msckf`.
- Original-mode smoke test with `use_learned_bias_dynamics=false` wrote
  `/tmp/liasuw_openvins_learned_off_estimate.txt` and
  `/tmp/liasuw_openvins_learned_off_std.txt`, each with `196` lines and final
  timestamp `1770303704.19843`.
- Opt-in `phys_delta` smoke test with `_use_learned_bias_dynamics:=true` wrote
  `/tmp/liasuw_openvins_phys_delta_estimate.txt` and
  `/tmp/liasuw_openvins_phys_delta_std.txt`, each with `196` lines and final
  timestamp `1770303704.19843`.
- The known `class_loader::LibraryUnloadException` still appears during ROS
  shutdown after processing.

## Visual Dropout Comparison

To expose IMU propagation behavior more clearly, a second set of bags removes
both stereo image topics from `5 s` to `75 s` while keeping the full IMU stream:

```text
bags_dropout_5_75/constant.bag
bags_dropout_5_75/network.bag
```

The generation script is:

```bash
python3 plato/scripts/openvins_eval/make_visual_dropout_bags.py \
  --methods constant network --dropout-start-s 5 --dropout-end-s 75
```

Because OpenVINS' `estimate.txt` is saved on visual updates, the dropout
experiment records `/ros1_serial_msckf/odomimu` with `ov_eval/pose_to_file`.
The summary is:

```text
runs/dropout_5_75_summary.md
runs/dropout_5_75_summary.json
```

Key result:

| method | dropout pos RMSE m | dropout final pos m | dropout rot RMSE deg | dropout final rot deg | overall final pos m | overall final rot deg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| constant | 1302.991 | 3345.384 | 21.162 | 35.989 | 105910.695 | 111.092 |
| network | 1101.948 | 2869.502 | 8.196 | 15.511 | 5763.405 | 18.162 |

In this artificial visual-failure test, network-debiased IMU produces a much
smaller orientation drift and a much smaller post-dropout failure than constant
bias. Position drift during the 70 second pure-IMU span is still large for both
methods, so this is evidence of improved propagation robustness, not yet a
complete solution.

## Shorter Visual Dropout Windows

Shorter visual-failure windows were added to make the OpenVINS result easier to
interpret without letting pure-IMU position drift dominate the whole sequence.
The tested bags are:

```text
bags_dropout_5_25/constant.bag
bags_dropout_5_25/network.bag
bags_dropout_5_35/constant.bag
bags_dropout_5_35/network.bag
```

They were generated with:

```bash
python3 plato/scripts/openvins_eval/make_visual_dropout_bags.py \
  --methods constant network --dropout-start-s 5 --dropout-end-s 25 \
  --compression bz2

python3 plato/scripts/openvins_eval/make_visual_dropout_bags.py \
  --methods constant network --dropout-start-s 5 --dropout-end-s 35 \
  --compression bz2
```

OpenVINS was run with the same GT initialization and unmodified propagation
config. The helper below starts `ov_eval/pose_to_file`, runs serial OpenVINS,
and records `/ros1_serial_msckf/odomimu`:

```bash
plato/scripts/openvins_eval/run_openvins_dropout_trial.sh \
  --method constant --dropout-start 5 --dropout-end 25
plato/scripts/openvins_eval/run_openvins_dropout_trial.sh \
  --method network --dropout-start 5 --dropout-end 25
plato/scripts/openvins_eval/run_openvins_dropout_trial.sh \
  --method constant --dropout-start 5 --dropout-end 35
plato/scripts/openvins_eval/run_openvins_dropout_trial.sh \
  --method network --dropout-start 5 --dropout-end 35
```

The known OpenVINS `class_loader::LibraryUnloadException` still appears at
shutdown after outputs are written; the helper records console logs under each
run directory and treats status `134` as usable only when `odomimu.txt` exists.

Summary files:

```text
runs/dropout_5_25_summary.md
runs/dropout_5_25_summary.json
runs/dropout_5_35_summary.md
runs/dropout_5_35_summary.json
```

Key shorter-window results:

| dropout | method | dropout pos RMSE m | dropout final pos m | dropout rot RMSE deg | dropout final rot deg | overall final pos m | overall final rot deg |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5-25 s | constant | 41.277 | 103.658 | 7.548 | 10.333 | 17497.311 | 131.544 |
| 5-25 s | network | 34.027 | 85.077 | 2.978 | 4.179 | 3883.551 | 22.624 |
| 5-35 s | constant | 123.281 | 313.440 | 9.945 | 15.843 | 13214.952 | 166.345 |
| 5-35 s | network | 100.924 | 256.636 | 3.733 | 6.037 | 3495.267 | 17.101 |

Interpretation: with static/offline network bias correction, the learned bias
trajectory is consistently better than the constant-bias correction during
visual failure. The strongest gain is orientation: about `60%` lower rotation
RMSE for both shorter dropout windows. Position is also improved by about
`18%`, but remains too large for long visual outages.

## Frozen OpenVINS Bias Mode

OpenVINS normally keeps `b_g` and `b_a` as online EKF states. To test whether
the externally corrected IMU is being used without a second online bias
estimator absorbing the residuals, an opt-in frozen-bias mode was added in
external OpenVINS:

```yaml
freeze_imu_bias: true
frozen_imu_bias_g: [0.0, 0.0, 0.0]
frozen_imu_bias_a: [0.0, 0.0, 0.0]
frozen_imu_bias_cov: 1e-12
```

The default config remains reversible and original-mode:

```text
plato/configs/openvins/liasuw_0205/estimator_config.yaml
```

The frozen clean-IMU config is:

```text
plato/configs/openvins/liasuw_0205/estimator_config_freeze_bias.yaml
```

Implementation notes:

- `StateOptions` parses the frozen-bias options.
- `StateHelper` forces the configured `b_g/b_a`, zeros their covariance rows and
  columns, and zeros the bias part of EKF `dx`.
- `Propagator` suppresses bias random-walk process noise when frozen and lets
  the fixed bias values, usually zero for corrected IMU, be the only OpenVINS
  bias correction.
- `freeze_imu_bias=false` preserves the original OpenVINS path.

Verification on `bags_dropout_5_25/network.bag`:

- `catkin build ov_msckf` succeeded after the change.
- Default mode still online-updated bias, ending around
  `bg=[-0.0010,0.0043,-0.0033]` and
  `ba=[0.0398,0.0005,-0.0406]`.
- Frozen mode kept every logged bias line at `bg=0`, `ba=0`.
- The frozen smoke run processed 669 logged updates before the same known
  OpenVINS shutdown `class_loader::LibraryUnloadException`; no separate
  numerical error was reported in the log.

### Frozen-Bias Dropout Results

`run_openvins_dropout_trial.sh` and `summarize_liasuw_openvins_dropout.py` now
accept `--run-suffix`, so frozen-bias runs can be stored separately from the
older default-mode runs. The frozen runs below used:

```text
plato/configs/openvins/liasuw_0205/estimator_config_freeze_bias.yaml
```

Output summaries:

```text
runs/dropout_5_25_frozen_summary.md
runs/dropout_5_35_frozen_summary.md
runs/dropout_5_75_frozen_summary.md
```

| dropout | method | dropout pos RMSE m | dropout final pos m | dropout rot RMSE deg | dropout final rot deg | overall final pos m | overall final rot deg |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5-25 s | constant frozen | 51.681 | 130.228 | 6.873 | 9.565 | 5424.523 | 30.330 |
| 5-25 s | network frozen | 12.502 | 27.466 | 2.724 | 3.168 | 150.301 | 7.262 |
| 5-35 s | constant frozen | 156.429 | 400.398 | 8.744 | 13.381 | 5919.917 | 25.880 |
| 5-35 s | network frozen | 27.136 | 59.562 | 2.941 | 3.825 | 5445.206 | 24.507 |
| 5-75 s | constant frozen | 1728.868 | 4478.844 | 18.278 | 31.995 | 4480.310 | 31.989 |
| 5-75 s | network frozen | 137.992 | 305.222 | 6.043 | 10.906 | 7883.608 | 171.661 |

Interpretation: frozen-bias mode makes the externally corrected IMU matter
more directly. Across all dropout windows, network-corrected IMU is far better
than constant-bias correction inside the visual dropout interval. Relative to
constant frozen, network frozen reduces dropout position RMSE by about `76%`,
`83%`, and `92%` for 5-25 s, 5-35 s, and 5-75 s respectively, and reduces
dropout rotation RMSE by about `60-67%`. Long post-dropout behavior can still
diverge badly once the filter has accumulated a large inertial error, so the
dropout-window metric is the cleaner comparison for IMU propagation.
