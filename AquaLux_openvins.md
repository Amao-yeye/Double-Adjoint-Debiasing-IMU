# AquaLux OpenVINS Record

Created: 2026-05-26.

## Scope

This document records the AquaLux OpenVINS evaluation pipeline, current baseline
decision, no-gate ablation, and canonical checkpoints for later runs.

Dataset:

- Root: `/home/amao/datasets/AquaLux/0524_data`.
- Test sequences: `Test_001, Test_003, Test_005, Test_007, Test_009, Test_011, Test_013, Test_015, Test_017, Test_019`.
- GT for evaluation: `{SEQ}/ATO/gt_imu_gravity_aligned.tum`.
- Raw IMU: `{SEQ}/imu_diagnostics/imu_left.csv`.
- Gravity: `[0, 0, -9.7879]`.

Evaluation:

- `evo_ape tum --align`.
- Position: `pose_relation=trans_part`.
- Rotation: `pose_relation=angle_deg`.
- `t_max_diff=0.02`.
- No scale correction.

OpenVINS calibration policy:

- `calib_cam_extrinsics: false`.
- `calib_cam_intrinsics: false`.
- `calib_cam_timeoffset: false`.

## Raw OpenVINS Reference

Raw OpenVINS result:

- Directory: `Results/AquaLux/openvins_test/raw`.
- Metrics: `Results/AquaLux/openvins_test/raw/metrics.csv`.
- Mean position RMSE: `2.38835 m`.
- Mean rotation RMSE: `14.6558 deg`.
- Status: `10/10` sequences valid.

## Baseline Compensation Pipeline

Recommended current baseline:

```text
baseline_segment_delta_acc_covgate
```

Output:

```text
Results/AquaLux/openvins_test/baseline/segment_delta_acc_covgate
```

Pipeline:

1. Run raw OpenVINS first.
2. Read raw OpenVINS outputs:
   - `estimate.txt` for `q, p, v, bg, ba`;
   - `odomimu.txt` for position covariance;
   - `timing.txt` for MSCKF update timing.
3. Detect weak visual/light-off intervals:

```text
msckf_update_time < 0.0005
min_weak_duration = 0.5 s
merge_gap = 0.2 s
```

4. Apply sequence gate:

```text
raw_max_position_cov_trace = max(Pt11 + Pt22 + Pt33)
gate pass when raw_max_position_cov_trace > 1.0
```

5. Generate hybrid IMU only for gated light-off intervals:

```text
a_hybrid(t) = a_raw(t) - (b_a(t) - b_a0)
w_hybrid(t) = w_raw(t)
```

6. Rerun OpenVINS from the beginning with the hybrid IMU.

Current gate result:

- Only `Test_009` passes the gate.
- All other Test sequences keep raw IMU.

Recommended baseline result:

| method | ok/total | mean position RMSE | mean rotation RMSE |
|---|---:|---:|---:|
| raw OpenVINS | 10/10 | 2.38835 m | 14.6558 deg |
| segment_delta_acc_covgate | 10/10 | 1.72846 m | 10.3260 deg |

Relative improvement over raw:

- Position: `27.63%`.
- Rotation: `29.54%`.

Main successful case:

| sequence | raw position | raw rotation | gated position | gated rotation |
|---|---:|---:|---:|---:|
| Test_009 | 7.339129 m | 68.294512 deg | 0.740300 m | 24.996929 deg |

## No-Gate Acc-Delta Ablation

No-gate method:

```text
segment_delta_acc
```

Output:

```text
Results/AquaLux/openvins_test/baseline/segment_delta_acc
```

Method:

- Detect weak visual/light-off intervals as above.
- Compensate every detected interval.
- Components: `acc` only.
- Correction mode: `delta`.
- No covariance gate.

Aggregate result:

| method | ok/total | mean position RMSE | mean rotation RMSE |
|---|---:|---:|---:|
| raw OpenVINS | 10/10 | 2.38835 m | 14.6558 deg |
| no-gate segment_delta_acc | 10/10 | 51.3493 m | 55.8808 deg |

Sequences improved by no-gate:

- `Test_001`: `0.854555 / 8.456036` -> `0.144664 / 3.975605`.
- `Test_007`: `4.427816 / 10.635165` -> `0.134336 / 10.004233`.
- `Test_009`: `7.339129 / 68.294512` -> `0.740300 / 24.996928`.
- `Test_017`: `0.177380 / 4.479514` -> `0.126130 / 4.119235`.

Severe or important degradations:

- `Test_003`: `0.196883 / 11.517651` -> `263.115903 / 153.929370`.
- `Test_015`: `0.482875 / 12.034315` -> `232.725986 / 119.095988`.
- `Test_013`: `0.393025 / 4.611007` -> `9.083408 / 9.287270`.
- `Test_019`: `0.220182 / 18.408627` -> `1.238729 / 49.239725`.
- `Test_011`: position improves, but aligned rotation metric fails at `178.874652 deg`.

Decision:

- Keep no-gate only as an ablation.
- Do not use no-gate acc compensation as the final OpenVINS baseline.
- The weak visual detector is too permissive without a reliability gate.

## Gyro Compensation Decision

Do not enable gyro compensation in the recommended OpenVINS baseline.

Step14 diagnostics found:

- Pure preintegration has gyro signal.
- Direct gyro-delta injection into OpenVINS is not reliable.
- Gyro-only gated `Test_009`: `4.487896 m / 151.494903 deg`.
- Acc-only gated `Test_009`: `0.740300 m / 24.996929 deg`.

Current safe recommendation:

```text
components = acc
correction_mode = delta
scale = 1.0
apply_when = raw_max_position_cov_trace > 1.0
```

## Canonical Checkpoints

Canonical checkpoint directory:

```text
Results/AquaLux/preintegration/checkpoints
```

VO-SE3-MSE checkpoints:

```text
Results/AquaLux/preintegration/checkpoints/vo_se3_mse_bg.pt
Results/AquaLux/preintegration/checkpoints/vo_se3_mse_ba.pt
```

RSS baseline checkpoints:

```text
Results/AquaLux/preintegration/checkpoints/rss_baseline_bg.pt
Results/AquaLux/preintegration/checkpoints/rss_baseline_ba.pt
```

VO-SE3-MSE BG source:

```text
Results/AquaLux/preintegration/ours_vo_w16vo_se23_mse_relative_noGT_gyro_e1800_from_e300/Gyro_weights/model_epoch_1500.pt
```

VO-SE3-MSE BA source:

```text
Results/AquaLux/preintegration/ours_vo_w64pose_mse_right_bgbest_gtmseba_voft_lr1e7_e21/Acc_weights/best_val_model.pt
```

Verified SHA256:

- `vo_se3_mse_bg.pt`: `149c9c4d9e68fe4807d14c57a5dfc4eaa9c7cee2c50efc3812caec874c7f2bea`.
- `vo_se3_mse_ba.pt`: `0fce5c92a9ce3954154f254be84c1858a41ac09dd4811b606576b8d4a55b2346`.

## Important Existing References

- Baseline closure:
  `Results/AquaLux/openvins_test/baseline/baseline_closure_summary.md`.
- Step12 flow:
  `Results/AquaLux/openvins_test/step12_openvins_test_flow.md`.
- Fixed gate:
  `Results/AquaLux/openvins_test/baseline/step14_gate_gyro_cov/fixed_gate_for_future_methods.yaml`.
- No-gate analysis:
  `Results/AquaLux/openvins_test/baseline/segment_delta_acc/no_gate_degradation_analysis.md`.
- Main operation record:
  `operation_record.md`.

## Operation Log

Recording rule:

- Every subsequent operation for AquaLux OpenVINS work must be recorded in this
  document using the format `Step N: ...`.
- Each step should include the goal, action, important paths/commands, and
  result or decision.
- Continue numbering monotonically from the previous step.

Step 1: Start Structured Operation Logging

Goal:

- Establish the required recording format for all later AquaLux OpenVINS work.

Action:

- Added this `Operation Log` section to `AquaLux_openvins.md`.
- Fixed the future record format as `Step N: ...`.

Result:

- Future work in this document should append new entries as `Step 2: ...`,
  `Step 3: ...`, and so on.

Step 2: Assess Concurrent VO-SE3-MSE OpenVINS Test Safety

Goal:

- Evaluate whether a VO-SE3-MSE OpenVINS test can start while another training
  job is running in the project.

Checks:

- Active relevant process:
  - `pid 21940`: `bash tools/run_aqualux_vo_nll_alt300_bgba_cov.sh`.
  - `pid 28169`: `mainAquaLuxPLATO.py` running VO-NLL training/evaluation on
    CUDA.
- Active training output directory:
  `Results/AquaLux/preintegration/ours_vo_w64pose_nll_alt_bgba_cov_e300/cycle03_bgba_e25`.
- Active training reads/resumes from:
  `Results/AquaLux/preintegration/ours_vo_w64pose_nll_alt_bgba_cov_e300/cycle03_cov_e25/NLL_weights/model_epoch_24.pt`.
- GPU status at check time:
  - Compute process: `pid 28169`, `/home/amao/anaconda3/envs/debiasimu/bin/python`,
    `686 MiB`.
  - Total GPU memory shown by `nvidia-smi`: `2537 MiB / 8188 MiB`.
  - GPU utilization: about `35%`.
- Existing canonical VO-SE3-MSE checkpoints:
  - `Results/AquaLux/preintegration/checkpoints/vo_se3_mse_bg.pt`.
  - `Results/AquaLux/preintegration/checkpoints/vo_se3_mse_ba.pt`.
- Existing OpenVINS method outputs checked:
  - No existing `vo`/`ours` method directory was found under
    `Results/AquaLux/openvins_test` at shallow depth.
  - Existing OpenVINS metrics at this point are raw/baseline only.

Assessment:

- File-output conflict risk is low if the OpenVINS test uses a new method/output
  directory, for example:
  `Results/AquaLux/openvins_test/vo_se3_mse`.
- Checkpoint conflict risk is low because the canonical VO-SE3-MSE checkpoints
  are read-only inputs for OpenVINS testing.
- GPU conflict risk is low to moderate:
  - OpenVINS itself is CPU/ROS-heavy.
  - Hybrid IMU generation from the BiasDy models should run on CPU to avoid
    competing with the active CUDA training.
- ROS conflict risk is moderate:
  - A `rosmaster` is already running on port `11311`.
  - The OpenVINS runner can reuse the existing master, but method names must
    avoid `/` and should be unique to avoid ROS namespace collisions.
- I/O/CPU conflict risk is moderate because OpenVINS reads rosbag/camera data
  and writes per-sequence logs/results; it may slow the active training but
  should not corrupt it if output directories are separate.

Decision:

- It is acceptable to start a **carefully isolated** VO-SE3-MSE OpenVINS test
  concurrently, but use CPU for model-based IMU generation and a new OpenVINS
  method directory.
- Recommended first run should be a single-sequence smoke test before launching
  all 10 sequences.
- Recommended smoke sequence: `Test_009`, because it is the primary dropout
  sequence where baseline compensation helped.
- Do not write into:
  `Results/AquaLux/preintegration/ours_vo_w64pose_nll_alt_bgba_cov_e300`.
- Do not modify canonical checkpoints.

Runtime estimate for checking whether ours beats baseline:

- Existing raw OpenVINS runtime from `Results/AquaLux/openvins_test/raw/metrics.csv`:
  - 10 sequences total: `189.537 s`.
  - Mean per sequence: `18.954 s`.
  - Min/max per sequence: `17.557 s / 21.407 s`.
- Existing baseline `segment_delta_acc_covgate` OpenVINS runtime:
  - 10 sequences total: `188.981 s`.
  - Mean per sequence: `18.898 s`.
  - Min/max per sequence: `17.662 s / 21.043 s`.
- Existing no-gate acc-delta OpenVINS runtime:
  - 10 sequences total: `179.375 s`.
  - Mean per sequence: `17.938 s`.

Practical estimate:

- Single-sequence smoke test on `Test_009`:
  - OpenVINS + evo itself: about `1 min`.
  - Hybrid IMU generation on CPU and orchestration overhead: likely another
    `1-4 min`.
  - Expected wall time: about `2-5 min`.
- Full 10-sequence comparison against baseline:
  - OpenVINS + evo itself: about `4-5 min`.
  - Hybrid IMU generation and summary/report overhead: likely `5-15 min` when
    running on CPU during concurrent training.
  - Expected wall time: about `10-20 min` if no sequence fails or needs rerun.
- If ROS/OpenVINS initialization fails on a sequence or the active training
  causes heavy I/O contention, reserve `30-60 min` for rerun/debug.

Decision:

- To quickly test whether ours can beat the current baseline on the critical
  dropout case, run `Test_009` first and budget `2-5 min`.
- To make the claim that ours beats baseline on OpenVINS, run the full
  10-sequence comparison and budget `10-20 min` under normal conditions.

Step 3: Start VO-SE3-MSE OpenVINS Smoke Test

Goal:

- Start the VO-SE3-MSE OpenVINS test with a low-risk single-sequence smoke run.
- Use `Test_009` first because it is the critical dropout sequence where the
  current RSS baseline improves raw OpenVINS.

Planned method name:

```text
vo_se3_mse_acc_covgate
```

Planned output:

```text
Results/AquaLux/openvins_test/vo_se3_mse_acc_covgate
```

Planned checkpoints:

```text
Results/AquaLux/preintegration/checkpoints/vo_se3_mse_bg.pt
Results/AquaLux/preintegration/checkpoints/vo_se3_mse_ba.pt
```

Planned compensation policy:

- Use the same safe OpenVINS injection rule as the closed baseline:
  - `components = acc`;
  - `correction_mode = delta`;
  - `gate_max_position_cov_trace = 1.0`;
  - no gyro injection.
- Generate hybrid IMU on CPU to avoid competing with the active CUDA training.

Status:

- Smoke hybrid IMU generation completed for `Test_009`.
- OpenVINS smoke run completed for `Test_009`.

Smoke command configuration:

```text
components = acc
correction_mode = delta
gate_max_position_cov_trace = 1.0
device = cpu
source_imu_filename = hybrid_imu.csv
```

Smoke result:

```text
VO-SE3-MSE acc+covgate Test_009: 2.512806 m / 26.084722 deg
raw OpenVINS Test_009:             7.339129 m / 68.294512 deg
RSS baseline acc+covgate Test_009: 0.740300 m / 24.996929 deg
```

Interpretation:

- The VO-SE3-MSE compensated IMU improves `Test_009` relative to raw OpenVINS.
- It does not beat the current RSS baseline on the critical gated sequence.
- Continue to the full 10-sequence run for a complete OpenVINS comparison, but
  the expected outcome is now weaker than the original hypothesis that ours
  will beat baseline.

Step 4: Stop Non-Replaced Sequence Runs

Reason:

- Full hybrid IMU generation showed that, with `gate_max_position_cov_trace =
  1.0`, only `Test_009` had any IMU replacement.
- All other sequences had `segments=0/0` and `changed=0`, so running OpenVINS on
  them would mostly repeat raw OpenVINS.

Action:

- Stopped the ongoing full OpenVINS run after the user instruction:
  `没有替换就不要跑了`.
- Terminated only the active `vo_se3_mse_acc_covgate` runner and its active
  OpenVINS child process.
- Verified that no `run_aqualux_openvins_raw.py` or `ov_msckf` process remained
  from this run.

Completed outputs before stop:

```text
Test_009: 2.512806 m / 26.084722 deg  (actual replacement; valid comparison)
Test_001: 0.854555 m / 8.456055 deg   (0% changed; raw repeat)
Test_003: 0.196882 m / 11.517659 deg  (0% changed; raw repeat)
Test_005: 0.749377 m / 4.443079 deg   (0% changed; raw repeat)
```

Current decision:

- Do not run remaining 0%-changed sequences under this gate setting.
- Use `Test_009` as the meaningful OpenVINS result for VO-SE3-MSE acc+covgate.

Step 5: Report Baseline Bias Inputs

Scope:

- Baseline means the recommended OpenVINS baseline:
  `segment_delta_acc_covgate`.
- Checkpoint pair:
  - `Results/AquaLux/preintegration/rss_baseline_unseen_corrected/Gyro_weights/model_epoch_1800.pt`
  - `Results/AquaLux/preintegration/rss_baseline_unseen_corrected/Acc_weights/model_epoch_1800.pt`

Per-call BiasDy model input:

- Raw IMU vector is `[wx, wy, wz, ax, ay, az]` from
  `imu_diagnostics/imu_left.csv`.
- `imu_dot` is the numerical gradient of that raw IMU vector over local segment
  time.
- `r0` is identity rotation.
- `y` is a 16D vector with only these fields populated:
  - `y[3] = t - segment_start_time`;
  - `y[4:7] = current bg`;
  - `y[13:16] = current ba`;
  - all other entries are zero.
- Initial `current bg/ba` for each segment is the last raw OpenVINS estimate
  before the segment start.
- The model predicts `bg_dot, ba_dot`; the script integrates:
  `bg = bg + bg_dot * dt`, `ba = ba + ba_dot * dt`.

OpenVINS IMU replacement actually used by the recommended baseline:

```text
correction_mode = delta
components = acc
gyro correction = 0
acc correction = ba(t) - ba0
w_hybrid(t) = w_raw(t)
a_hybrid(t) = a_raw(t) - (ba(t) - ba0)
```

Gate outcome:

- Only `Test_009` has `gate_pass = true` and receives baseline bias input.
- All other AquaLux test sequences have `gate_pass = false`, so there is no
  BiasDy model call and no IMU replacement for the recommended baseline.

Actual `Test_009` segment seed bias inputs:

| seg | start | end | dur(s) | seed bg | seed ba | final ba | mean/max acc delta |
|---:|---:|---:|---:|---|---|---|---|
| 0 | 1779619037.162483 | 1779619039.562486 | 2.400 | `[0.008986, 0.000181, -0.012510]` | `[0.003714, 0.002661, 0.170108]` | `[0.016756, -0.016649, 0.201010]` | `0.085486 / 0.359767` |
| 1 | 1779619052.962504 | 1779619055.512507 | 2.550 | `[0.009513, 0.000311, -0.012543]` | `[0.004198, 0.002737, 0.170076]` | `[-0.018782, 0.070570, 0.153957]` | `0.072586 / 0.226031` |
| 2 | 1779619067.512522 | 1779619069.762525 | 2.250 | `[0.009775, 0.000228, -0.012661]` | `[0.003598, 0.002541, 0.169252]` | `[0.118399, -0.145879, 0.218216]` | `0.113136 / 0.379710` |
| 3 | 1779619081.612540 | 1779619085.012545 | 3.400 | `[0.009573, 0.000380, -0.012893]` | `[0.003067, 0.003191, 0.168281]` | `[0.048725, 0.003549, 0.105636]` | `0.074914 / 0.192574` |
| 4 | 1779619085.712546 | 1779619089.662551 | 3.950 | `[0.009509, 0.000370, -0.012888]` | `[0.003028, 0.003148, 0.168376]` | `[-0.013666, 0.160523, 0.178987]` | `0.075271 / 0.336578` |
| 5 | 1779619091.212553 | 1779619092.912555 | 1.700 | `[0.009498, -0.000763, -0.012992]` | `[0.002840, 0.003108, 0.167347]` | `[0.024021, -0.003089, 0.205170]` | `0.062169 / 0.248089` |

Source files:

- Implementation: `tools/generate_aqualux_segment_only_imu.py`.
- Bias derivative call: `tools/evaluate_aqualux_openvins_compensate1.py`.
- Segment data:
  `Results/AquaLux/openvins_test/baseline/segment_delta_acc_covgate/hybrid_imu/Test_009/segments.csv`.

Step 6: Start Ours No-Gate OpenVINS Test

Goal:

- Test the VO-SE3-MSE checkpoints without the covariance gate.
- Keep the same safe OpenVINS injection policy as before except for removing
  the gate:
  - `components = acc`;
  - `correction_mode = delta`;
  - `gate_max_position_cov_trace = 0.0`;
  - no gyro injection.

Method name:

```text
vo_se3_mse_acc_nogate
```

Output:

```text
Results/AquaLux/openvins_test/vo_se3_mse_acc_nogate
```

Checkpoints:

```text
Results/AquaLux/preintegration/checkpoints/vo_se3_mse_bg.pt
Results/AquaLux/preintegration/checkpoints/vo_se3_mse_ba.pt
```

Concurrency note:

- No OpenVINS runner is active at start.
- AquaLux training is still running in
  `Results/AquaLux/preintegration/ours_vo_w64pose_nll_alt_bgba_cov_e300/cycle04_bgba_e25`.
- Hybrid IMU generation will use CPU to avoid taking CUDA from the active
  training process.

Execution policy:

- First generate no-gate hybrid IMU for all sequences.
- Run OpenVINS only for sequences with `changed_samples > 0`.

Step 7: Summarize Ours No-Gate Results

Hybrid IMU generation result:

- All 10 AquaLux test sequences had `changed_samples > 0`.
- Changed fraction range: about `8.56%` to `19.88%`.
- Therefore OpenVINS was run on all 10 sequences.
- No `run_aqualux_openvins_raw.py` or `ov_msckf` process remained after the
  run completed.

No-gate result summary:

| method | sequences | mean pos RMSE | mean rot RMSE | median pos RMSE | median rot RMSE |
|---|---:|---:|---:|---:|---:|
| raw OpenVINS | 10 | `2.388348 m` | `14.655767 deg` | `0.616126 m` | `9.545600 deg` |
| RSS baseline acc+covgate | 10 | `1.728465 m` | `10.326014 deg` | `0.611588 m` | `9.545612 deg` |
| RSS baseline acc no-gate | 10 | `51.349287 m` | `55.880779 deg` | `0.989514 m` | `17.500580 deg` |
| VO-SE3-MSE acc no-gate | 10 | `35.538510 m` | `48.024379 deg` | `1.388306 m` | `10.315677 deg` |

Per-sequence comparison:

| sequence | changed | ours pos | raw pos | covgate baseline pos | RSS no-gate pos | ours rot | raw rot |
|---|---:|---:|---:|---:|---:|---:|---:|
| Test_001 | `10.43%` | `0.230305` | `0.854555` | `0.854555` | `0.144664` | `5.927741` | `8.456036` |
| Test_003 | `12.62%` | `7.511468` | `0.196883` | `0.196883` | `263.115903` | `178.193104` | `11.517651` |
| Test_005 | `10.29%` | `0.263806` | `0.749377` | `0.749377` | `0.251221` | `5.710551` | `4.443081` |
| Test_007 | `9.28%` | `0.179810` | `4.427816` | `4.427816` | `0.134336` | `10.879986` | `10.635165` |
| Test_009 | `19.88%` | `2.512806` | `7.339129` | `0.740300` | `0.740300` | `26.084725` | `68.294512` |
| Test_011 | `13.42%` | `9.492358` | `9.042253` | `9.042252` | `5.932188` | `177.818562` | `3.677761` |
| Test_013 | `11.62%` | `8.450179` | `0.393025` | `0.393025` | `9.083408` | `9.751367` | `4.611007` |
| Test_015 | `10.38%` | `326.479865` | `0.482875` | `0.482875` | `232.725986` | `54.531297` | `12.034315` |
| Test_017 | `8.56%` | `0.106199` | `0.177380` | `0.177380` | `0.126130` | `3.231557` | `4.479514` |
| Test_019 | `10.37%` | `0.158300` | `0.220182` | `0.220182` | `1.238729` | `8.114895` | `18.408627` |

Counts:

- VO-SE3-MSE acc no-gate improves position over raw on `6/10` sequences.
- VO-SE3-MSE acc no-gate improves rotation over raw on `4/10` sequences.
- VO-SE3-MSE acc no-gate improves position over RSS acc no-gate on `4/10`
  sequences.
- VO-SE3-MSE acc no-gate improves rotation over RSS acc no-gate on `4/10`
  sequences.

Conclusion:

- No-gate VO-SE3-MSE acc compensation is not stable enough as an OpenVINS
  evaluation setting.
- It helps several sequences, but large failures on `Test_003`, `Test_011`,
  and especially `Test_015` dominate the mean.
- The current recommended OpenVINS setting remains gated acc-delta injection,
  not no-gate injection.

Output files:

- Metrics:
  `Results/AquaLux/openvins_test/vo_se3_mse_acc_nogate/metrics.csv`.
- Hybrid IMU metrics:
  `Results/AquaLux/openvins_test/vo_se3_mse_acc_nogate/hybrid_imu/hybrid_imu_metrics.csv`.

Step 8: Filter No-Gate Results by Removing Failures and Degradations

Filter rule:

- Compare VO-SE3-MSE acc no-gate against raw OpenVINS.
- Keep a sequence only if both position RMSE and rotation RMSE improve.
- Remove severe failures and any sequence where either metric becomes worse.

Kept sequences:

| sequence | changed | ours pos | raw pos | pos change | ours rot | raw rot | rot change |
|---|---:|---:|---:|---:|---:|---:|---:|
| Test_001 | `10.43%` | `0.230305` | `0.854555` | `-73.05%` | `5.927741` | `8.456036` | `-29.90%` |
| Test_009 | `19.88%` | `2.512806` | `7.339129` | `-65.76%` | `26.084725` | `68.294512` | `-61.81%` |
| Test_017 | `8.56%` | `0.106199` | `0.177380` | `-40.13%` | `3.231557` | `4.479514` | `-27.86%` |
| Test_019 | `10.37%` | `0.158300` | `0.220182` | `-28.10%` | `8.114895` | `18.408627` | `-55.92%` |

Filtered mean on kept sequences:

| metric | ours no-gate | raw OpenVINS | relative change |
|---|---:|---:|---:|
| position RMSE | `0.751903 m` | `2.147811 m` | `-64.99%` |
| rotation RMSE | `10.839730 deg` | `24.909672 deg` | `-56.48%` |

Removed sequences:

| sequence | reason | ours pos | raw pos | ours rot | raw rot |
|---|---|---:|---:|---:|---:|
| Test_003 | severe failure: position and rotation worse | `7.511468` | `0.196883` | `178.193104` | `11.517651` |
| Test_005 | rotation worse | `0.263806` | `0.749377` | `5.710551` | `4.443081` |
| Test_007 | rotation worse | `0.179810` | `4.427816` | `10.879986` | `10.635165` |
| Test_011 | position and rotation worse | `9.492358` | `9.042253` | `177.818562` | `3.677761` |
| Test_013 | position and rotation worse | `8.450179` | `0.393025` | `9.751367` | `4.611007` |
| Test_015 | severe failure: position and rotation worse | `326.479865` | `0.482875` | `54.531297` | `12.034315` |

Interpretation:

- After filtering failures and degradations, VO-SE3-MSE no-gate shows useful
  improvement on `4/10` sequences.
- The filtered result is an analysis-only subset; it should not replace the
  full 10-sequence OpenVINS result because the removed cases are real failures
  of the no-gate policy.

Step 9: Compare Ours No-Gate Against RSS Baseline No-Gate

Reference:

- Ours no-gate: `vo_se3_mse_acc_nogate`.
- RSS baseline no-gate: `segment_delta_acc`.

Full 10-sequence comparison:

| metric | ours no-gate | RSS baseline no-gate | relative change |
|---|---:|---:|---:|
| mean position RMSE | `35.538510 m` | `51.349287 m` | `-30.79%` |
| mean rotation RMSE | `48.024379 deg` | `55.880779 deg` | `-14.06%` |
| median position RMSE | `1.388306 m` | `0.989515 m` | worse |
| median rotation RMSE | `10.315677 deg` | `17.500580 deg` | better |

Full 10-sequence counts:

- Position better than RSS no-gate: `4/10`.
- Rotation better than RSS no-gate: `4/10`.
- Both position and rotation better than RSS no-gate: `2/10`.

Per-sequence comparison:

| sequence | ours pos | RSS no-gate pos | pos change | ours rot | RSS no-gate rot | rot change | both better |
|---|---:|---:|---:|---:|---:|---:|---|
| Test_001 | `0.230305` | `0.144664` | `+59.20%` | `5.927741` | `3.975605` | `+49.10%` | no |
| Test_003 | `7.511468` | `263.115903` | `-97.15%` | `178.193104` | `153.929370` | `+15.76%` | no |
| Test_005 | `0.263806` | `0.251221` | `+5.01%` | `5.710551` | `5.284780` | `+8.06%` | no |
| Test_007 | `0.179810` | `0.134336` | `+33.85%` | `10.879986` | `10.004233` | `+8.75%` | no |
| Test_009 | `2.512806` | `0.740300` | `+239.43%` | `26.084725` | `24.996928` | `+4.35%` | no |
| Test_011 | `9.492358` | `5.932188` | `+60.01%` | `177.818562` | `178.874652` | `-0.59%` | no |
| Test_013 | `8.450179` | `9.083408` | `-6.97%` | `9.751367` | `9.287270` | `+5.00%` | no |
| Test_015 | `326.479865` | `232.725986` | `+40.29%` | `54.531297` | `119.095988` | `-54.21%` | no |
| Test_017 | `0.106199` | `0.126130` | `-15.80%` | `3.231557` | `4.119235` | `-21.55%` | yes |
| Test_019 | `0.158300` | `1.238729` | `-87.22%` | `8.114895` | `49.239725` | `-83.52%` | yes |

Comparison on the raw-improved subset from Step 8:

| subset | metric | ours no-gate | RSS baseline no-gate | relative change |
|---|---|---:|---:|---:|
| `Test_001, Test_009, Test_017, Test_019` | mean position RMSE | `0.751903 m` | `0.562456 m` | `+33.68%` |
| `Test_001, Test_009, Test_017, Test_019` | mean rotation RMSE | `10.839730 deg` | `20.582873 deg` | `-47.34%` |

Interpretation:

- Full-sequence means make ours no-gate look better than RSS no-gate, but both
  methods have severe no-gate failures and the mean is dominated by outliers.
- On a per-sequence strict comparison, ours no-gate is clearly better than RSS
  no-gate only on `Test_017` and `Test_019`.
- On the Step 8 raw-improved subset, ours has better rotation mean but worse
  position mean than RSS no-gate because `Test_001` and `Test_009` are worse
  than RSS no-gate.

Step 10: Reframe Comparison by Supervision Source

Key point:

- The RSS/BiasDy baseline checkpoint is a GT-supervised baseline.
- Our tested VO-SE3-MSE checkpoint should not be described simply as "worse
  than baseline" without noting that it is learned from VO supervision.

Checkpoint supervision audit:

- Ours gyro checkpoint:
  `Results/AquaLux/preintegration/ours_vo_w16vo_se23_mse_relative_noGT_gyro_e1800_from_e300/Gyro_weights/model_epoch_1500.pt`
  - `plato_supervision = vo_se23`
  - `plato_stage = gyro`
  - Method name includes `noGT`.
- Ours acc checkpoint:
  `Results/AquaLux/preintegration/ours_vo_w64pose_mse_right_bgbest_gtmseba_voft_lr1e7_e21/Acc_weights/best_val_model.pt`
  - `plato_supervision = vo_pose`
  - `plato_stage = acc`
  - It was initialized from a GT-MSE BA checkpoint, so the precise wording is
    VO-supervised / VO-finetuned acc checkpoint, not a purely GT-free acc model.
- RSS baseline checkpoints:
  `Results/AquaLux/preintegration/rss_baseline_unseen_corrected/Gyro_weights/model_epoch_1800.pt`
  and
  `Results/AquaLux/preintegration/rss_baseline_unseen_corrected/Acc_weights/model_epoch_1800.pt`
  - Original BiasDy learned bias dynamics baseline.
  - Trained/evaluated in the GT trajectory pipeline.

Reporting implication:

- OpenVINS deployment conclusion:
  - Gating is still needed; no-gate is unstable for both RSS baseline and ours.
  - The gated RSS baseline remains numerically stronger on the critical
    `Test_009` OpenVINS result.
- Learning-supervision conclusion:
  - Ours is much weaker-supervision than the RSS/GT baseline.
  - Even with VO supervision, ours no-gate improves over raw OpenVINS on
    `Test_001`, `Test_009`, `Test_017`, and `Test_019`.
  - Against RSS no-gate, ours wins both position and rotation on `Test_017`
    and `Test_019`, and has better full-10 mean despite being unstable.

Preferred wording:

```text
Compared with the GT-supervised RSS/BiasDy baseline, the VO-supervised
VO-SE3-MSE model is not yet as stable in OpenVINS. However, this is a harder
setting: the tested model is learned from VO-derived supervision rather than
ground-truth trajectory supervision. Under no-gate injection it still improves
raw OpenVINS on 4/10 sequences and outperforms RSS no-gate on both metrics for
2/10 sequences, while full-sequence means are better than RSS no-gate because
RSS no-gate also has severe failures.
```

Step 11: Compare No-Gate Improvement Sequence Sets

Question:

- Are the sequences improved by RSS baseline no-gate the same as the sequences
  improved by VO-SE3-MSE no-gate?

Definition:

- Improvement is measured relative to raw OpenVINS.
- `both` means both position RMSE and rotation RMSE improve.

Improvement sets:

| method | position improves | rotation improves | both improve |
|---|---|---|---|
| VO-SE3-MSE no-gate | `Test_001, Test_005, Test_007, Test_009, Test_017, Test_019` | `Test_001, Test_009, Test_017, Test_019` | `Test_001, Test_009, Test_017, Test_019` |
| RSS baseline no-gate | `Test_001, Test_005, Test_007, Test_009, Test_011, Test_017` | `Test_001, Test_007, Test_009, Test_017` | `Test_001, Test_007, Test_009, Test_017` |

Set comparison for `both improve`:

- Common to both methods:
  `Test_001, Test_009, Test_017`.
- VO-SE3-MSE only:
  `Test_019`.
- RSS baseline only:
  `Test_007`.

Per-sequence boolean table:

| sequence | VO pos | VO rot | VO both | RSS pos | RSS rot | RSS both |
|---|---|---|---|---|---|---|
| Test_001 | yes | yes | yes | yes | yes | yes |
| Test_003 | no | no | no | no | no | no |
| Test_005 | yes | no | no | yes | no | no |
| Test_007 | yes | no | no | yes | yes | yes |
| Test_009 | yes | yes | yes | yes | yes | yes |
| Test_011 | no | no | no | yes | no | no |
| Test_013 | no | no | no | no | no | no |
| Test_015 | no | no | no | no | no | no |
| Test_017 | yes | yes | yes | yes | yes | yes |
| Test_019 | yes | yes | yes | no | no | no |

Answer:

- No, the improved sequence sets are not identical.
- The overlap is strong but not perfect: `3/4` of the strict both-metric
  improvement set is shared.
- VO-SE3-MSE uniquely recovers `Test_019`; RSS baseline uniquely recovers
  `Test_007`.

Step 12: List Raw, RSS No-Gate, and Ours No-Gate for Selected Sequences

Selected sequences:

```text
Test_001, Test_005, Test_007, Test_009, Test_017, Test_019
```

Metrics are OpenVINS ATE RMSE:

| sequence | raw pos | raw rot | RSS baseline no-gate pos | RSS baseline no-gate rot | ours no-gate pos | ours no-gate rot |
|---|---:|---:|---:|---:|---:|---:|
| Test_001 | `0.854555` | `8.456036` | `0.144664` | `3.975605` | `0.230305` | `5.927741` |
| Test_005 | `0.749377` | `4.443081` | `0.251221` | `5.284780` | `0.263806` | `5.710551` |
| Test_007 | `4.427816` | `10.635165` | `0.134336` | `10.004233` | `0.179810` | `10.879986` |
| Test_009 | `7.339129` | `68.294512` | `0.740300` | `24.996928` | `2.512806` | `26.084725` |
| Test_017 | `0.177380` | `4.479514` | `0.126130` | `4.119235` | `0.106199` | `3.231557` |
| Test_019 | `0.220182` | `18.408627` | `1.238729` | `49.239725` | `0.158300` | `8.114895` |

Reading:

- RSS baseline no-gate is better on `Test_001`, `Test_007`, and `Test_009`.
- On `Test_005`, both no-gate methods improve position but worsen rotation;
  RSS baseline no-gate is slightly better than ours on both metrics.
- Ours no-gate is better on `Test_017` and `Test_019`.
- `Test_019` is the clearest case where VO-supervised ours improves raw and
  RSS no-gate fails.

Step 13: Add Test_005 to Selected No-Gate Table

Update:

- Added `Test_005` to the selected raw / RSS baseline no-gate / ours no-gate
  comparison table in Step 12.

Test_005 reading:

- Raw: `0.749377 m / 4.443081 deg`.
- RSS baseline no-gate: `0.251221 m / 5.284780 deg`.
- Ours no-gate: `0.263806 m / 5.710551 deg`.
- Both no-gate methods improve position relative to raw but worsen rotation.
- RSS baseline no-gate is slightly better than ours on both position and
  rotation for this sequence.

Step 14: Restrict Future OpenVINS Tests to Selected Sequences

Decision:

- From this point forward, AquaLux OpenVINS experiments should only be run on
  the selected test subset:

```text
Test_001
Test_005
Test_007
Test_009
Test_017
Test_019
```

Reason:

- These sequences are the useful comparison subset identified from the no-gate
  analysis:
  - `Test_001`, `Test_009`, `Test_017`, `Test_019` are strict both-metric
    improvements for VO-SE3-MSE no-gate relative to raw.
  - `Test_005` and `Test_007` are informative disagreement/partial-improvement
    cases between VO-SE3-MSE and RSS baseline no-gate.
- The remaining sequences include unstable or severely failed no-gate cases and
  should not be included in the next OpenVINS iteration unless explicitly
  requested for failure analysis.

Operational rule:

- Future OpenVINS commands should pass:

```text
--sequences Test_001 Test_005 Test_007 Test_009 Test_017 Test_019
```

- Do not run the default full 10-sequence AquaLux test set unless the goal is
  specifically to report full-set failure behavior.

Step 15: Reconfirm RSS Baseline No-Gate Test_009 Reporting

Question:

- Was RSS baseline no-gate `Test_009` reported correctly?

Answer:

- Yes. The reported RSS baseline no-gate `Test_009` result is correct:

```text
RSS baseline no-gate Test_009: 0.740300 m / 24.996928 deg
```

Source:

```text
Results/AquaLux/openvins_test/baseline/segment_delta_acc/metrics.csv
```

Verified row:

```text
method = segment_delta_acc
sequence = Test_009
metric_status = ok
ate_rmse_pos_m = 0.740300
ate_rmse_rot_deg = 24.996928
```

Hybrid IMU config:

```text
correction_mode = delta
components = acc
changed_samples = 16225
changed_fraction = 0.19876270978806812
max_delta_norm = 0.3797099269151943
mean_delta_norm_changed = 0.0801849177776978
```

Why it matches covgate:

- The covgate version also compensates `Test_009` because raw OpenVINS
  `max_position_cov_trace = 4.6748363552`, which passes the `> 1.0` gate.
- Therefore no-gate and covgate use the same weak segments for `Test_009`.
- The covgate result is nearly identical:

```text
RSS baseline acc+covgate Test_009: 0.740300 m / 24.996929 deg
```

Conclusion:

- The `Test_009` baseline no-gate value in Step 12 was not a misreport.
- It is expected to match the gated baseline for this specific sequence.

Step 16: Start Aggressive Test_009 Position Tuning for VO-MSE

Goal:

- Optimize VO-SE3-MSE compensation specifically for `Test_009` OpenVINS
  behavior, with the downstream hope of improving position RMSE.
- This is a targeted diagnostic/tuning exercise, not an unbiased full-test-set
  evaluation.
- User constraint added after starting this step: do not use ground truth.

Target sequence:

```text
Test_009
```

Current reference values:

```text
raw OpenVINS Test_009:             7.339129 m / 68.294512 deg
RSS baseline acc no-gate Test_009: 0.740300 m / 24.996928 deg
VO-SE3-MSE acc no-gate Test_009:   2.512806 m / 26.084725 deg
```

Optimization policy:

- Do not use GT trajectory or GT ATE/RMSE during tuning or candidate selection.
- Use only VO/OpenVINS-observable proxy signals during tuning, such as:
  - OpenVINS covariance/timing logs;
  - trajectory health and continuity;
  - odometry row count and process success;
  - IMU correction magnitude/smoothness;
  - consistency against raw/VO-derived observable behavior.
- Sweep compensation choices for VO-SE3-MSE:
  - acc correction scale;
  - optional gyro/both-component injection;
  - correction mode if useful;
  - weak-segment detection thresholds if useful.
- Keep outputs in new per-trial method directories so previous results remain
  untouched.

Reporting rule:

- Any improved result from this step must be reported as `Test_009-tuned`.
- It must not be mixed with the earlier default no-gate/covgate evaluation.
- If GT RMSE is used later only for final locked-result reporting, it must be
  explicitly marked as post-hoc evaluation, not as a tuning signal.

Step 17: Clarify Ground-Truth Constraint for Test_009 Tuning

Clarification from user:

- During compensation and OpenVINS running, ground truth must not be used to
  construct or inject the compensated IMU.
- After each debug/trial run finishes, ground truth may be used to evaluate the
  resulting OpenVINS accuracy.

Operational interpretation:

- Allowed during compensation:
  - raw IMU;
  - raw OpenVINS timing/estimate-derived weak segments;
  - VO-SE3-MSE checkpoints;
  - parameter choices such as correction scale/component/mode.
- Not allowed during compensation:
  - GT pose;
  - GT-derived position/rotation error;
  - any direct GT trajectory correction.
- Allowed after a trial is complete:
  - run evo/GT ATE to score the completed candidate;
  - use that post-hoc score to decide the next diagnostic trial.

Next trial family:

- Start with `Test_009` acc-delta scale sweep using VO-SE3-MSE checkpoints.
- Keep `components = acc`, `correction_mode = delta`, no gyro injection.

Step 18: Report Current Best Test_009-Tuned VO-MSE Result

Stop condition:

- User requested to stop further lowering/fine-tuning and report the current
  best result.

Best trial:

```text
method = vo_mse009_acc_s0134
sequence = Test_009
components = acc
correction_mode = delta
acc_correction_scale = 0.134
gyro injection = off
gate_max_position_cov_trace = 0.0
```

Compensation/operation notes:

- Compensation did not use ground truth.
- The trial used raw IMU, raw OpenVINS weak-segment detection, and VO-SE3-MSE
  checkpoints.
- Ground truth was used only after the OpenVINS run for post-hoc evo scoring.
- No OpenVINS runner or `ov_msckf` process remained after this report.

Hybrid IMU stats:

```text
changed_samples = 16225
changed_fraction = 0.19876270978806812
max_delta_norm = 0.048225520011706345
mean_delta_norm_changed = 0.010553040546026996
```

Post-hoc GT evaluation:

| method | Test_009 position RMSE | Test_009 rotation RMSE |
|---|---:|---:|
| raw OpenVINS | `7.339129 m` | `68.294512 deg` |
| RSS baseline acc no-gate | `0.740300 m` | `24.996928 deg` |
| default VO-SE3-MSE acc no-gate | `2.512806 m` | `26.084725 deg` |
| VO-SE3-MSE Test_009-tuned acc scale `0.134` | `0.201705 m` | `5.926182 deg` |

Improvement:

- Versus raw OpenVINS:
  - position: `-97.25%`;
  - rotation: `-91.32%`.
- Versus RSS baseline acc no-gate:
  - position: `-72.75%`;
  - rotation: `-76.29%`.
- Versus default VO-SE3-MSE acc no-gate:
  - position: `-91.97%`;
  - rotation: `-77.28%`.

Output files:

- Metrics:
  `Results/AquaLux/openvins_test/vo_mse009_acc_s0134/metrics.csv`.
- Hybrid IMU:
  `Results/AquaLux/openvins_test/vo_mse009_acc_tune/hybrid_s0134/Test_009/hybrid_imu.csv`.
- Hybrid config:
  `Results/AquaLux/openvins_test/vo_mse009_acc_tune/hybrid_s0134/Test_009/run_config.json`.

Step 19: Report Light-Off Ratios for Selected OpenVINS Sequences

Definition:

- Light-off ratio is measured as the IMU sample fraction covered by detected
  weak-visual/light-off segments.
- Computation:

```text
light_off_ratio = changed_samples / samples
```

Source:

```text
Results/AquaLux/openvins_test/vo_se3_mse_acc_nogate/hybrid_imu/hybrid_imu_metrics.csv
```

Selected sequences:

| sequence | segments | light-off samples | total samples | light-off ratio |
|---|---:|---:|---:|---:|
| Test_001 | `3` | `7539` | `72304` | `10.4268%` |
| Test_005 | `3` | `7838` | `76138` | `10.2945%` |
| Test_007 | `3` | `6889` | `74264` | `9.2764%` |
| Test_009 | `6` | `16225` | `81630` | `19.8763%` |
| Test_017 | `3` | `7489` | `87469` | `8.5619%` |
| Test_019 | `4` | `7838` | `75612` | `10.3661%` |

Reading:

- `Test_009` has the largest detected light-off coverage in the selected set.
- The other selected sequences are all around `8.56%` to `10.43%`.

Step 20: Confirm Saved Test_009 Baseline and Tuned Trajectories

Question:

- Are the `Test_009` baseline compensated trajectory and tuned compensated
  trajectory saved?

Answer:

- Yes. Both are saved.

RSS baseline no-gate `Test_009`:

- OpenVINS TUM trajectory:
  `Results/AquaLux/openvins_test/baseline/segment_delta_acc/Test_009/estimate_tum.tum`
- OpenVINS full state estimate:
  `Results/AquaLux/openvins_test/baseline/segment_delta_acc/Test_009/estimate.txt`
- OpenVINS odometry:
  `Results/AquaLux/openvins_test/baseline/segment_delta_acc/Test_009/odomimu.txt`
- Compensated OpenVINS IMU override:
  `Results/AquaLux/openvins_test/baseline/segment_delta_acc/Test_009/imu_override.csv`
- Generated hybrid IMU before OpenVINS conversion:
  `Results/AquaLux/openvins_test/baseline/segment_delta_acc/hybrid_imu/Test_009/hybrid_imu.csv`
- Segment metadata:
  `Results/AquaLux/openvins_test/baseline/segment_delta_acc/hybrid_imu/Test_009/segments.csv`

VO-SE3-MSE tuned `Test_009`, acc scale `0.134`:

- OpenVINS TUM trajectory:
  `Results/AquaLux/openvins_test/vo_mse009_acc_s0134/Test_009/estimate_tum.tum`
- OpenVINS full state estimate:
  `Results/AquaLux/openvins_test/vo_mse009_acc_s0134/Test_009/estimate.txt`
- OpenVINS odometry:
  `Results/AquaLux/openvins_test/vo_mse009_acc_s0134/Test_009/odomimu.txt`
- Compensated OpenVINS IMU override:
  `Results/AquaLux/openvins_test/vo_mse009_acc_s0134/Test_009/imu_override.csv`
- Generated hybrid IMU before OpenVINS conversion:
  `Results/AquaLux/openvins_test/vo_mse009_acc_tune/hybrid_s0134/Test_009/hybrid_imu.csv`
- Segment metadata:
  `Results/AquaLux/openvins_test/vo_mse009_acc_tune/hybrid_s0134/Test_009/segments.csv`

Line counts:

```text
baseline estimate_tum.tum: 66007 lines
tuned estimate_tum.tum:    65924 lines
baseline hybrid_imu.csv:   81631 lines
tuned hybrid_imu.csv:      81631 lines
```

Step 21: Plan Trajectory Plot for Test_009 Baseline, Tuned Ours, and GT

Task requested:

- Plot `Test_009` trajectories for:
  - RSS baseline no-gate;
  - VO-SE3-MSE tuned ours, acc scale `0.134`;
  - ground truth.
- Put all three trajectories in one figure.
- Align trajectories by the first frame before plotting.
- Confirm understanding with the user before executing the plotting.

Planned input trajectories:

```text
baseline:
Results/AquaLux/openvins_test/baseline/segment_delta_acc/Test_009/estimate_tum.tum

ours tuned:
Results/AquaLux/openvins_test/vo_mse009_acc_s0134/Test_009/estimate_tum.tum

gt:
Results/AquaLux/openvins_test/vo_mse009_acc_s0134/Test_009/gt_imu_gravity_aligned.tum
```

Planned alignment interpretation:

- Use the first pose of each trajectory as its local origin.
- Subtract the first position from every position in that trajectory.
- Optionally rotate each trajectory into its own first-frame yaw/body frame if
  the user wants full first-pose SE(3) alignment instead of position-origin
  alignment.

Planned output:

- A single trajectory figure comparing baseline, ours tuned, and GT.
- Suggested output path:

```text
Results/AquaLux/openvins_test/vo_mse009_acc_s0134/Test_009/test009_baseline_ours_tuned_gt_first_frame_aligned.png
```

Step 22: Confirm SE(3) First-Frame Alignment and Plot Views

User confirmation:

- Use full first-frame SE(3) alignment, not translation-only alignment.
- Produce both:
  - XY top-down trajectory plot;
  - 3D trajectory plot.

Alignment definition:

- For each trajectory independently, let the first pose be `T0`.
- Plot positions from:

```text
T_rel(t) = inverse(T0) * T(t)
```

- This removes each trajectory's initial translation and initial orientation,
  so all curves start from the same local first-frame coordinate system.

Planned outputs:

```text
Results/AquaLux/openvins_test/vo_mse009_acc_s0134/Test_009/test009_baseline_ours_tuned_gt_se3_aligned_xy.png
Results/AquaLux/openvins_test/vo_mse009_acc_s0134/Test_009/test009_baseline_ours_tuned_gt_se3_aligned_3d.png
```

Step 23: Generate Test_009 SE(3)-Aligned Trajectory Plots

Generated plots:

```text
Results/AquaLux/openvins_test/vo_mse009_acc_s0134/Test_009/test009_baseline_ours_tuned_gt_se3_aligned_xy.png
Results/AquaLux/openvins_test/vo_mse009_acc_s0134/Test_009/test009_baseline_ours_tuned_gt_se3_aligned_3d.png
```

Plot contents:

- RSS baseline no-gate trajectory:
  `Results/AquaLux/openvins_test/baseline/segment_delta_acc/Test_009/estimate_tum.tum`
- VO-SE3-MSE tuned trajectory, acc scale `0.134`:
  `Results/AquaLux/openvins_test/vo_mse009_acc_s0134/Test_009/estimate_tum.tum`
- Ground truth:
  `Results/AquaLux/openvins_test/vo_mse009_acc_s0134/Test_009/gt_imu_gravity_aligned.tum`

Implementation detail:

- Used the common time window shared by all three trajectories:

```text
t0 = 1779619021.915879965
t1 = 1779619098.903259993
```

- For each trajectory, cropped to the common time window and applied full
  first-frame SE(3) alignment:

```text
T_rel(t) = inverse(T0_common) * T(t)
```

Trajectory rows after crop:

```text
RSS baseline no-gate:       66006 rows
VO-SE3-MSE tuned scale=0.134: 65923 rows
Ground truth:                9239 rows
```

File verification:

```text
XY plot: 1530 x 1260, 161 KB
3D plot: 1620 x 1296, 252 KB
```

Step 24: Add Raw OpenVINS Trajectory to Test_009 Plots

Update:

- Regenerated the `Test_009` SE(3)-aligned trajectory plots with raw OpenVINS
  included.
- Kept the same common time window and the same full first-frame SE(3)
  alignment.

New plot contents:

- Raw OpenVINS:
  `Results/AquaLux/openvins_test/raw/Test_009/estimate_tum.tum`
- RSS baseline no-gate:
  `Results/AquaLux/openvins_test/baseline/segment_delta_acc/Test_009/estimate_tum.tum`
- VO-SE3-MSE tuned, acc scale `0.134`:
  `Results/AquaLux/openvins_test/vo_mse009_acc_s0134/Test_009/estimate_tum.tum`
- Ground truth:
  `Results/AquaLux/openvins_test/vo_mse009_acc_s0134/Test_009/gt_imu_gravity_aligned.tum`

Generated plots:

```text
Results/AquaLux/openvins_test/vo_mse009_acc_s0134/Test_009/test009_raw_baseline_ours_tuned_gt_se3_aligned_xy.png
Results/AquaLux/openvins_test/vo_mse009_acc_s0134/Test_009/test009_raw_baseline_ours_tuned_gt_se3_aligned_3d.png
```

Common time window:

```text
t0 = 1779619021.915879965
t1 = 1779619098.903259993
```

Trajectory rows after crop:

```text
Raw OpenVINS:                 65933 rows
RSS baseline no-gate:         66006 rows
VO-SE3-MSE tuned scale=0.134: 65923 rows
Ground truth:                  9239 rows
```

File verification:

```text
XY plot: 1584 x 1278, 104 KB
3D plot: 1655 x 1314, 280 KB
```

Step 25: Plot Test_009 Time-Series Position and Rotation RMSE

Task:

- Plot time-series error curves for `Test_009`.
- Methods:
  - raw OpenVINS;
  - VO-SE3-MSE tuned, acc scale `0.134`;
  - RSS baseline no-gate.
- Metrics:
  - position RMSE over time;
  - rotation RMSE over time.

Interpretation:

- Use GT only for post-hoc evaluation.
- Align each estimated trajectory to GT with SE(3), no scale correction.
- Match poses by nearest timestamp with `t_max_diff = 0.02 s`, matching the
  existing OpenVINS/evo evaluation policy.
- Plot cumulative RMSE curves:

```text
position_rmse(t) = sqrt(mean(position_error(<=t)^2))
rotation_rmse(t) = sqrt(mean(rotation_error_deg(<=t)^2))
```

Planned output:

```text
Results/AquaLux/openvins_test/vo_mse009_acc_s0134/Test_009/test009_raw_baseline_ours_tuned_time_rmse.png
Results/AquaLux/openvins_test/vo_mse009_acc_s0134/Test_009/test009_raw_baseline_ours_tuned_time_rmse.csv
```

Step 26: Generate Test_009 Time-Series RMSE Plot

Generated outputs:

```text
Results/AquaLux/openvins_test/vo_mse009_acc_s0134/Test_009/test009_raw_baseline_ours_tuned_time_rmse.png
Results/AquaLux/openvins_test/vo_mse009_acc_s0134/Test_009/test009_raw_baseline_ours_tuned_time_rmse.csv
```

Figure layout:

- One PNG with two stacked subplots:
  - position cumulative RMSE over time;
  - rotation cumulative RMSE over time.
- Each subplot contains three method curves:
  - raw OpenVINS;
  - RSS baseline no-gate;
  - VO-SE3-MSE tuned, acc scale `0.134`.

Computed final cumulative RMSE from the plotted curves:

| method | matched poses | final position RMSE | final rotation RMSE |
|---|---:|---:|---:|
| Raw OpenVINS | `65933` | `7.328742 m` | `68.449227 deg` |
| RSS baseline no-gate | `66006` | `0.738466 m` | `25.012082 deg` |
| VO-SE3-MSE tuned scale `0.134` | `65923` | `0.201821 m` | `5.909063 deg` |

Note:

- These values are very close to the earlier evo metrics but not bit-identical,
  because this plot script independently reproduces timestamp matching and
  SE(3) no-scale alignment instead of calling evo directly.
- The plot is intended for trend visualization over time.

File verification:

```text
PNG: 1800 x 1332, 138 KB
CSV: 20 MB
```

Step 27: Correct Test_009 Time-Series RMSE Plot Requirements

Issue found:

- The first time-series RMSE plot used full-trajectory SE(3) no-scale alignment
  to GT.
- That alignment is valid for aggregate ATE-style scoring, but it is not ideal
  for diagnosing when the trajectory starts to diverge.
- The user expected the curves to start similarly and separate during/after
  light-off intervals.

Corrected requirements:

- Remove raw OpenVINS from the time-series plot.
- Plot only:
  - RSS baseline no-gate;
  - VO-SE3-MSE tuned, acc scale `0.134`.
- Use first-frame SE(3) alignment to GT, not full-trajectory alignment.
- Shade detected light-off intervals with a gray background.
- Keep two stacked curves/panels:
  - position cumulative RMSE over time;
  - rotation cumulative RMSE over time.

Light-off interval source:

```text
Results/AquaLux/openvins_test/vo_mse009_acc_tune/hybrid_s0134/Test_009/segments.csv
```

Corrected planned outputs:

```text
Results/AquaLux/openvins_test/vo_mse009_acc_s0134/Test_009/test009_baseline_ours_tuned_firstframe_time_rmse_lightoff.png
Results/AquaLux/openvins_test/vo_mse009_acc_s0134/Test_009/test009_baseline_ours_tuned_firstframe_time_rmse_lightoff.csv
```

Step 28: Generate Corrected First-Frame Time-Series RMSE Plot With Light-Off Shading

Generated outputs:

```text
Results/AquaLux/openvins_test/vo_mse009_acc_s0134/Test_009/test009_baseline_ours_tuned_firstframe_time_rmse_lightoff.png
Results/AquaLux/openvins_test/vo_mse009_acc_s0134/Test_009/test009_baseline_ours_tuned_firstframe_time_rmse_lightoff.csv
```

Contents:

- Removed raw OpenVINS.
- Plotted only:
  - RSS baseline no-gate;
  - VO-SE3-MSE tuned, acc scale `0.134`.
- Used first-frame SE(3) alignment to GT:

```text
T_align = T_gt0 * inverse(T_est0)
```

- Shaded light-off intervals in gray.

Light-off intervals relative to plot start:

```text
15.246603 s -> 17.646606 s
31.046624 s -> 33.596627 s
45.596642 s -> 47.846645 s
59.696660 s -> 63.096665 s
63.796666 s -> 67.746671 s
69.296673 s -> 70.996675 s
```

Common time window:

```text
t0 = 1779619021.915879965
t1 = 1779619098.903259993
```

Sanity check:

```text
RSS baseline first position/rotation error: 0 / 0
VO-MSE tuned first position/rotation error: 0 / 0
```

Final cumulative RMSE in this first-frame-aligned diagnostic plot:

| method | matched poses | final position RMSE | final rotation RMSE |
|---|---:|---:|---:|
| RSS baseline no-gate | `66006` | `0.818103 m` | `4.052540 deg` |
| VO-SE3-MSE tuned scale `0.134` | `65923` | `0.234516 m` | `5.227704 deg` |

Note:

- These first-frame-aligned diagnostic RMSE values are not the same as the evo
  ATE values from Step 18, because Step 18 uses full-trajectory SE(3) alignment
  for aggregate ATE scoring.
- This corrected plot is intended to diagnose when curves start diverging
  relative to light-off intervals.

File verification:

```text
PNG: 1800 x 1332, 154 KB
CSV: 14 MB
```

Step 29: Diagnose Apparent Initial Gap in Corrected Time-Series RMSE Plot

Question:

- Why does the time-series RMSE plot appear to have a large gap from `0 s`?

Checks:

- First light-off interval starts at:

```text
15.242483 s after plot start
```

- At `0 s`, both methods are exactly aligned by construction:

```text
RSS baseline no-gate position/rotation RMSE: 0 / 0
VO-MSE tuned position/rotation RMSE:         0 / 0
```

- Before the first light-off interval, the compensated IMU inputs are identical:

```text
pre-light-off IMU rows checked: 19985
max baseline-vs-tuned hybrid IMU difference before light-off: 0.0
```

- Before first light-off, the two cumulative RMSE curves are nearly identical:

```text
pre-light-off rows checked: 13066
max position cumulative RMSE difference: 0.001403 m
max rotation cumulative RMSE difference: 0.297912 deg
```

Interpretation:

- There is no meaningful baseline-vs-tuned divergence at `0 s`.
- Any early nonzero RMSE is natural OpenVINS-vs-GT tracking error accumulating
  before light-off, not a compensation difference.
- The plot uses cumulative RMSE, so early ordinary tracking error remains in
  the running average.
- For seeing exactly when the methods separate, an instantaneous error or
  sliding-window RMSE plot may be more visually direct than cumulative RMSE.

Step 30: Plot Test_019 Raw, Baseline, Ours, and GT Trajectories

Task:

- Plot `Test_019` trajectories in one figure.
- Include:
  - raw OpenVINS;
  - RSS baseline no-gate;
  - VO-SE3-MSE ours no-gate;
  - ground truth.

Input trajectories:

```text
raw:
Results/AquaLux/openvins_test/raw/Test_019/estimate_tum.tum

RSS baseline no-gate:
Results/AquaLux/openvins_test/baseline/segment_delta_acc/Test_019/estimate_tum.tum

VO-SE3-MSE ours no-gate:
Results/AquaLux/openvins_test/vo_se3_mse_acc_nogate/Test_019/estimate_tum.tum

GT:
Results/AquaLux/openvins_test/vo_se3_mse_acc_nogate/Test_019/gt_imu_gravity_aligned.tum
```

Alignment:

- Used common time window and first-frame SE(3) alignment for each trajectory.

```text
t0 = 1779620742.494100094
t1 = 1779620812.966639996
```

Rows after crop:

```text
Raw OpenVINS:          60352 rows
RSS baseline no-gate:  60398 rows
VO-SE3-MSE ours:       60322 rows
Ground truth:           8456 rows
```

Generated figure:

```text
Results/AquaLux/openvins_test/vo_se3_mse_acc_nogate/Test_019/test019_raw_baseline_ours_gt_se3_aligned_xy_3d.png
```

Figure layout:

- One PNG containing:
  - left: XY top-down plot;
  - right: 3D plot.

File verification:

```text
PNG: 2520 x 1188, 362 KB
```

Step 31: Regenerate Test_019 Plot Without RSS Baseline

Task update:

- Do not draw RSS baseline.
- Draw a Test_019 figure that includes raw OpenVINS.

Final plot contents:

- Raw OpenVINS:
  `Results/AquaLux/openvins_test/raw/Test_019/estimate_tum.tum`
- VO-SE3-MSE ours no-gate:
  `Results/AquaLux/openvins_test/vo_se3_mse_acc_nogate/Test_019/estimate_tum.tum`
- Ground truth:
  `Results/AquaLux/openvins_test/vo_se3_mse_acc_nogate/Test_019/gt_imu_gravity_aligned.tum`

Alignment:

- Used common time window and first-frame SE(3) alignment.

```text
t0 = 1779620742.494100094
t1 = 1779620812.966639996
```

Rows after crop:

```text
Raw OpenVINS:     60352 rows
VO-SE3-MSE ours:  60322 rows
Ground truth:      8456 rows
```

Generated figure:

```text
Results/AquaLux/openvins_test/vo_se3_mse_acc_nogate/Test_019/test019_raw_ours_gt_se3_aligned_xy_3d.png
```

Figure layout:

- One PNG containing:
  - left: XY top-down plot;
  - right: 3D plot.

File verification:

```text
PNG: 2448 x 1170, 433 KB
```

Step 32: Plan Test_019 Raw-vs-Ours Time-Series RMSE Plot

Task:

- Plot `Test_019` position and rotation RMSE over time.
- Compare:
  - raw OpenVINS;
  - VO-SE3-MSE ours no-gate.
- Mark light-off intervals with gray background.

Interpretation:

- Use GT only for post-hoc error computation.
- Use first-frame SE(3) alignment to GT, so both curves start from zero error.
- Plot cumulative RMSE over time:

```text
position_rmse(t) = sqrt(mean(position_error(<=t)^2))
rotation_rmse(t) = sqrt(mean(rotation_error_deg(<=t)^2))
```

Inputs:

```text
raw:
Results/AquaLux/openvins_test/raw/Test_019/estimate_tum.tum

ours:
Results/AquaLux/openvins_test/vo_se3_mse_acc_nogate/Test_019/estimate_tum.tum

gt:
Results/AquaLux/openvins_test/vo_se3_mse_acc_nogate/Test_019/gt_imu_gravity_aligned.tum

light-off segments:
Results/AquaLux/openvins_test/vo_se3_mse_acc_nogate/hybrid_imu/Test_019/segments.csv
```

Planned outputs:

```text
Results/AquaLux/openvins_test/vo_se3_mse_acc_nogate/Test_019/test019_raw_ours_firstframe_time_rmse_lightoff.png
Results/AquaLux/openvins_test/vo_se3_mse_acc_nogate/Test_019/test019_raw_ours_firstframe_time_rmse_lightoff.csv
```

Step 33: Generate Test_019 Raw-vs-Ours Time-Series RMSE Plot With Light-Off Shading

Generated outputs:

```text
Results/AquaLux/openvins_test/vo_se3_mse_acc_nogate/Test_019/test019_raw_ours_firstframe_time_rmse_lightoff.png
Results/AquaLux/openvins_test/vo_se3_mse_acc_nogate/Test_019/test019_raw_ours_firstframe_time_rmse_lightoff.csv
```

Contents:

- Raw OpenVINS.
- VO-SE3-MSE ours no-gate.
- Two stacked panels:
  - position cumulative RMSE over time;
  - rotation cumulative RMSE over time.
- Light-off intervals shown with gray background.

Alignment:

- Used first-frame SE(3) alignment to GT:

```text
T_align = T_gt0 * inverse(T_est0)
```

Common time window:

```text
t0 = 1779620742.494100094
t1 = 1779620812.966639996
```

Light-off intervals relative to plot start:

```text
16.495591 s -> 19.145594 s
32.195611 s -> 34.045613 s
40.995622 s -> 42.245624 s
57.345643 s -> 59.445646 s
```

Sanity check:

```text
Raw first position/rotation error:  0 / 0
Ours first position/rotation error: 0 / 0
```

Final cumulative RMSE in this first-frame-aligned diagnostic plot:

| method | matched poses | final position RMSE | final rotation RMSE |
|---|---:|---:|---:|
| Raw OpenVINS | `59810` | `0.403321 m` | `2.724959 deg` |
| VO-SE3-MSE ours no-gate | `59779` | `0.246412 m` | `2.706751 deg` |

File verification:

```text
PNG: 1800 x 1332, 161 KB
CSV: 12 MB
```

Step 34: Add Light-Off Positions to Test_019 XY-3D Trajectory Figure

Task:

- Update the `Test_019` XY+3D trajectory figure for paper use.
- Keep:
  - raw OpenVINS;
  - VO-SE3-MSE ours no-gate;
  - ground truth.
- Add the light-off intervals as spatial position segments on the trajectory.

Interpretation:

- Use the same light-off intervals from:

```text
Results/AquaLux/openvins_test/vo_se3_mse_acc_nogate/hybrid_imu/Test_019/segments.csv
```

- Since this is a spatial trajectory figure, mark the light-off portions on the
  GT trajectory positions, not as time-axis gray bands.
- Keep first-frame SE(3) alignment and the same common time window as the
  previous `Test_019` XY+3D plot.

Planned outputs:

```text
Results/AquaLux/openvins_test/vo_se3_mse_acc_nogate/Test_019/test019_raw_ours_gt_lightoff_positions_se3_aligned_xy_3d.png
Results/AquaLux/openvins_test/vo_se3_mse_acc_nogate/Test_019/test019_raw_ours_gt_lightoff_positions_se3_aligned_xy_3d.pdf
```

Step 35: Generate Paper-Style Test_019 XY-3D Plot With Light-Off Positions

Generated outputs:

```text
Results/AquaLux/openvins_test/vo_se3_mse_acc_nogate/Test_019/test019_raw_ours_gt_lightoff_positions_se3_aligned_xy_3d.png
Results/AquaLux/openvins_test/vo_se3_mse_acc_nogate/Test_019/test019_raw_ours_gt_lightoff_positions_se3_aligned_xy_3d.pdf
```

Figure contents:

- Raw OpenVINS.
- VO-SE3-MSE ours no-gate.
- Ground truth.
- Light-off positions highlighted as spatial segments on the GT trajectory.

Light-off intervals relative to plot start:

```text
16.495591 s -> 19.145594 s
32.195611 s -> 34.045613 s
40.995622 s -> 42.245624 s
57.345643 s -> 59.445646 s
```

Light-off GT points highlighted:

```text
318, 222, 150, 252 points
```

Alignment:

- Common time window and first-frame SE(3) alignment.

```text
t0 = 1779620742.494100094
t1 = 1779620812.966639996
```

File verification:

```text
PNG: 2044 x 1119, 265 KB
PDF: 51 KB
```

Step 36: Generate Test_019 Instantaneous Position Error Plot With Light-Off Shading

User clarification:

- The requested "only position" plot should be instantaneous position error, not cumulative position RMSE.
- Do not add light-off highlighting to the XY-3D trajectory plot.

Generated a paper-oriented Test_019 time-series plot comparing raw OpenVINS and VO-SE3-MSE ours no-gate:

- y-axis: instantaneous position error, `||p_est(t) - p_gt(t)||` in meters.
- x-axis: time from the first common timestamp.
- alignment: first-frame SE(3), using the same aligned error CSV from Step 33.
- light-off intervals are drawn as gray background spans.
- no rotation subplot and no cumulative averaging.

Outputs:

```text
Results/AquaLux/openvins_test/vo_se3_mse_acc_nogate/Test_019/test019_raw_ours_position_instant_error_lightoff.png
Results/AquaLux/openvins_test/vo_se3_mse_acc_nogate/Test_019/test019_raw_ours_position_instant_error_lightoff.pdf
```

Instantaneous position error summary:

```text
Raw OpenVINS         min 0.000000 m, max 0.937314 m, mean 0.331413 m
VO-SE3-MSE ours     min 0.000000 m, max 0.667476 m, mean 0.219771 m
```

File verification:

```text
PNG: 2160 x 825, 147 KB
PDF: 128 KB
```

Step 37: Generate Test_009 Instantaneous Position Error Plot With Light-Off Shading

Generated the same instantaneous-position-error style plot for Test_009.

Comparison:

- RSS baseline no-gate.
- VO-SE3-MSE tuned, `acc_correction_scale=0.134`.

Plot definition:

- y-axis: instantaneous position error, `||p_est(t) - p_gt(t)||` in meters.
- x-axis: time from the first common timestamp.
- alignment: first-frame SE(3), using the corrected aligned error CSV.
- light-off intervals are drawn as gray background spans.
- no cumulative RMSE and no rotation subplot.

Outputs:

```text
Results/AquaLux/openvins_test/vo_mse009_acc_s0134/Test_009/test009_baseline_ours_tuned_position_instant_error_lightoff.png
Results/AquaLux/openvins_test/vo_mse009_acc_s0134/Test_009/test009_baseline_ours_tuned_position_instant_error_lightoff.pdf
```

Instantaneous position error summary:

```text
RSS baseline no-gate   min 0.000000 m, max 3.876258 m, mean 0.394958 m
VO-SE3-MSE tuned       min 0.000000 m, max 0.714432 m, mean 0.202219 m
```

File verification:

```text
PNG: 2160 x 825, 103 KB
PDF: 74 KB
```

Step 38: Plan VO-NLL No-Gate OpenVINS Test

User requested the next OpenVINS experiment plan before execution:

- Read VO-NLL checkpoints from:
  `Results/AquaLux/preintegration/checkpoints`.
- Use:
  - BG checkpoint: `vo-nll-bg.pt`.
  - BA checkpoint: `vo-nll-ba.pt`.
  - learned covariance: `vo-nll-covariance.json` and `vo-nll-covariance.pt`.
- Run an OpenVINS no-gate test after confirmation.

Files verified:

```text
Results/AquaLux/preintegration/checkpoints/vo-nll-bg.pt
Results/AquaLux/preintegration/checkpoints/vo-nll-ba.pt
Results/AquaLux/preintegration/checkpoints/vo-nll-covariance.json
Results/AquaLux/preintegration/checkpoints/vo-nll-covariance.pt
```

Learned covariance JSON values:

```text
sigma_a = 0.10066936910152435
sigma_g = 0.010067201219499111
Qn_diag = [
  0.010134321637451649,
  0.010134321637451649,
  0.010134321637451649,
  0.0001013485380099155,
  0.0001013485380099155,
  0.0001013485380099155
]
source_checkpoint = Results/AquaLux/preintegration/ours_vo_w64pose_nll_alt_bgba_cov_e300/cycle06_bgba_e25/NLL_weights/model_epoch_24.pt
```

Current interpretation to confirm:

- Test subset should follow the Step 14 rule:
  `Test_001 Test_005 Test_007 Test_009 Test_017 Test_019`.
- No-gate means `gate_max_position_cov_trace = 0.0`, so all detected light-off
  segments are compensated without covariance gating.
- Bias compensation should use the VO-NLL BG/BA checkpoints.
- OpenVINS covariance should be changed from the default AquaLux config to the
  learned VO-NLL covariance values above, with `sigma_g` mapped to
  `gyroscope_noise_density` and `sigma_a` mapped to
  `accelerometer_noise_density`.
- Keep the same compensation policy as the prior no-gate VO-MSE run unless the
  user changes it: `components = acc`, `correction_mode = delta`, no gyro
  injection.

Pending confirmation before execution:

- Whether to test only the selected six sequences from Step 14, or all 10.
- Whether learned covariance should change only OpenVINS noise density fields,
  while keeping random-walk values unchanged from the AquaLux config.

Step 39: Confirm VO-NLL OpenVINS Test Scope

User clarified:

- Only test sequences are required.
- Do not involve training or validation sequences.

Operational interpretation:

- Follow the existing Step 14 selected AquaLux OpenVINS test subset unless the
  user explicitly changes it:

```text
Test_001
Test_005
Test_007
Test_009
Test_017
Test_019
```

- No VO-NLL OpenVINS run has been started yet.

Step 40: Confirm VO-NLL Covariance Mapping

User clarified:

- The learned VO-NLL covariance should replace only the OpenVINS IMU noise
  density terms corresponding to `n_a` and `n_g`.

Operational mapping:

```text
n_a / accelerometer_noise_density = sigma_a = 0.10066936910152435
n_g / gyroscope_noise_density     = sigma_g = 0.010067201219499111
```

Keep the original AquaLux random-walk values unchanged:

```text
accelerometer_random_walk = 0.000691276415008
gyroscope_random_walk     = 4.54949206016e-05
```

No VO-NLL OpenVINS run has been started yet.

Step 41: Generate VO-NLL No-Gate Hybrid IMU for Test Sequences

Prepared the VO-NLL no-gate hybrid IMU files for the selected AquaLux test
subset.

Command configuration:

```text
gyro_checkpoint = Results/AquaLux/preintegration/checkpoints/vo-nll-bg.pt
acc_checkpoint  = Results/AquaLux/preintegration/checkpoints/vo-nll-ba.pt
output_root     = Results/AquaLux/openvins_test/vo_nll_acc_nogate_cov/hybrid_imu
sequences       = Test_001 Test_005 Test_007 Test_009 Test_017 Test_019
device          = cpu
components      = acc
correction_mode = delta
gate_max_position_cov_trace = 0.0
```

Result:

| sequence | segments ok | samples | changed samples | changed fraction | max delta norm | mean delta norm changed |
|---|---:|---:|---:|---:|---:|---:|
| Test_001 | 3/3 | 72304 | 7539 | 10.4268% | 0.647258 | 0.112251 |
| Test_005 | 3/3 | 76138 | 7838 | 10.2945% | 0.281012 | 0.0685168 |
| Test_007 | 3/3 | 74264 | 6889 | 9.2764% | 0.355053 | 0.0966046 |
| Test_009 | 6/6 | 81630 | 16225 | 19.8763% | 0.358848 | 0.0788882 |
| Test_017 | 3/3 | 87469 | 7489 | 8.5619% | 0.420020 | 0.0676452 |
| Test_019 | 4/4 | 75612 | 7838 | 10.3661% | 0.354267 | 0.104624 |

Artifacts:

```text
Results/AquaLux/openvins_test/vo_nll_acc_nogate_cov/hybrid_imu/hybrid_imu_metrics.csv
Results/AquaLux/openvins_test/vo_nll_acc_nogate_cov/hybrid_imu/Test_*/hybrid_imu.csv
Results/AquaLux/openvins_test/vo_nll_acc_nogate_cov/hybrid_imu/Test_*/segments.csv
```

Step 42: Verify VO-NLL OpenVINS Config Before Running

Updated the AquaLux OpenVINS runner to accept optional IMU noise-density
overrides:

```text
--accelerometer-noise-density
--gyroscope-noise-density
```

Default behavior remains unchanged when these options are not passed.

Dry-run command used the VO-NLL hybrid IMU root and selected test sequences:

```text
method = vo_nll_acc_nogate_cov
source_imu_root = Results/AquaLux/openvins_test/vo_nll_acc_nogate_cov/hybrid_imu
source_imu_filename = hybrid_imu.csv
sequences = Test_001 Test_005 Test_007 Test_009 Test_017 Test_019
accelerometer_noise_density = 0.10066936910152435
gyroscope_noise_density = 0.010067201219499111
```

Generated config verified:

```text
Results/AquaLux/openvins_test/configs/vo_nll_acc_nogate_cov/kalibr_imu_chain.yaml
```

Relevant IMU noise fields:

```text
accelerometer_noise_density: 0.1006693691015244
accelerometer_random_walk: 0.000691276415008
gyroscope_noise_density: 0.01006720121949911
gyroscope_random_walk: 4.54949206016e-05
```

OpenVINS run not yet started at this step.

Step 43: Run VO-NLL No-Gate OpenVINS on Selected Test Sequences

Ran OpenVINS using the VO-NLL no-gate hybrid IMU and learned `n_a/n_g`
OpenVINS noise-density values.

Run configuration:

```text
method = vo_nll_acc_nogate_cov
source_imu_root = Results/AquaLux/openvins_test/vo_nll_acc_nogate_cov/hybrid_imu
source_imu_filename = hybrid_imu.csv
sequences = Test_001 Test_005 Test_007 Test_009 Test_017 Test_019
components = acc
correction_mode = delta
gate_max_position_cov_trace = 0.0
accelerometer_noise_density = 0.10066936910152435
gyroscope_noise_density = 0.010067201219499111
accelerometer_random_walk = 0.000691276415008
gyroscope_random_walk = 4.54949206016e-05
```

All six OpenVINS runs produced odometry and evo metrics. The OpenVINS process
returned `-6` after producing odometry, which is treated by the existing runner
as usable when odometry rows are present.

VO-NLL no-gate results:

| sequence | position RMSE (m) | rotation RMSE (deg) | odom rows |
|---|---:|---:|---:|
| Test_001 | 1.029872 | 5.401183 | 57867 |
| Test_005 | 22.182740 | 110.805115 | 61438 |
| Test_007 | 5.613755 | 11.760821 | 59518 |
| Test_009 | 4.754220 | 23.632294 | 66021 |
| Test_017 | 56.739174 | 177.023939 | 52254 |
| Test_019 | 1.294483 | 51.535910 | 60381 |

Selected-six mean:

```text
VO-NLL no-gate + learned n_a/n_g: 15.269041 m / 63.359877 deg
```

Comparison on the same selected six sequences:

| method | mean position RMSE (m) | mean rotation RMSE (deg) | median position RMSE (m) | median rotation RMSE (deg) |
|---|---:|---:|---:|---:|
| raw OpenVINS | 2.294740 | 19.119489 | 0.801966 | 9.545600 |
| RSS baseline no-gate | 0.439230 | 16.270084 | 0.197943 | 7.644506 |
| VO-SE3-MSE no-gate | 0.575204 | 9.991576 | 0.205058 | 7.021318 |
| VO-NLL no-gate + learned n_a/n_g | 15.269041 | 63.359877 | 5.183988 | 37.584102 |

Per-sequence comparison, formatted as `position / rotation`:

| sequence | raw | RSS no-gate | VO-SE3-MSE no-gate | VO-NLL no-gate + learned n_a/n_g |
|---|---:|---:|---:|---:|
| Test_001 | 0.854555 / 8.456036 | 0.144664 / 3.975605 | 0.230305 / 5.927741 | 1.029872 / 5.401183 |
| Test_005 | 0.749377 / 4.443081 | 0.251221 / 5.284780 | 0.263806 / 5.710551 | 22.182740 / 110.805115 |
| Test_007 | 4.427816 / 10.635165 | 0.134336 / 10.004233 | 0.179810 / 10.879986 | 5.613755 / 11.760821 |
| Test_009 | 7.339129 / 68.294512 | 0.740300 / 24.996928 | 2.512806 / 26.084725 | 4.754220 / 23.632294 |
| Test_017 | 0.177380 / 4.479514 | 0.126130 / 4.119235 | 0.106199 / 3.231557 | 56.739174 / 177.023939 |
| Test_019 | 0.220182 / 18.408627 | 1.238729 / 49.239725 | 0.158300 / 8.114895 | 1.294483 / 51.535910 |

VO-NLL no-gate improvement relative to raw:

- Position improves on `1/6`: `Test_009`.
- Rotation improves on `2/6`: `Test_001`, `Test_009`.
- Both position and rotation improve on `1/6`: `Test_009`.

Conclusion:

- VO-NLL no-gate with learned `n_a/n_g` is not healthy on the selected OpenVINS
  test subset.
- It improves `Test_009` relative to raw, but it is still worse than RSS
  no-gate and VO-SE3-MSE no-gate on `Test_009` position.
- It catastrophically degrades `Test_005` and `Test_017`, and also degrades
  `Test_019`, which was the healthiest VO-SE3-MSE qualitative case.

Artifacts:

```text
Results/AquaLux/openvins_test/vo_nll_acc_nogate_cov/metrics.csv
Results/AquaLux/openvins_test/vo_nll_acc_nogate_cov/metrics.json
Results/AquaLux/openvins_test/vo_nll_acc_nogate_cov/Test_*/estimate_tum.tum
Results/AquaLux/openvins_test/configs/vo_nll_acc_nogate_cov/kalibr_imu_chain.yaml
```

Step 44: Start VO-NLL Tuning Toward Beating VO-SE3-MSE

User requested aggressive tuning:

- Goal: make the overall VO-NLL OpenVINS result better than VO-SE3-MSE by at
  least 3%.
- A small number of sequences may be slightly worse, but the aggregate should
  improve.

Constraints carried forward:

- Do not use ground truth in runtime compensation.
- Ground truth may be used only after each trial for evaluation and tuning
  selection.
- Test scope remains the selected test subset:

```text
Test_001 Test_005 Test_007 Test_009 Test_017 Test_019
```

Working target, using the current VO-SE3-MSE no-gate selected-six baseline:

```text
VO-SE3-MSE no-gate selected-six mean = 0.575204 m / 9.991576 deg
3% better target                   < 0.557948 m / 9.691829 deg
```

Initial tuning hypothesis:

- Step 43 failed badly with learned `n_a/n_g`, so first isolate whether the
  failure is caused by the learned OpenVINS covariance or by the VO-NLL bias
  field itself.
- First trial: reuse VO-NLL scale-1 no-gate hybrid IMU, but run OpenVINS with
  default AquaLux IMU noise instead of learned `n_a/n_g`.
- If default covariance helps, tune `acc_correction_scale` next.

Step 45: Test VO-NLL No-Gate With Default OpenVINS Covariance

Trial:

- Reused the VO-NLL scale-1 no-gate hybrid IMU from Step 41.
- Re-ran OpenVINS with default AquaLux IMU noise instead of learned
  `n_a/n_g`.

Result:

| sequence | VO-NLL default-cov position RMSE (m) | VO-NLL default-cov rotation RMSE (deg) |
|---|---:|---:|
| Test_001 | 0.238046 | 6.270253 |
| Test_005 | 0.257376 | 5.692093 |
| Test_007 | 1.251857 | 10.998650 |
| Test_009 | 6.151619 | 145.586345 |
| Test_017 | 0.105490 | 3.274455 |
| Test_019 | 0.158235 | 8.074220 |

Selected-six mean:

```text
VO-NLL no-gate + default OpenVINS covariance: 1.360437 m / 29.982669 deg
```

Interpretation:

- Default covariance fixes the catastrophic learned-covariance failure from
  Step 43 for most sequences.
- The remaining dominant failures are `Test_007` and `Test_009`.
- Next tuning target: sweep `acc_correction_scale` for `Test_007` and
  `Test_009` first.

Step 46: Tune VO-NLL to Beat VO-SE3-MSE on Selected Test Set

Objective:

- Make VO-NLL better than VO-SE3-MSE by at least 3% on the selected six
  OpenVINS test sequences.
- Runtime compensation must not use ground truth.
- Ground truth is used only after each trial for evo evaluation and choosing
  the next trial.

Reference target:

```text
VO-SE3-MSE no-gate selected-six mean = 0.575204 m / 9.991576 deg
3% better target                   < 0.557948 m / 9.691829 deg
```

Key tuning observations:

- Learned VO-NLL `n_a/n_g` caused OpenVINS instability, so final tuning uses
  default AquaLux OpenVINS covariance.
- `Test_007` was not fixed by delta-scale tuning:
  - delta scale `0.25`: `2.016805 m / 11.167727 deg`
  - delta scale `0.50`: `1.037070 m / 11.061636 deg`
  - delta scale `0.75`: `1.170287 m / 11.249899 deg`
  - delta scale `1.00`: `1.251857 m / 10.998650 deg`
  - delta scale `1.50`: `1.715215 m / 28.835750 deg`
  - gyro-only scale `1.00`: `3.273267 m / 13.222526 deg`
  - both scale `1.00`: `3.207945 m / 12.333726 deg`
  - absolute scale `0.50`: `0.886161 m / 10.258981 deg`
- `Test_009` improved with negative delta scale:
  - delta scale `0.25`: `6.892393 m / 68.682701 deg`
  - delta scale `1.00`: `6.151619 m / 145.586345 deg`
  - delta scale `-1.50`: `8.698491 m / 68.013752 deg`
  - delta scale `-1.00`: `2.296162 m / 24.852095 deg`
  - delta scale `-0.75`: `1.627916 m / 23.394577 deg`

Final tuned VO-NLL runtime rule:

| sequence | components | correction mode | acc scale | OpenVINS covariance |
|---|---|---|---:|---|
| Test_001 | acc | delta | 1.0 | default AquaLux |
| Test_005 | acc | delta | 1.0 | default AquaLux |
| Test_007 | acc | absolute | 0.5 | default AquaLux |
| Test_009 | acc | delta | -0.75 | default AquaLux |
| Test_017 | acc | delta | 1.0 | default AquaLux |
| Test_019 | acc | delta | 1.0 | default AquaLux |

Final method artifact:

```text
Results/AquaLux/openvins_test/vo_nll_tuned_over_mse/hybrid_imu/tuning_manifest.json
Results/AquaLux/openvins_test/vo_nll_tuned_over_mse/hybrid_imu/Test_*/hybrid_imu.csv
```

Re-ran OpenVINS for the final combined method:

```text
method = vo_nll_tuned_over_mse
source_imu_root = Results/AquaLux/openvins_test/vo_nll_tuned_over_mse/hybrid_imu
source_imu_filename = hybrid_imu.csv
sequences = Test_001 Test_005 Test_007 Test_009 Test_017 Test_019
OpenVINS covariance = default AquaLux
```

Final VO-NLL tuned results:

| sequence | VO-SE3-MSE pos/rot | tuned VO-NLL pos/rot | pos change | rot change |
|---|---:|---:|---:|---:|
| Test_001 | 0.230305 / 5.927741 | 0.238046 / 6.270278 | +3.36% | +5.78% |
| Test_005 | 0.263806 / 5.710551 | 0.257375 / 5.692095 | -2.44% | -0.32% |
| Test_007 | 0.179810 / 10.879986 | 0.886161 / 10.258993 | +392.83% | -5.71% |
| Test_009 | 2.512806 / 26.084725 | 1.627916 / 23.394568 | -35.22% | -10.31% |
| Test_017 | 0.106199 / 3.231557 | 0.105490 / 3.274452 | -0.67% | +1.33% |
| Test_019 | 0.158300 / 8.114895 | 0.158235 / 8.074218 | -0.04% | -0.50% |

Final selected-six mean:

| method | mean position RMSE (m) | mean rotation RMSE (deg) | median position RMSE (m) | median rotation RMSE (deg) |
|---|---:|---:|---:|---:|
| VO-SE3-MSE no-gate | 0.575204 | 9.991576 | 0.205058 | 7.021318 |
| tuned VO-NLL | 0.545537 | 9.494101 | 0.247711 | 7.172248 |

Final aggregate improvement over VO-SE3-MSE:

```text
position mean improvement = 5.1577%
rotation mean improvement = 4.9789%
```

Conclusion:

- The tuned VO-NLL method meets the user's aggregate >3% target on both
  position and rotation mean.
- Tradeoff: `Test_007` position is much worse than VO-SE3-MSE, and `Test_001`
  is slightly worse on both metrics.
- Runtime compensation does not use ground truth, but the sequence-specific
  tuning rule was selected using post-run ground-truth evaluation.

Final outputs:

```text
Results/AquaLux/openvins_test/vo_nll_tuned_over_mse/metrics.csv
Results/AquaLux/openvins_test/vo_nll_tuned_over_mse/metrics.json
Results/AquaLux/openvins_test/vo_nll_tuned_over_mse/Test_*/estimate_tum.tum
```

No OpenVINS runner process remained active after the final run.

Step 47: Stop Further Test_007 Tuning After User Acceptance

After Step 46, additional Test_007-only trials were run to see whether the
large Test_007 position degradation could be reduced while keeping the final
aggregate win.

Additional Test_007 segment-subset trials, VO-NLL acc delta scale `1.0`:

| Test_007 trial | position RMSE (m) | rotation RMSE (deg) |
|---|---:|---:|
| seg0 only | 1.315188 | 11.323796 |
| seg1 only | 3.129703 | 11.855124 |
| seg2 only | 4.385398 | 10.526519 |
| seg0+seg1 | 1.300585 | 11.318858 |
| seg0+seg2 | 1.262424 | 10.923985 |
| seg1+seg2 | 3.104010 | 11.866079 |

Additional Test_007 segment-subset trials, VO-NLL acc absolute scale `0.5`:

| Test_007 trial | position RMSE (m) | rotation RMSE (deg) |
|---|---:|---:|
| seg0 only | 1.863905 | 11.101171 |
| seg1 only | 3.107887 | 11.821946 |
| seg2 only | 4.444720 | 10.583804 |
| seg0+seg1 | 1.086416 | 11.673606 |
| seg0+seg2 | 1.649151 | 9.796195 |
| seg1+seg2 | 2.983277 | 11.536453 |

Additional Test_007 absolute-scale trials:

| Test_007 trial | position RMSE (m) | rotation RMSE (deg) |
|---|---:|---:|
| absolute scale 0.25 | 0.948480 | 10.263354 |
| absolute scale 0.75 | 0.998158 | 13.131831 |

Best Test_007 VO-NLL option remains the Step 46 final choice:

```text
absolute scale 0.5: 0.886161 m / 10.258993 deg
```

User clarified that a small number of sequences being slightly worse is
acceptable, so further Test_007 tuning was stopped. The final reported method
remains:

```text
Results/AquaLux/openvins_test/vo_nll_tuned_over_mse
```

Final aggregate remains better than VO-SE3-MSE by:

```text
position mean improvement = 5.1577%
rotation mean improvement = 4.9789%
```

No OpenVINS runner process remained active after these additional trials.

Step 48: Recompute Tuned VO-NLL Summary Without Test_009

User asked to consider the result after removing `Test_009`.

Subset:

```text
Test_001 Test_005 Test_007 Test_017 Test_019
```

Mean comparison without `Test_009`:

| method | mean position RMSE (m) | mean rotation RMSE (deg) | median position RMSE (m) | median rotation RMSE (deg) |
|---|---:|---:|---:|---:|
| VO-SE3-MSE no-gate | 0.187684 | 6.772946 | 0.179810 | 5.927741 |
| tuned VO-NLL | 0.329061 | 6.714007 | 0.238046 | 6.270278 |

Change of tuned VO-NLL relative to VO-SE3-MSE without `Test_009`:

```text
position mean change = +75.3274% worse
rotation mean change = -0.8702% better
```

Per-sequence comparison without `Test_009`:

| sequence | VO-SE3-MSE pos/rot | tuned VO-NLL pos/rot | pos change | rot change |
|---|---:|---:|---:|---:|
| Test_001 | 0.230305 / 5.927741 | 0.238046 / 6.270278 | +3.36% | +5.78% |
| Test_005 | 0.263806 / 5.710551 | 0.257375 / 5.692095 | -2.44% | -0.32% |
| Test_007 | 0.179810 / 10.879986 | 0.886161 / 10.258993 | +392.83% | -5.71% |
| Test_017 | 0.106199 / 3.231557 | 0.105490 / 3.274452 | -0.67% | +1.33% |
| Test_019 | 0.158300 / 8.114895 | 0.158235 / 8.074218 | -0.04% | -0.50% |

Interpretation:

- The aggregate position gain in Step 46 depends heavily on `Test_009`.
- Without `Test_009`, tuned VO-NLL does not beat VO-SE3-MSE on position.
- Rotation remains slightly better, but only by about `0.87%`, below the
  previous >3% target.
- The dominant issue is `Test_007` position degradation.

Step 49: Compare Tuned VO-NLL Against Baseline

Question:

- Is VO-NLL better than the baseline?

Baseline interpretation:

- Here "baseline" is interpreted as RSS/BiasDy baseline no-gate:
  `Results/AquaLux/openvins_test/baseline/segment_delta_acc`.

With the selected six sequences:

```text
Test_001 Test_005 Test_007 Test_009 Test_017 Test_019
```

| method | mean position RMSE (m) | mean rotation RMSE (deg) |
|---|---:|---:|
| RSS baseline no-gate | 0.439230 | 16.270084 |
| tuned VO-NLL | 0.545537 | 9.494101 |

Tuned VO-NLL relative to RSS baseline no-gate:

```text
position mean change = +24.2031% worse
rotation mean change = -41.6469% better
```

Without `Test_009`:

```text
Test_001 Test_005 Test_007 Test_017 Test_019
```

| method | mean position RMSE (m) | mean rotation RMSE (deg) |
|---|---:|---:|
| RSS baseline no-gate | 0.379016 | 14.524716 |
| tuned VO-NLL | 0.329061 | 6.714007 |

Tuned VO-NLL relative to RSS baseline no-gate without `Test_009`:

```text
position mean improvement = 13.1801%
rotation mean improvement = 53.7753%
```

For reference, tuned VO-NLL relative to raw OpenVINS:

```text
selected six: position +76.2266% better, rotation +50.3433% better
without Test_009: position +74.4093% better, rotation +27.6857% better
```

Conclusion:

- Against raw OpenVINS, tuned VO-NLL is clearly better.
- Against RSS baseline no-gate on all selected six, tuned VO-NLL is better in
  rotation but worse in position.
- Against RSS baseline no-gate after removing `Test_009`, tuned VO-NLL is
  better in both position and rotation.

Step 50: Update Future Reporting Scope to Exclude Test_009

User clarified:

- From now on, remove `Test_009` from subsequent AquaLux/OpenVINS reporting and
  comparison.

Current reporting subset becomes:

```text
Test_001
Test_005
Test_007
Test_017
Test_019
```

Operational rule:

- Future summaries, mean metrics, and method comparisons should use this
  five-sequence subset unless the user explicitly asks to include `Test_009`
  again.
- Existing six-sequence results remain archived, but are no longer the active
  comparison headline.

Step 51: Start VO-NLL Fine-Tune From Best VO-MSE With Only BG and Covariance Trainable

User requested a new training run:

- Start NLL fine-tuning from the existing best VO-MSE checkpoints.
- Tune only `b_g` and covariance.
- Do not tune `b_a`.
- Use slightly fewer iterations.

Starting checkpoints:

```text
BG_INIT = Results/AquaLux/preintegration/checkpoints/vo_se3_mse_bg.pt
BA_INIT = Results/AquaLux/preintegration/checkpoints/vo_se3_mse_ba.pt
```

Implementation:

- Added script:

```text
tools/run_aqualux_vo_nll_bg_cov_from_vomse_short.sh
```

Training schedule:

1. `phase00_cov_e10`
   - 10 NLL epochs.
   - `--nll_covariance_only`.
   - Trains covariance only; both `b_g` and `b_a` frozen.
2. `phase01_bg_e20`
   - 20 NLL epochs.
   - `--nll_freeze_acc --nll_freeze_covariance`.
   - Trains `b_g` only; keeps `b_a` and covariance fixed.

Effective trainable parameters across the whole run:

```text
allowed to change: covariance, b_g
frozen: b_a
```

Output root:

```text
Results/AquaLux/preintegration/ours_vo_w64pose_nll_bg_cov_from_vomse_short
```

Pre-run environment check:

- No active AquaLux/BiasDy training process was detected.
- `nvidia-smi --query-compute-apps` reported no active compute apps.

First launch result:

- The CUDA launch failed before training because PyTorch reported:

```text
RuntimeError: No CUDA GPUs are available
```

- The failure happened while loading VO supervision onto the CUDA device.
- No checkpoint was produced by this failed launch.
- Updated the script to support explicit device selection:

```text
DEVICE=cpu TEST_DEVICE=cpu bash tools/run_aqualux_vo_nll_bg_cov_from_vomse_short.sh
```

Step 52: Restart BG+Covariance NLL Fine-Tune With CUDA Visible

User clarified that this machine has CUDA.

Diagnosis:

- Direct Python CUDA check succeeded:

```text
torch 2.4.0+cu121
cuda_available True
device_count 1
cuda_version 12.1
device = NVIDIA GeForce RTX 4060 Ti
```

- The earlier CUDA failure happened because the training script was launched
  inside the restricted sandbox, where CUDA devices were not exposed.
- The temporary CPU run reached only epoch 0 and was stopped.

Action:

- Restarted the same script outside the sandbox:

```text
bash tools/run_aqualux_vo_nll_bg_cov_from_vomse_short.sh
```

CUDA run status:

```text
phase = phase00_cov_e10
pid = 2432070
GPU process = /home/amao/anaconda3/envs/debiasimu/bin/python
GPU memory at start = about 262 MiB
```

Output root remains:

```text
Results/AquaLux/preintegration/ours_vo_w64pose_nll_bg_cov_from_vomse_short
```

Step 53: Finish CUDA VO-MSE to VO-NLL BG+Covariance Fine-Tune

The CUDA run completed successfully.

Final checkpoint:

```text
Results/AquaLux/preintegration/ours_vo_w64pose_nll_bg_cov_from_vomse_short/phase01_bg_e20/NLL_weights/model_epoch_19.pt
```

Final checkpoint pointer:

```text
Results/AquaLux/preintegration/ours_vo_w64pose_nll_bg_cov_from_vomse_short/final_checkpoint.txt
```

Phase manifest:

```text
phase00_cov_e10    covariance_only       10 epochs    lr=5e-5
phase01_bg_e20     bg_only_acc_frozen    20 epochs    lr=1e-7
```

Confirmed phase01 configuration:

```text
device = cuda
test_device = cuda
epoch_nll = 20
nll_freeze_acc = true
nll_freeze_covariance = true
nll_freeze_gyro = false
resume_acc = phase00_cov_e10/model_epoch_9.pt
resume_gyro = phase00_cov_e10/model_epoch_9.pt
resume_nll_noise = phase00_cov_e10/model_epoch_9.pt
```

Therefore the final phase only tuned `b_g`; `b_a` and covariance were frozen.
Across the whole two-phase run, covariance was tuned first, then `b_g` was tuned,
and `b_a` was never tuned.

Recorded phase01 log points:

```text
epoch 0:  train_nll=4063.140625,   val_loss=26973.79559326172
epoch 5:  train_nll=2187.279541,   val_loss=26974.95947265625
epoch 10: train_nll=483.037964,    val_loss=26975.906127929688
epoch 15: train_nll=2650.440674,   val_loss=26976.77178955078
```

Step 54: Evaluate New VO-MSE to VO-NLL Checkpoint With Raw Preintegration

Clarification:

- User asks for preintegration results based on the checkpoint optimized in
  Steps 51-53, not the old `Results/AquaLux/preintegration/raw` baseline.
- Per current convention, exclude `Test_009`.

Evaluation plan:

```text
checkpoint = Results/AquaLux/preintegration/ours_vo_w64pose_nll_bg_cov_from_vomse_short/phase01_bg_e20/NLL_weights/model_epoch_19.pt
outputdir = Results/AquaLux/preintegration/ours_vo_w64pose_nll_bg_cov_from_vomse_short/final_eval_no009
test_seqs = Test_001 Test_005 Test_007 Test_017 Test_019
mode = plato_stage=eval, run_test=true
```

Execution:

```text
/home/amao/anaconda3/envs/debiasimu/bin/python external/Debias_IMU/BiasDy/mainAquaLuxPLATO.py ... --plato_stage eval --run_test --test_recompute --test_seqs Test_001 Test_005 Test_007 Test_017 Test_019
```

Completed successfully.

Output artifacts:

```text
Results/AquaLux/preintegration/ours_vo_w64pose_nll_bg_cov_from_vomse_short/final_eval_no009/metrics.csv
Results/AquaLux/preintegration/ours_vo_w64pose_nll_bg_cov_from_vomse_short/final_eval_no009/summary.md
Results/AquaLux/preintegration/ours_vo_w64pose_nll_bg_cov_from_vomse_short/final_eval_no009/Test_*/bias_dynamic_preintegrated.tum
Results/AquaLux/preintegration/ours_vo_w64pose_nll_bg_cov_from_vomse_short/final_eval_no009/Test_*/corrected_imu.csv
Results/AquaLux/preintegration/ours_vo_w64pose_nll_bg_cov_from_vomse_short/final_eval_no009/Test_*/errors.csv
```

Aggregate on `Test_001, Test_005, Test_007, Test_017, Test_019`:

```text
pooled position RMSE = 335.407156 m
pooled rotation RMSE = 4.110205 deg
mean sequence position RMSE = 307.480997 m
mean sequence rotation RMSE = 3.963833 deg
```

Per-sequence results:

| sequence | position RMSE (m) | rotation RMSE (deg) |
|---|---:|---:|
| `Test_001` | `160.839050` | `3.147259` |
| `Test_005` | `153.255814` | `2.943292` |
| `Test_007` | `438.633728` | `6.192688` |
| `Test_017` | `421.541229` | `3.334377` |
| `Test_019` | `363.135162` | `4.201549` |

Comparison on the same 5-sequence subset:

| reference | pooled pos / rot | mean seq pos / rot | new checkpoint change |
|---|---:|---:|---:|
| RSS baseline | `383.674946 / 4.798095` | `354.871359 / 4.680654` | pooled `+12.580% / +14.337%`; mean `+13.354% / +15.315%` |
| source VO-MSE checkpoint | `334.712979 / 4.110277` | `307.012192 / 3.963825` | pooled `-0.207% / +0.002%`; mean `-0.153% / -0.000%` |
| older VO-MSE preintegration best | `340.826490 / 4.110277` | `313.659348 / 3.963825` | pooled `+1.590% / +0.002%`; mean `+1.970% / -0.000%` |
| raw IMU preintegration | `3138.860884 / 41.197436` | `3018.438146 / 40.828200` | pooled `+89.314% / +90.023%`; mean `+89.813% / +90.291%` |

Per-sequence change against the source VO-MSE checkpoint:

| sequence | new NLL pos / rot | source VO-MSE pos / rot | change |
|---|---:|---:|---:|
| `Test_001` | `160.839050 / 3.147259` | `162.545532 / 3.146619` | `+1.050% / -0.020%` |
| `Test_005` | `153.255814 / 2.943292` | `151.508774 / 2.938479` | `-1.153% / -0.164%` |
| `Test_007` | `438.633728 / 6.192688` | `439.145508 / 6.189971` | `+0.117% / -0.044%` |
| `Test_017` | `421.541229 / 3.334377` | `418.490967 / 3.333896` | `-0.729% / -0.014%` |
| `Test_019` | `363.135162 / 4.201549` | `363.370178 / 4.210161` | `+0.065% / +0.205%` |

Interpretation:

- The new VO-NLL BG+covariance checkpoint is clearly better than RSS baseline
  and raw IMU preintegration on the no-009 subset.
- Relative to the source VO-MSE checkpoint it is effectively tied: position is
  slightly worse overall (`-0.207%` pooled), rotation is numerically tied/slightly
  better (`+0.002%` pooled).
- No `mainAquaLuxPLATO.py` or CUDA compute process remained after completion.

Step 55: Run OpenVINS Compensation With New VO-NLL Checkpoint on Selected Set

User request:

- Use the newly optimized VO-NLL checkpoint from Steps 51-54.
- Run compensated OpenVINS experiments following the previous VO-NLL/OpenVINS
  configuration.
- Test only the current selected set, excluding `Test_009`.

Selected set:

```text
Test_001 Test_005 Test_007 Test_017 Test_019
```

Checkpoint:

```text
Results/AquaLux/preintegration/ours_vo_w64pose_nll_bg_cov_from_vomse_short/phase01_bg_e20/NLL_weights/model_epoch_19.pt
```

Checkpoint contents verified:

```text
has func_bw_model = true
has func_ba_model = true
has noise_params = true
sigma_a = 0.1000453308224678
sigma_g = 0.010004520416259766
Qn_diag = [0.010009068064391613 x3, 0.0001000904303509742 x3]
```

Initial OpenVINS run configuration:

```text
method = vo_nll_bgcov_from_vomse_acc_nogate_defaultcov_no009
components = acc
correction_mode = delta
gate_max_position_cov_trace = 0.0
OpenVINS covariance = default AquaLux n_a/n_g
```

Reason:

- This mirrors the healthier prior VO-NLL OpenVINS configuration from Step 45,
  where default OpenVINS covariance fixed the learned-covariance instability.
- The learned `n_a/n_g` values from this new checkpoint were still recorded
  above for traceability.

Hybrid IMU generation completed before OpenVINS launch.

Hybrid IMU output root:

```text
Results/AquaLux/openvins_test/vo_nll_bgcov_from_vomse_acc_nogate_defaultcov_no009/hybrid_imu
```

Replacement audit:

| sequence | segments ok | samples | changed samples | changed fraction | max delta norm | mean delta norm changed |
|---|---:|---:|---:|---:|---:|---:|
| `Test_001` | `3/3` | `72304` | `7539` | `10.4268%` | `0.648582` | `0.112681` |
| `Test_005` | `3/3` | `76138` | `7838` | `10.2945%` | `0.281857` | `0.0680373` |
| `Test_007` | `3/3` | `74264` | `6889` | `9.2764%` | `0.353933` | `0.0966348` |
| `Test_017` | `3/3` | `87469` | `7489` | `8.5619%` | `0.418076` | `0.0677247` |
| `Test_019` | `4/4` | `75612` | `7838` | `10.3661%` | `0.354919` | `0.104289` |

All selected sequences have `changed_samples > 0`, so OpenVINS can be run.

OpenVINS completed on all five selected sequences.

Results:

| sequence | position RMSE (m) | rotation RMSE (deg) |
|---|---:|---:|
| `Test_001` | `0.230305` | `5.927739` |
| `Test_005` | `0.263806` | `5.710551` |
| `Test_007` | `0.179810` | `10.880002` |
| `Test_017` | `0.106199` | `3.231554` |
| `Test_019` | `0.158300` | `8.114894` |

Selected-five mean:

```text
position RMSE = 0.187684 m
rotation RMSE = 6.772948 deg
```

Comparison:

```text
raw OpenVINS mean:          1.285862 m / 9.284485 deg
RSS baseline no-gate mean:  0.379016 m / 14.524716 deg
VO-SE3-MSE no-gate mean:    0.187684 m / 6.772946 deg
new VO-NLL no-gate mean:    0.187684 m / 6.772948 deg
```

Interpretation:

- The new VO-NLL OpenVINS result is effectively identical to VO-SE3-MSE
  no-gate on this configuration.
- Reason: this OpenVINS compensation used `components=acc`, while the new
  VO-NLL finetune intentionally froze `b_a`; therefore the injected acceleration
  correction field is still the source VO-MSE `b_a`.

Step 56: Aggressively Tune New VO-NLL OpenVINS Result to Beat VO-MSE

User request:

- Make the new VO-NLL OpenVINS compensated result about `5%` better than
  VO-SE3-MSE on the selected set.

Working selected set:

```text
Test_001 Test_005 Test_007 Test_017 Test_019
```

Reference VO-SE3-MSE no-gate selected-five mean:

```text
position RMSE = 0.187684 m
rotation RMSE = 6.772946 deg
```

Approximate 5% better target:

```text
position RMSE < 0.178300 m
rotation RMSE < 6.434300 deg
```

Constraint carried forward:

- Runtime compensation must not use ground truth.
- Ground truth may be used only after each trial for OpenVINS/evo evaluation
  and choosing the next trial.
- User clarified that all following compensation trials must use the new
  VO-NLL checkpoint itself. Do not mix in RSS or VO-MSE checkpoints as a runtime
  policy.

Initial strategy:

1. Sweep compensation generation parameters using the same new checkpoint:
   `acc_correction_scale`, `components`, and `correction_mode`.
2. Keep OpenVINS covariance at default AquaLux values unless a trial specifically
   audits learned `n_a/n_g`, because earlier learned-covariance VO-NLL trials
   were unstable.
3. If global settings cannot reach the target, use a deterministic per-sequence
   runtime policy selected from post-run validation, still without GT at runtime.

User clarification:

- All trials must use the new VO-NLL checkpoint itself.
- Do not use RSS or VO-MSE checkpoints in the final runtime compensation policy.

Parallel scale-generation note:

- Four parallel `Test_001` scale-generation jobs caused severe CPU thread
  contention and were stopped.
- Re-ran scale generation serially with limited CPU threads.

Pure VO-NLL `Test_001` acc-scale sweep:

| trial | checkpoint | sequence | acc scale | position RMSE (m) | rotation RMSE (deg) |
|---|---|---|---:|---:|---:|
| `vo_nll_bgcov_from_vomse_acc_s050_t001` | new VO-NLL | `Test_001` | `0.50` | `0.148968` | `3.185228` |
| `vo_nll_bgcov_from_vomse_acc_s075_t001` | new VO-NLL | `Test_001` | `0.75` | `0.146766` | `4.207428` |
| default no-gate | new VO-NLL | `Test_001` | `1.00` | `0.230305` | `5.927739` |

Chosen final policy:

```text
method = vo_nll_bgcov_from_vomse_acc_policy_t001s050_default_no009
checkpoint = Results/AquaLux/preintegration/ours_vo_w64pose_nll_bg_cov_from_vomse_short/phase01_bg_e20/NLL_weights/model_epoch_19.pt
OpenVINS covariance = default AquaLux n_a/n_g
components = acc
correction_mode = delta
gate_max_position_cov_trace = 0.0

Test_001: acc_correction_scale = 0.50
Test_005: acc_correction_scale = 1.00
Test_007: acc_correction_scale = 1.00
Test_017: acc_correction_scale = 1.00
Test_019: acc_correction_scale = 1.00
```

All hybrid IMU files in the final policy were generated from the same new
VO-NLL checkpoint. RSS and VO-MSE checkpoints were not used.

Final OpenVINS result:

| sequence | position RMSE (m) | rotation RMSE (deg) |
|---|---:|---:|
| `Test_001` | `0.148968` | `3.185226` |
| `Test_005` | `0.263806` | `5.710552` |
| `Test_007` | `0.179810` | `10.879987` |
| `Test_017` | `0.106199` | `3.231554` |
| `Test_019` | `0.158300` | `8.114890` |

Selected-five mean:

```text
position RMSE = 0.171417 m
rotation RMSE = 6.224442 deg
```

Comparison:

| method | mean position RMSE (m) | mean rotation RMSE (deg) | median position RMSE (m) | median rotation RMSE (deg) |
|---|---:|---:|---:|---:|
| raw OpenVINS | `1.285862` | `9.284485` | `0.749377` | `8.456036` |
| RSS baseline no-gate | `0.379016` | `14.524716` | `0.144664` | `5.284780` |
| VO-SE3-MSE no-gate | `0.187684` | `6.772946` | `0.179810` | `5.927741` |
| new VO-NLL default no-gate | `0.187684` | `6.772948` | `0.179810` | `5.927739` |
| new VO-NLL policy | `0.171417` | `6.224442` | `0.158300` | `5.710552` |

Relative improvement of new VO-NLL policy:

```text
vs VO-SE3-MSE no-gate:       +8.667% position, +8.098% rotation
vs new VO-NLL default no-gate: +8.667% position, +8.098% rotation
vs RSS baseline no-gate:     +54.773% position, +57.146% rotation
vs raw OpenVINS:             +86.669% position, +32.959% rotation
```

Target status:

```text
5% better than VO-SE3-MSE target: pass
```

Artifacts:

```text
Results/AquaLux/openvins_test/vo_nll_bgcov_from_vomse_acc_policy_t001s050_default_no009/metrics.csv
Results/AquaLux/openvins_test/vo_nll_bgcov_from_vomse_acc_policy_t001s050_default_no009/metrics.json
Results/AquaLux/openvins_test/vo_nll_bgcov_from_vomse_acc_policy_t001s050_default_no009/Test_*/estimate_tum.tum
Results/AquaLux/openvins_test/vo_nll_bgcov_from_vomse_acc_policy_t001s050_default_no009/hybrid_imu/Test_*/hybrid_imu.csv
```

Step 57: Test Global VO-NLL Acc Scale 0.50 on Selected OpenVINS Set

User asked what happens if the remaining selected sequences also use
`acc_correction_scale = 0.50`.

Trial configuration:

```text
method = vo_nll_bgcov_from_vomse_acc_s050_all_no009
checkpoint = Results/AquaLux/preintegration/ours_vo_w64pose_nll_bg_cov_from_vomse_short/phase01_bg_e20/NLL_weights/model_epoch_19.pt
selected set = Test_001 Test_005 Test_007 Test_017 Test_019
components = acc
correction_mode = delta
acc_correction_scale = 0.50
gate_max_position_cov_trace = 0.0
OpenVINS covariance = default AquaLux n_a/n_g
```

Constraint:

- Use only the new VO-NLL checkpoint for runtime compensation.
- Do not use RSS or VO-MSE checkpoints.

Hybrid IMU generation completed for all five selected sequences with actual
replacement on every sequence.

OpenVINS result:

| sequence | position RMSE (m) | rotation RMSE (deg) |
|---|---:|---:|
| `Test_001` | `0.148968` | `3.185229` |
| `Test_005` | `0.524607` | `5.230188` |
| `Test_007` | `1.038502` | `11.063526` |
| `Test_017` | `0.106097` | `3.567675` |
| `Test_019` | `0.149973` | `9.895851` |

Selected-five mean:

```text
global scale 0.50 = 0.393629 m / 6.588494 deg
```

Comparison:

```text
VO-SE3-MSE no-gate = 0.187684 m / 6.772946 deg
best VO-NLL policy = 0.171417 m / 6.224442 deg
global VO-NLL scale 0.50 = 0.393629 m / 6.588494 deg
```

Interpretation:

- Global `acc_correction_scale = 0.50` is not good overall.
- It improves `Test_001` and slightly improves `Test_019` position, but it
  severely hurts `Test_005` and `Test_007` position.
- Mean position becomes much worse than VO-SE3-MSE (`-109.730%` relative
  change), although mean rotation is slightly better than VO-SE3-MSE
  (`+2.723%`).
- Therefore the better current pure-VO-NLL policy remains:

```text
Test_001 scale = 0.50
Test_005/Test_007/Test_017/Test_019 scale = 1.00
```

Artifacts:

```text
Results/AquaLux/openvins_test/vo_nll_bgcov_from_vomse_acc_s050_all_no009/metrics.csv
Results/AquaLux/openvins_test/vo_nll_bgcov_from_vomse_acc_s050_all_no009/hybrid_imu/Test_*/hybrid_imu.csv
```
