# Mocap-Camera Alignment Notes

This document records the current PLATO understanding of LIASuw mocap, ORB-SLAM3, and camera/IMU frame alignment. It is intended to prevent future agents from silently treating different pose frames as the same frame.

Important rule: different datasets, sessions, rigs, and even different exported GT products may define the left-camera frame, IMU/body frame, and motion-capture rigid-body frame differently. Do not reuse a fixed rotation estimated on one dataset or sequence unless it has been validated for the same physical rig and mocap body definition.

## TUM Format Check

For LIASuw `0912s15`, the mocap groundtruth used by PLATO is:

`/home/amao/datasets/LIASuw/LIASuw_Training/0912event/data/GT/15/mc.tum`

The file follows the standard TUM pose order:

```text
timestamp tx ty tz qx qy qz qw
```

Evidence:

- The source CSV header is `Body_0_timestamp, Body_0_Pos_X(mm), Body_0_Pos_Y(mm), Body_0_Pos_Z(mm), Body_0_Qua_X, Body_0_Qua_Y, Body_0_Qua_Z, Body_0_Qua_R`.
- The local converter `.../GT/15/capture2evo.py` converts `mc.csv` to `mc.tum` by writing `timestamp tx ty tz qx qy qz qw`.
- `capture2evo.py` converts mocap position from millimeters to meters.

The TUM column order is therefore correct. The remaining uncertainty is the semantic pose frame of mocap `Body_0`, not the column order.

## Current Diagnosis

For `results/pseudo_gt/LIASuw/0912s15/`, PLATO now has a diagnostic sweep:

`results/pseudo_gt/LIASuw/0912s15/diagnostics/frame_time/`

The diagnostic script is:

`plato/scripts/orbslam3_export/diagnose_liasuw_frame_time_alignment.py`

Current best diagnostic result:

- Time offset: `est_time - 1.665 s`.
- GT quaternion mode: `as_is`; inverting the mocap quaternion did not improve the result.
- ORB-SLAM3 pose frame mode: `camera`; applying `cam0 -> imu0` changed the fitted residual by a negligible amount.
- Constant frame rotation fitted between ORB-SLAM3 left camera and mocap `Body_0`.
- Orientation RMSE after fitting the fixed frame rotation: about `3.88 deg`.
- Orientation RMSE before fitting the fixed frame rotation: about `147.09 deg`.

Interpretation: the large orientation offset is dominated by a missing fixed transform between ORB-SLAM3's left-camera pose frame and the mocap `Body_0` pose frame. It is not caused by TUM column order, quaternion inversion, or a major timestamp offset error.

## Additional 0912s06 Test

PLATO also tested LIASuw `0912s06`, selected because it has longer motion and stronger rotation excitation than `0912s15`. The current `0912s06` PLATO result directory has been overwritten with the first 80 seconds only:

- Rosbag: `/home/amao/datasets/LIASuw/LIASuw_Training/0912event/data/S06.bag`
- GT: `/home/amao/datasets/LIASuw/LIASuw_Training/0912event/data/GT/06/mc.tum`
- Result directory: `results/pseudo_gt/LIASuw/0912s06/`
- Exported stereo pairs before truncation: `3111`
- Kept stereo pairs after first-80-second truncation: `1600`
- Kept IMU measurements: `79804`
- Actual kept image timestamp span: about `79.950 s`
- Best diagnostic time offset: `est_time - 1.475 s`
- GT quaternion mode: `as_is`
- ORB pose frame mode: `camera`
- Frame-aligned position RMSE: about `0.042 m`
- Frame-aligned orientation RMSE: about `3.56 deg`
- Frame-aligned orientation max error: about `9.88 deg`
- Raw orientation RMSE before fixed frame rotation: about `148.92 deg`

The fixed frame rotation estimated for the first-80-second `0912s06` run should still be treated as sequence/session-specific until a real camera/mocap body extrinsic is found or validated across multiple sequences.

The ORB-SLAM3 log for `0912s06` still includes multiple `Fail to track local map!` and `Relocalized!!` messages. The truncated sequence has promising trajectory metrics after alignment, but it is not yet a clean pseudo-groundtruth candidate without tracking-quality review and better frame calibration.

## Known Calibration Files

Camera-IMU calibration for the current 0912 event data is available at:

`/home/amao/datasets/LIASuw/LIASuw_Training/0912event/Calibration files/cam-imu.yaml`

This file provides:

- `T_ci`: `imu0 to cam0`
- `T_ic`: `cam0 to imu0`
- `T_ci`: `imu0 to cam1`
- `T_ic`: `cam1 to imu0`
- camera timestamp shifts relative to IMU
- stereo baseline

The file does not define the transform from mocap `Body_0` to `cam0` or `imu0`.

The ESVIO config:

`/home/amao/datasets/LIASuw/LIASuw_Training/0912event/esvio_uw/esvio.yaml`

contains body-camera parameters for ESVIO, but this still needs careful interpretation before treating the ESVIO body frame as the mocap `Body_0` frame.

## Needed Transform

The missing calibrated extrinsic is one of:

```text
T_mocap_body_cam0
T_mocap_body_imu0
T_cam0_mocap_body
T_imu0_mocap_body
```

The exact direction is less important than documenting it unambiguously. Once available, PLATO should convert ORB-SLAM3's left-camera pose into the same physical frame used by mocap `Body_0` before computing orientation error or exporting pseudo-groundtruth.

## Practical Calibration Strategy

If a true dataset-provided `Body_0` extrinsic cannot be found, estimate it from synchronized motion:

1. Use ORB-SLAM3 stereo to estimate `T_world_cam0(t)`.
2. Use mocap to read `T_mocap_world_body0(t)` or the equivalent `Body_0` pose convention.
3. Time-associate the poses and sweep timestamp offset.
4. Compute relative rotations over pose pairs with sufficient rotation excitation.
5. Solve a hand-eye style problem:

```text
R_cam_rel * R_cam_to_body = R_cam_to_body * R_body_rel
```

6. Validate the estimated transform on a different sequence. Do not trust a transform that only fits one short sequence.

Important requirements:

- The motion must contain enough rotations around multiple axes.
- Pure translation or nearly planar/yaw-only motion gives a weak extrinsic estimate.
- Estimate on one sequence and validate on another sequence such as another 0912 training candidate.
- Keep using metric stereo scale without Sim(3) scale correction unless a specific dataset problem is identified.

## Current Recommendation

Use the current fitted fixed rotation only as a diagnostic and temporary alignment aid. Before using LIASuw ORB-SLAM3 trajectories as privileged supervision for IMU bias learning, either find the real mocap `Body_0` to `cam0`/`imu0` extrinsic or validate a hand-eye estimated transform across multiple LIASuw sequences.
