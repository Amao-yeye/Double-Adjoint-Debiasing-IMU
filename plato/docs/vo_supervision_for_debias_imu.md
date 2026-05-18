# VO Supervision For Debias_IMU

Date: 2026-05-09

## Correct VO Supervision

For Debias_IMU training, use stereo VO poses transformed to the IMU/body frame and then transformed into the mocap/GT world frame:

```text
T_Wv_I = T_Wv_C * T_C_I
T_Wm_I = T_Wm_Wv * T_Wv_I
```

Where:

- `T_Wv_C`: ORB-SLAM3 stereo left-camera pose in VO world.
- `T_C_I`: camera-to-IMU/body extrinsic from the LIASuw calibration file.
- `T_Wv_I`: IMU/body pose in VO world, saved as `vo_imu_traj.tum`.
- `T_Wm_Wv`: fixed VO-world to mocap/GT-world transform, saved in `calibration_result.yaml` as `T_WO`.
- `T_Wm_I`: corrected VO supervision pose for Debias_IMU.

The body frame is IMU/body `I`; the world frame must be mocap/GT world `Wm` so Debias_IMU's default gravity remains:

```text
g_Wm = [0, 0, -9.81]
```

## Files To Use

Per sequence, start from:

```text
results/pseudo_gt/LIASuw/<sequence>/debias_imu_body_gt_200hz/vo_imu_traj.tum
```

Then apply:

```text
results/pseudo_gt/LIASuw/<sequence>/debias_imu_body_gt_200hz/calibration_result.yaml
```

Specifically use `T_WO` from that YAML as `T_Wm_Wv`.

The PLATO runner does this automatically with:

```bash
/home/amao/anaconda3/envs/debiasimu/bin/python -u \
  plato/scripts/train_bias_dynamics/run_liasuw_debias_ad_pipeline.py \
  --trainval-supervision-source vo_imu_noninterp \
  --vo-supervision-world-frame gt
```

`--vo-supervision-world-frame gt` is the correct setting. Do not use raw `--vo-supervision-world-frame vo` for final training unless gravity is also transformed into VO world.

## Why This Is Necessary

`vo_imu_traj.tum` is already in the IMU/body frame, but its world frame is still VO world `Wv`. A frame check showed `Wv` and mocap/GT world `Wm` differ by about `146-180 deg` across the current LIASuw sequences. If raw VO-world poses are used directly with Debias_IMU's default gravity, the gravity direction is wrong by about `133-144 deg`.

Therefore the old full run:

```text
results/debias_imu/LIASuw_phys_delta_vo_pseudogt_e1800_w16pose_b64/
```

is an invalid old-frame diagnostic, not a corrected VO-supervision result.

## Verification Commands

Compare raw VO world with GT world:

```bash
/home/amao/anaconda3/envs/debiasimu/bin/python \
  plato/scripts/train_bias_dynamics/compare_vo_gt_world_frames.py
```

Verify cached corrected VO supervision poses:

```bash
/home/amao/anaconda3/envs/debiasimu/bin/python \
  plato/scripts/train_bias_dynamics/verify_liasuw_vo_supervision_frames.py \
  --cache-dir results/debias_imu/cache/<corrected_vo_cache_dir>
```

The corrected cache should report `world_frame=gt` and residuals comparable to the original VO/mocap calibration residuals.
