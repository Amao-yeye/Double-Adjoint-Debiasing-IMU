# Dataset Guide

Last updated: 2026-05-13

Local dataset root on this machine:

```text
/home/amao/datasets
```

PLATO currently expects to use these datasets:

- EuRoC MAV: `/home/amao/datasets/EuRoC`
- LIASuw: `/home/amao/datasets/LIASuw`
- corldata: `/home/amao/datasets/corldata`

Do not copy these datasets into the PLATO git repository. `PLATO/datasets/` is a placeholder for local symlinks, small manifests, or instructions only; large data should stay under `/home/amao/datasets` and should not be committed.

## Recommended Path Policy

Use absolute dataset paths in local run configs or pass them as CLI arguments. For reusable PLATO scripts, prefer a single configurable dataset root:

```bash
export PLATO_DATASET_ROOT=/home/amao/datasets
```

Scripts should derive dataset-specific paths from that root:

```text
${PLATO_DATASET_ROOT}/EuRoC
${PLATO_DATASET_ROOT}/LIASuw
${PLATO_DATASET_ROOT}/corldata
```

If a tool requires a path inside the PLATO tree, create a local symlink under `PLATO/datasets/` and keep it ignored by git:

```bash
ln -s /home/amao/datasets/EuRoC datasets/EuRoC
ln -s /home/amao/datasets/LIASuw datasets/LIASuw
ln -s /home/amao/datasets/corldata datasets/corldata
```

Do not hardcode old external-project paths such as `external/Debias_IMU/rss_debiasing_restart/Debias_IMU/data` unless the task specifically requires compatibility with existing Debias_IMU code.

## EuRoC

Root:

```text
/home/amao/datasets/EuRoC
```

Observed ASL-format sequence directories:

```text
MH_01_easy
MH_02_easy
MH_03_medium
MH_04_difficult
MH_05_difficult
V1_01_easy
V1_02_medium
V1_03_difficult
V2_01_easy
V2_02_medium
V2_03_difficult
```

Observed sequence layout:

```text
<sequence>/mav0/cam0/
<sequence>/mav0/cam1/
<sequence>/mav0/imu0/
<sequence>/mav0/state_groundtruth_estimate0/
<sequence>/mav0/body.yaml
```

Representative files:

```text
/home/amao/datasets/EuRoC/MH_01_easy/mav0/cam0/data.csv
/home/amao/datasets/EuRoC/MH_01_easy/mav0/cam1/data.csv
/home/amao/datasets/EuRoC/MH_01_easy/mav0/imu0/data.csv
/home/amao/datasets/EuRoC/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv
```

Observed rosbag directory:

```text
/home/amao/datasets/EuRoC/EurocRosbags
```

Observed rosbags:

```text
MH_01_easy.bag
MH_02_easy.bag
MH_03_medium.bag
MH_04_difficult.bag
MH_05_difficult.bag
V1_01_easy.bag
V1_02_medium.bag
V1_03_difficult.bag
V2_01_easy.bag
V2_02_medium.bag
V2_03_difficult.bag
```

Observed archive backup directory:

```text
/home/amao/datasets/EuRoC/zip copyback
```

PLATO usage:

- ORB-SLAM3 stereo VO can use ASL-format image folders and timestamps.
- Debias_IMU currently has EuRoC-oriented training code and expects EuRoC-like IMU and supervision data.
- OpenVINS has existing EuRoC config files under `external/OpenVINS/src/open_vins/config/euroc_mav`.

Important cautions:

- Some sequence directories include `__MACOSX` or `.DS_Store` artifacts. Scripts should ignore these.
- ORB-SLAM3 stereo VO output must be checked for camera/body frame conventions before being used as Debias_IMU supervision.
- OpenVINS rosbag evaluation should use `/home/amao/datasets/EuRoC/EurocRosbags` unless a task explicitly requests ASL-format input.

## LIASuw

Root:

```text
/home/amao/datasets/LIASuw
```

Observed split directories:

```text
LIASuw_Training
LIASuw_Testing
LIASuw_Unusable
```

Observed split manifest:

```text
/home/amao/datasets/LIASuw/LIASuw_split_manifest.csv
```

Observed date/session groups:

```text
0203
0205
0912event
```

Training bags observed:

```text
LIASuw_Training/0203/s01.bag
LIASuw_Training/0203/s02.bag
LIASuw_Training/0203/s03.bag
LIASuw_Training/0203/s05.bag
LIASuw_Training/0203/s06.bag
LIASuw_Training/0203/s07.bag
LIASuw_Training/0203/s15.bag
LIASuw_Training/0203/s17.bag
LIASuw_Training/0203/s18.bag
LIASuw_Training/0203/s19.bag
LIASuw_Training/0203/s20.bag
LIASuw_Training/0203/s21.bag
LIASuw_Training/0203/s22.bag
LIASuw_Training/0203/s23.bag
LIASuw_Training/0203/s24.bag
LIASuw_Training/0205/s13.bag
LIASuw_Training/0205/s14.bag
LIASuw_Training/0205/s15.bag
LIASuw_Training/0205/s16.bag
LIASuw_Training/0205/s17.bag
LIASuw_Training/0912event/data/S06.bag
LIASuw_Training/0912event/data/S07.bag
LIASuw_Training/0912event/data/S09.bag
LIASuw_Training/0912event/data/S10.bag
LIASuw_Training/0912event/data/S11.bag
LIASuw_Training/0912event/data/S12.bag
LIASuw_Training/0912event/data/S13.bag
LIASuw_Training/0912event/data/S14.bag
LIASuw_Training/0912event/data/S15.bag
```

Testing bags observed:

```text
LIASuw_Testing/0203/s08.bag
LIASuw_Testing/0203/s09.bag
LIASuw_Testing/0203/s10.bag
LIASuw_Testing/0205/s11.bag
LIASuw_Testing/0205/s18.bag
LIASuw_Testing/0912event/data/S01.bag
LIASuw_Testing/0912event/data/S02.bag
LIASuw_Testing/0912event/data/S03.bag
LIASuw_Testing/0912event/data/S04.bag
LIASuw_Testing/0912event/data/S05.bag
```

Unusable bags observed:

```text
LIASuw_Unusable/0203/s04.bag
LIASuw_Unusable/0203/s11.bag
LIASuw_Unusable/0203/s12.bag
LIASuw_Unusable/0203/s13.bag
LIASuw_Unusable/0203/s14.bag
LIASuw_Unusable/0203/s16.bag
LIASuw_Unusable/0205/s01.bag
LIASuw_Unusable/0205/s02.bag
LIASuw_Unusable/0205/s03.bag
LIASuw_Unusable/0205/s04.bag
LIASuw_Unusable/0205/s05.bag
LIASuw_Unusable/0205/s06.bag
LIASuw_Unusable/0205/s07.bag
LIASuw_Unusable/0205/s08.bag
LIASuw_Unusable/0205/s09.bag
LIASuw_Unusable/0205/s10.bag
```

Observed calibration/config files:

```text
/home/amao/datasets/LIASuw/LIASuw_Training/esio.yaml
/home/amao/datasets/LIASuw/LIASuw_Training/event0_esio.yaml
/home/amao/datasets/LIASuw/LIASuw_Training/event1_esio.yaml
/home/amao/datasets/LIASuw/LIASuw_Training/0912event/Calibration files/cam-imu.yaml
/home/amao/datasets/LIASuw/LIASuw_Training/0912event/Calibration files/cam0(left)_intrinsic.yaml
/home/amao/datasets/LIASuw/LIASuw_Training/0912event/Calibration files/cam1right_intrinsic.yaml
/home/amao/datasets/LIASuw/LIASuw_Training/0912event/Calibration files/event0(left)_intrinsic.yaml
/home/amao/datasets/LIASuw/LIASuw_Training/0912event/Calibration files/event1(right)_intrinsic.yaml
```

Observed ground-truth / mocap-related areas:

```text
/home/amao/datasets/LIASuw/0203/GT
/home/amao/datasets/LIASuw/0203/未处理
/home/amao/datasets/LIASuw/0205/0205gt
/home/amao/datasets/LIASuw/0912event/contestGT
```

PLATO usage:

- ORB-SLAM3 stereo VO should use the visually favorable LIASuw training sequences first.
- Debias_IMU local tools already contain LIASuw pseudo-bias analysis scripts under `external/Debias_IMU/rss_debiasing_restart/Debias_IMU/plato_extensions/tools/liasuw_bias`; old repository-level `tools/*.py` paths are compatibility wrappers.
- Existing ORB-SLAM3 local tools mention LIASuw batch runs and visual-quality analysis under `external/ORB_SLAM3/tools`.
- PLATO should create new wrappers under `plato/scripts/` and avoid adding new tools directly to external repositories.

Important cautions:

- LIASuw uses mixed naming styles: lowercase `sXX.bag` in 0203/0205 folders and uppercase `SXX.bag` in `0912event/data`.
- Some paths contain spaces, for example `Calibration files`; scripts must use `pathlib.Path` or proper shell quoting.
- Some paths contain non-ASCII directory names, for example `0203/未处理`; scripts should avoid byte-string assumptions.
- The `LIASuw_Unusable` split should not be used for training or headline evaluation unless an experiment explicitly studies failure cases.
- The visually favorable versus visually degraded split must be documented per experiment. Do not assume the existing Training/Testing/Unusable split is identical to the final PLATO visual-quality split.

## corldata

Root:

```text
/home/amao/datasets/corldata
```

Observed session:

```text
/home/amao/datasets/corldata/0511_data
```

Observed bags:

```text
0511_data/S00.bag
0511_data/S01.bag
0511_data/S02.bag
0511_data/S03.bag
0511_data/S04.bag
0511_data/S05.bag
```

Observed S00 ROS topics:

```text
/davis_left/image_raw   sensor_msgs/Image
/davis_right/image_raw  sensor_msgs/Image
/davis_left/imu         sensor_msgs/Imu
/davis_left/events      dvs_msgs/EventArray
/davis_right/events     dvs_msgs/EventArray
/camera/color/image_raw sensor_msgs/Image
```

Observed ground-truth area:

```text
0511_data/gt/S01
0511_data/gt/S02
0511_data/gt/S03
0511_data/gt/S04
0511_data/gt/S05
```

`0511_data/gt/S00/Body_0.tum` is byte-identical to `0511_data/0511/0511/Take_000/Body_0.tum` and was used for the first S00 VO evaluation.

PLATO usage:

- Use DAVIS stereo image topics `/davis_left/image_raw` and `/davis_right/image_raw` for ORB-SLAM3 VO unless a task explicitly requests the RGB topic.
- The first S00 run uses `plato/configs/orbslam3/corldata_davis_liasuw_0205_baseline.yaml`, which reuses the LIASuw 0203/0205 DAVIS APS stereo calibration and sets `Camera.fps: 38`.
- Retained per-sequence VO and evaluation outputs live under `results/pseudo_gt/corldata/Sxx/`: raw ORB-SLAM3 output in `raw/`, exported images/timestamps in `sequence/`, fitted frame/time config in `alignment_config/`, converted GT in `gt_aligned/`, evo output in `evo/`, and current summary/aligned CSV in `evo_gt_aligned/`.
- The current user-approved generation/evaluation workflow is `plato/scripts/orbslam3_export/run_corldata_evo_pseudogt_workflow.py`; it generates ORB-SLAM3 stereo VO from the bag when `raw/estimate_f_seconds.tum` is missing, then runs the evo-GT-aligned evaluation. Aggregate outputs use the selected `--summary-prefix`, for example `results/pseudo_gt/corldata/S07_S23_evo_gt_aligned_report.md`, `.csv`, and `.json`.
- Canonical engineering workflow: `plato/docs/corldata_pseudogt_evaluation_workflow.md`.
- Newly added `S07`, `S13`, `S15`, `S20`, `S21`, `S22`, and `S23` were processed end-to-end on 2026-05-14. Best-looking new sequences by current metrics are `S15` and `S20` at about `0.026 m` position RMSE; `S22` is poor at `0.147 m` position RMSE with `0.677 m` max error and should not be accepted without further diagnostics.
- User decision on 2026-05-14: corldata sequences from `S13` onward have deliberately degraded visual conditions, with lights on for one interval and off for another. Treat `S13+` as testing data for validating network training results, not as default training pseudo-GT candidates.
- `plato/scripts/orbslam3_export/detect_corldata_light_segments.py` detects lights-on/off intervals for the testing sequences from exported DAVIS grayscale images and writes full-GT XY PNGs under `results/pseudo_gt/corldata/light_segments/`; black means lights on, gray means lights off.

Important cautions:

- Do not treat S00 as accepted pseudo-groundtruth yet. The evo translation APE uses SE(3) alignment without scale correction and reports RMSE about `0.0547 m` for both left-camera and camera-to-IMU/body trajectories, but direct rotation APE is still dominated by the fixed DAVIS/IMU-to-mocap `Body_0` frame mismatch. Calibration/time/frame conventions still need confirmation.
- Do not treat S01-S05 as accepted pseudo-groundtruth yet. The current evo-GT-aligned report uses automatically selected per-sequence time offsets plus a fitted per-sequence diagnostic camera/body transform, then evo `--align`; these are diagnostics until the corldata time and camera/IMU/Body_0 conventions are validated.
- Do not promote `S13+` into the training pseudo-GT pool by metric quality alone. Their role is testing under visual degradation; keep their VO/evaluation outputs for analysis and benchmarking learned models.
- `results/pseudo_gt/corldata/gt_quality/gt_quality_report.md` records a first GT quality diagnostic. It flags obvious local mocap/GT issues, especially S04 around relative `30.160 s` with a `1.1485 m` GT position step and S05 with many tiny timestamp steps/gaps around roughly `31-58 s`.
- Historical `run_corldata_batch.py`, `evaluate_corldata_masked_segments.py`, and `make_corldata_se3_fit_alignment.py` outputs were cleaned on 2026-05-14 after the `evo_gt_aligned` workflow superseded them. The scripts remain available if those diagnostics need to be regenerated.
- The current user-approved corldata pseudo-GT workflow estimates time offset and a per-sequence diagnostic camera/body transform, converts mocap `Body_0` GT into fitted camera-frame GT, then uses evo `--align` without scale correction to produce the final aligned trajectory. Draggable HTML pages are under `results/pseudo_gt/corldata/interactive_3d/` as `evo_gt_aligned` variants only; each page includes XY/YZ/XZ projection views with meter axis ticks and an automatic scale bar.
- The existing `external/ORB_SLAM3/tools/export_liasuw_bag.py` happens to match the corldata DAVIS topic names; this is reuse of a local exporter, not evidence that all LIASuw timing/frame assumptions apply to corldata.

## Integration Conventions For Future Scripts

Recommended CLI options for PLATO scripts:

```text
--dataset-root /home/amao/datasets
--dataset euroc|liasuw|corldata
--sequence <sequence_id>
--split train|test|unusable
```

Recommended output locations:

```text
results/orbslam3_export/<dataset>/<sequence_id>/
results/train_bias_dynamics/<run_name>/
results/openvins_eval/<run_name>/
```

Recommended config locations:

```text
plato/configs/orbslam3/
plato/configs/debias_imu/
plato/configs/openvins/
```

Before running long experiments, write a small manifest file that records:

- dataset root
- dataset name
- sequence IDs
- split
- raw input paths
- trajectory supervision path
- calibration path
- output directory
- command used

This keeps PLATO runs reproducible without copying large data into git.
