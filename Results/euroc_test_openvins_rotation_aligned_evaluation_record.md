# EuRoC OpenVINS Rotation-Aligned Evaluation Record

Date: 2026-05-22

## Purpose

This diagnostic evaluates position and rotation under the same global rotation
alignment. It is intended to complement the standard OpenVINS/evo position
alignment metrics, where position alignment and rotation residual can be
optimized under different objectives.

## Source Data

- OpenVINS result root:
  `/home/amao/workspace/formalPLATO/Results/openvins_imu/openvins_results_rigorous_rerun`
- EuRoC GT root:
  `/home/amao/workspace/PLATO/external/OpenVINS/src/open_vins/ov_data/euroc_mav`
- Estimated trajectory file used for each run:
  `<run-root>/<method>/<sequence>/odomimu.txt`
- Association threshold: `0.020 s`

## Test Sequence Durations

Durations are computed from the OpenVINS EuRoC GT files used by this evaluation:
`duration = last_timestamp - first_timestamp`.

| sequence | start time (s) | end time (s) | duration (s) | duration (min) | GT poses |
|---|---:|---:|---:|---:|---:|
| MH_02_easy | 1403636859.536670 | 1403637009.496670 | 149.960 | 2.499 | 29993 |
| MH_04_difficult | 1403638128.940100 | 1403638227.700100 | 98.760 | 1.646 | 19753 |
| V1_01_easy | 1403715273.262140 | 1403715417.962140 | 144.700 | 2.412 | 2895 |
| V1_03_difficult | 1403715888.379060 | 1403715993.034060 | 104.655 | 1.744 | 20932 |
| V2_02_medium | 1413393887.225760 | 1413394002.675760 | 115.450 | 1.924 | 23091 |
| **Total** | - | - | **613.525** | **10.225** | **96664** |

## Reproduction Command

```bash
/home/amao/anaconda3/envs/debiasimu/bin/python \
  plato/scripts/openvins_eval/evaluate_rotation_aligned_openvins.py \
  --run-root /home/amao/workspace/formalPLATO/Results/openvins_imu/openvins_results_rigorous_rerun
```

## Metric Definition

First solve one constant rotation correction from all matched poses:

```text
C = argmin_C sum_i || C R_est_i - R_gt_i ||_F,  C in SO(3)
```

Then apply the same correction to estimated positions and fit translation only:

```text
p_est_aligned_i = C p_est_i + t
t = mean_i(p_gt_i - C p_est_i)
```

Reported values:

- `rot_aligned_pos_rmse_m`: RMSE of `||p_est_aligned_i - p_gt_i||`.
- `rot_residual_rmse_deg`: RMSE of the angle of `R_gt_i^T C R_est_i`.
- `constant_rot_correction_deg`: angle magnitude of `C`.

This metric uses GT orientation to estimate a constant rotation correction, so
it should be reported as a diagnostic/complementary metric, not as a replacement
for the standard VIO ATE.

## Results

Full results are saved at:

- `/home/amao/workspace/formalPLATO/Results/openvins_imu/openvins_results_rigorous_rerun/rotation_aligned_metric.md`
- `/home/amao/workspace/formalPLATO/Results/openvins_imu/openvins_results_rigorous_rerun/rotation_aligned_metric.csv`
- `/home/amao/workspace/formalPLATO/Results/euroc_test_openvins_rotation_aligned_metric.md`
- `/home/amao/workspace/formalPLATO/Results/euroc_test_openvins_rotation_aligned_metric.csv`

### Key Sequences

Values are `rot_residual_rmse_deg / rot_aligned_pos_rmse_m`.

| sequence | GT SE23-MSE | GT NLL | VO SE23-MSE | VO NLL |
|---|---:|---:|---:|---:|
| V1_03_difficult | 0.874340 / 0.095875 | 0.903422 / 0.103367 | 0.779696 / 0.098422 | 0.827467 / 0.101462 |
| V2_02_medium | 1.260316 / 0.057579 | 1.234782 / 0.045075 | 1.201094 / 0.053219 | 1.037441 / 0.052559 |

### All-Test Average

Average over `MH_02_easy`, `MH_04_difficult`, `V1_01_easy`,
`V1_03_difficult`, and `V2_02_medium`.

| method | mean rot-aligned pos RMSE (m) | mean rot residual RMSE (deg) |
|---|---:|---:|
| GT SE23-MSE | 0.190740 | 0.953271 |
| GT NLL | 0.311433 | 0.985727 |
| VO SE23-MSE | 0.176129 | 0.876555 |
| VO NLL | 0.307772 | 0.960874 |

## Interpretation

For V1_03_difficult, the rotation residual drops from the standard aligned
metric's roughly 3 deg level to roughly 0.8-0.9 deg after estimating one constant
SO(3) correction. The position RMSE becomes roughly 0.10 m, which is higher than
the position-only aligned result. This confirms that the previous low position
error and high rotation error are caused by different alignment objectives, not
by a simple evaluator failure.

For V2_02_medium, the rotation-aligned position remains close to the standard
position-aligned result, while the rotation residual is around 1.0-1.3 deg.
