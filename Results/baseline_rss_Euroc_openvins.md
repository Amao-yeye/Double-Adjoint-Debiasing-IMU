# EuRoC OpenVINS RSS Baseline And NLL Results

Metric order: `AOE(deg) / APE(m)`. Lower is better.

## RSS Baseline

| sequence | RSS baseline |
|---|---:|
| MH_02 | 1.01 / 0.22 |
| MH_04 | 0.65 / 0.31 |
| V1_01 | 1.44 / 0.11 |
| V1_03 | 0.85 / 0.12 |
| V2_02 | 2.60 / 0.16 |
| ave | 1.31 / 0.18 |

## NLL, Fixed OpenVINS Camera Calibration

OpenVINS settings used for these NLL reruns:

```text
calib_cam_extrinsics=false
calib_cam_intrinsics=false
calib_cam_timeoffset=false
```

Standard posyaw-aligned ATE:

| sequence | GT NLL | VO NLL |
|---|---:|---:|
| MH_02 | 0.758 / 0.0838 | 0.696 / 0.0796 |
| MH_04 | 1.084 / 0.1490 | 1.322 / 0.2147 |
| V1_01 | 0.826 / 0.0396 | 0.825 / 0.0392 |
| V1_03 | 3.561 / 0.0654 | 3.546 / 0.0659 |
| V2_02 | 1.045 / 0.0493 | 1.059 / 0.0509 |
| ave | 1.455 / 0.0774 | 1.490 / 0.0901 |

Rotation-aligned AOE/APE diagnostic:

| sequence | GT NLL | VO NLL |
|---|---:|---:|
| MH_02 | 0.734 / 0.0834 | 0.680 / 0.0792 |
| MH_04 | 0.433 / 0.1915 | 0.799 / 0.2485 |
| V1_01 | 0.568 / 0.0435 | 0.565 / 0.0432 |
| V1_03 | 1.141 / 0.1108 | 1.147 / 0.1106 |
| V2_02 | 0.903 / 0.0478 | 0.919 / 0.0496 |
| ave | 0.756 / 0.0954 | 0.822 / 0.1062 |

Source run roots:

- `results/openvins_eval/euroc_nll_fixedcalib_openvins/summary.csv`
- `results/openvins_eval/euroc_nll_fixedcalib_openvins/rotation_aligned_metric.csv`
