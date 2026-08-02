# OpenVINS Debiased IMU Exports

Date: 2026-05-21

## Status

This record describes the exported OpenVINS IMU batch generated from the earlier
fair no-gate GT branch and the VO branch. The current paper-facing GT mainline
uses strict SE23 right residuals; if OpenVINS comparisons are reported for that
mainline, regenerate or clearly label exports from the strict checkpoints listed
in:

```text
docs/README.md
docs/SE23_strict_definition_memory.md
```

## Output Location

Generated debiased EuRoC test-sequence IMU files for OpenVINS:

```text
/home/amao/workspace/formalPLATO/Results/openvins_imu
```

Each method/sequence uses EuRoC-style layout:

```text
Results/openvins_imu/{method}/{sequence}/mav0/imu0/data.csv
Results/openvins_imu/{method}/{sequence}/bias.csv
```

Manifest:

```text
Results/openvins_imu/manifest.json
```

Export script:

```text
tools/export_debiased_imu_for_openvins.py
```

## Methods

Exported methods:

```text
gt_mse
gt_nll
vo_mse
vo_nll
```

Test sequences:

```text
MH_02_easy
MH_04_difficult
V2_02_medium
V1_03_difficult
V1_01_easy
```

## Source Results

### gt_mse

```text
external/Debias_IMU/results/Euroc_PLATO_bg_damped145e3_ba_rate_tanh1e2_oldV1P10_eval_recheck
```

Fair no-gate GT-SE23-MSE.

### gt_nll

```text
external/Debias_IMU/results/Euroc_PLATO_sparse_gt_nll_logres_from_nogate_rate_tanh1e2_w64_b1000_lr1e6_anchor001_freezeA_eval_e40
```

Fair no-gate sparse GT NLL from no-gate GT-MSE, epoch 40.

### vo_mse

```text
external/Debias_IMU/results/Euroc_PLATO_vo32_5seq_mse_rel_hybrid_bg145_ba_rate_tanh1e2_ft_lr1e5_e11_eval_best
```

VO32 relative MSE.

### vo_nll

```text
external/Debias_IMU/results/Euroc_PLATO_vo32_sparse_nll_from_vomse_lr1e6_anchor001_freezeA_eval_e0
```

VO32 sparse NLL epoch 0, selected because AOE, velocity RMSE, and APE are all
better than VO32-MSE.

## Export Rule

For each sequence:

```text
gyro_corrected = gyro_raw - bg_pred
acc_corrected  = acc_raw  - ba_pred
```

The exported `data.csv` columns are:

```text
#timestamp [ns],
w_RS_S_x [rad s^-1],
w_RS_S_y [rad s^-1],
w_RS_S_z [rad s^-1],
a_RS_S_x [m s^-2],
a_RS_S_y [m s^-2],
a_RS_S_z [m s^-2]
```

The exported `bias.csv` columns are:

```text
#timestamp [ns],
b_g_x [rad s^-1],
b_g_y [rad s^-1],
b_g_z [rad s^-1],
b_a_x [m s^-2],
b_a_y [m s^-2],
b_a_z [m s^-2]
```

Timestamps are copied from the original EuRoC IMU CSV:

```text
external/Debias_IMU/data/EUROC/{sequence}/mav0/imu0/data.csv
```

Only the cached Debias_IMU evaluation interval is exported. The exact interval
and sample count for every method and sequence are recorded in
`Results/openvins_imu/manifest.json`.

## Validation

Checked after export:

```text
all 4 methods present
all 5 test sequences present per method
all data.csv files have samples matching manifest.json
all timestamps are strictly increasing
all timestamp_source values are original_euroc_imu_csv
```
