# EuRoC OpenVINS MSE Conservative Noise vs NLL

This is a sensitivity/stress-test table, not the primary fair-comparison table. MSE methods are rerun with the matching NLL OpenVINS IMU white-noise density; bias random walk stays at EuRoC defaults.

Metric: posyaw-aligned ATE. Position RMSE is in meters; rotation RMSE is SO(3) geodesic angle in degrees. Timestamp association threshold is 0.02 s. No scale alignment is used.

## Method Average

| method | sequences | mean pos RMSE (m) | mean rot RMSE (deg) | mean pos mean (m) | mean rot mean (deg) |
|---|---:|---:|---:|---:|---:|
| gt_mse_conservative_noise | 5 | 0.293700 | 1.790088 | 0.261716 | 1.634106 |
| gt_nll | 5 | 0.293382 | 1.868540 | 0.265158 | 1.759503 |
| vo_mse_conservative_noise | 5 | 0.294692 | 1.834842 | 0.260035 | 1.718432 |
| vo_nll | 5 | 0.288634 | 1.836876 | 0.255903 | 1.717593 |

## Per-Sequence Results

| sequence | method | matched poses | pos RMSE (m) | rot RMSE (deg) | pos mean (m) | rot mean (deg) | pos max (m) | rot max (deg) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MH_02_easy | gt_mse_conservative_noise | 22130 | 0.168808 | 1.280824 | 0.153048 | 1.141025 | 0.382397 | 2.416867 |
| MH_02_easy | gt_nll | 22174 | 0.146016 | 1.184474 | 0.132688 | 1.070146 | 0.297899 | 2.203494 |
| MH_02_easy | vo_mse_conservative_noise | 22107 | 0.139455 | 1.149791 | 0.127701 | 1.025152 | 0.262050 | 2.203909 |
| MH_02_easy | vo_nll | 22162 | 0.120351 | 1.073582 | 0.108318 | 1.021515 | 0.252488 | 1.785664 |
| MH_04_difficult | gt_mse_conservative_noise | 15765 | 1.130166 | 2.442477 | 0.999340 | 2.116266 | 1.916156 | 4.508662 |
| MH_04_difficult | gt_nll | 15766 | 1.146583 | 2.628850 | 1.030413 | 2.516856 | 1.912606 | 4.023475 |
| MH_04_difficult | vo_mse_conservative_noise | 15773 | 1.176446 | 2.928324 | 1.025609 | 2.740464 | 2.303378 | 5.257903 |
| MH_04_difficult | vo_nll | 15804 | 1.154874 | 2.953059 | 1.015004 | 2.678048 | 2.181851 | 5.430331 |
| V1_01_easy | gt_mse_conservative_noise | 24234 | 0.050155 | 1.015266 | 0.046571 | 0.889751 | 0.102546 | 2.569055 |
| V1_01_easy | gt_nll | 24201 | 0.064414 | 0.946736 | 0.060781 | 0.854317 | 0.119264 | 2.643299 |
| V1_01_easy | vo_mse_conservative_noise | 24200 | 0.045754 | 0.867031 | 0.043026 | 0.760378 | 0.100900 | 2.548194 |
| V1_01_easy | vo_nll | 24217 | 0.047920 | 0.900209 | 0.044959 | 0.807318 | 0.112114 | 2.692711 |
| V1_03_difficult | gt_mse_conservative_noise | 19714 | 0.065297 | 3.083607 | 0.059599 | 3.004948 | 0.147573 | 4.030383 |
| V1_03_difficult | gt_nll | 19650 | 0.064892 | 3.187729 | 0.060183 | 3.118168 | 0.149436 | 4.217179 |
| V1_03_difficult | vo_mse_conservative_noise | 19677 | 0.064223 | 3.097210 | 0.060015 | 3.037536 | 0.162733 | 4.109754 |
| V1_03_difficult | vo_nll | 19684 | 0.064224 | 3.099679 | 0.060013 | 3.040161 | 0.162775 | 4.109770 |
| V2_02_medium | gt_mse_conservative_noise | 22356 | 0.054074 | 1.128264 | 0.050021 | 1.018537 | 0.131846 | 1.990808 |
| V2_02_medium | gt_nll | 22390 | 0.045004 | 1.394913 | 0.041726 | 1.238028 | 0.135518 | 2.635595 |
| V2_02_medium | vo_mse_conservative_noise | 22424 | 0.047584 | 1.131855 | 0.043824 | 1.028628 | 0.126798 | 1.953212 |
| V2_02_medium | vo_nll | 22399 | 0.055802 | 1.157849 | 0.051222 | 1.040923 | 0.136255 | 2.052957 |

MSE conservative result root: `/home/amao/workspace/formalPLATO/Results/openvins_imu/openvins_results_mse_conservative_noise`
NLL reference result root: `/home/amao/workspace/formalPLATO/Results/openvins_imu/openvins_results`
CSV table: `/home/amao/workspace/formalPLATO/Results/euroc_openvins_mse_conservative_noise_vs_nll.csv`
Method-average CSV: `/home/amao/workspace/formalPLATO/Results/euroc_openvins_mse_conservative_noise_vs_nll_method_average.csv`
