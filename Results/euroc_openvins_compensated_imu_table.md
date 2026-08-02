# EuRoC OpenVINS Compensated IMU Results

Metric: posyaw-aligned ATE. Position is RMSE in meters; rotation is SO(3) geodesic RMSE in degrees. Timestamp association threshold is 0.02 s. No scale alignment is used.

Note: V1_02_medium is not included because compensated IMU CSV files for V102 are not present under `openvins_imu` yet.

## Method Average

| method | sequences | mean pos RMSE (m) | mean rot RMSE (deg) | mean pos mean (m) | mean rot mean (deg) |
|---|---:|---:|---:|---:|---:|
| gt_mse | 5 | 0.183322 | 1.457765 | 0.162944 | 1.358966 |
| gt_nll | 5 | 0.293382 | 1.868540 | 0.265158 | 1.759503 |
| vo_mse | 5 | 0.168486 | 1.404600 | 0.153457 | 1.303609 |
| vo_nll | 5 | 0.288634 | 1.836876 | 0.255903 | 1.717593 |

## Per-Sequence Results

| sequence | method | matched poses | pos RMSE (m) | rot RMSE (deg) | pos mean (m) | rot mean (deg) | pos max (m) | rot max (deg) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MH_02_easy | gt_mse | 22139 | 0.158030 | 0.788921 | 0.131986 | 0.667609 | 0.449559 | 2.311474 |
| MH_02_easy | gt_nll | 22174 | 0.146016 | 1.184474 | 0.132688 | 1.070146 | 0.297899 | 2.203494 |
| MH_02_easy | vo_mse | 22123 | 0.123638 | 1.039708 | 0.118174 | 0.923572 | 0.188113 | 1.913539 |
| MH_02_easy | vo_nll | 22162 | 0.120351 | 1.073582 | 0.108318 | 1.021515 | 0.252488 | 1.785664 |
| MH_04_difficult | gt_mse | 15727 | 0.575396 | 1.219824 | 0.519109 | 1.118822 | 1.019802 | 3.129106 |
| MH_04_difficult | gt_nll | 15766 | 1.146583 | 2.628850 | 1.030413 | 2.516856 | 1.912606 | 4.023475 |
| MH_04_difficult | vo_mse | 15762 | 0.545312 | 1.047884 | 0.491189 | 0.902480 | 1.077686 | 3.316108 |
| MH_04_difficult | vo_nll | 15804 | 1.154874 | 2.953059 | 1.015004 | 2.678048 | 2.181851 | 5.430331 |
| V1_01_easy | gt_mse | 24243 | 0.064933 | 0.899349 | 0.056223 | 0.836200 | 0.157336 | 2.290344 |
| V1_01_easy | gt_nll | 24201 | 0.064414 | 0.946736 | 0.060781 | 0.854317 | 0.119264 | 2.643299 |
| V1_01_easy | vo_mse | 24192 | 0.054668 | 0.684244 | 0.050196 | 0.618888 | 0.124029 | 2.044219 |
| V1_01_easy | vo_nll | 24217 | 0.047920 | 0.900209 | 0.044959 | 0.807318 | 0.112114 | 2.692711 |
| V1_03_difficult | gt_mse | 19710 | 0.057819 | 3.013232 | 0.052297 | 2.956936 | 0.130549 | 3.936941 |
| V1_03_difficult | gt_nll | 19650 | 0.064892 | 3.187729 | 0.060183 | 3.118168 | 0.149436 | 4.217179 |
| V1_03_difficult | vo_mse | 19704 | 0.063006 | 2.965123 | 0.057062 | 2.931002 | 0.150662 | 3.800515 |
| V1_03_difficult | vo_nll | 19684 | 0.064224 | 3.099679 | 0.060013 | 3.040161 | 0.162775 | 4.109770 |
| V2_02_medium | gt_mse | 22419 | 0.060430 | 1.367497 | 0.055104 | 1.215264 | 0.168983 | 2.542696 |
| V2_02_medium | gt_nll | 22390 | 0.045004 | 1.394913 | 0.041726 | 1.238028 | 0.135518 | 2.635595 |
| V2_02_medium | vo_mse | 22334 | 0.055803 | 1.286043 | 0.050665 | 1.142101 | 0.146969 | 2.394941 |
| V2_02_medium | vo_nll | 22399 | 0.055802 | 1.157849 | 0.051222 | 1.040923 | 0.136255 | 2.052957 |

Source summary: `/home/amao/workspace/formalPLATO/Results/openvins_imu/openvins_results/summary.csv`
CSV table: `/home/amao/workspace/formalPLATO/Results/euroc_openvins_compensated_imu_table.csv`
Method-average CSV: `/home/amao/workspace/formalPLATO/Results/euroc_openvins_compensated_imu_method_average.csv`
