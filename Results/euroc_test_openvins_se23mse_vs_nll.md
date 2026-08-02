# EuRoC Test OpenVINS: SE23-MSE vs NLL

Source: `/home/amao/workspace/formalPLATO/Results/openvins_imu/openvins_results_rigorous_rerun/summary.csv`.
Metric: posyaw-aligned ATE; position RMSE in meters, rotation RMSE in degrees; no scale alignment.

## Method Average

| method | sequences | mean pos RMSE (m) | mean rot RMSE (deg) | mean pos mean (m) | mean rot mean (deg) |
|---|---:|---:|---:|---:|---:|
| GT SE23-MSE | 5 | 0.183164 | 1.457060 | 0.162875 | 1.358670 |
| GT NLL | 5 | 0.293184 | 1.866796 | 0.264964 | 1.757732 |
| VO SE23-MSE | 5 | 0.168624 | 1.404773 | 0.153616 | 1.303652 |
| VO NLL | 5 | 0.288395 | 1.835488 | 0.255679 | 1.716133 |

## Per-Sequence Comparison

| supervision | sequence | SE23-MSE pos RMSE (m) | SE23-MSE rot RMSE (deg) | NLL pos RMSE (m) | NLL rot RMSE (deg) |
|---|---|---:|---:|---:|---:|
| GT | MH_02_easy | 0.157919 | 0.788429 | 0.145974 | 1.183845 |
| GT | MH_04_difficult | 0.575210 | 1.219781 | 1.145884 | 2.624526 |
| GT | V1_01_easy | 0.064931 | 0.900679 | 0.064434 | 0.945406 |
| GT | V1_03_difficult | 0.057819 | 3.013844 | 0.064880 | 3.185612 |
| GT | V2_02_medium | 0.059940 | 1.362569 | 0.044746 | 1.394591 |
| VO | MH_02_easy | 0.123606 | 1.037478 | 0.120336 | 1.072776 |
| VO | MH_04_difficult | 0.546147 | 1.050172 | 1.154104 | 2.952459 |
| VO | V1_01_easy | 0.054608 | 0.684295 | 0.047920 | 0.900199 |
| VO | V1_03_difficult | 0.062931 | 2.965150 | 0.064199 | 3.095521 |
| VO | V2_02_medium | 0.055829 | 1.286770 | 0.055415 | 1.156486 |

CSV: `/home/amao/workspace/formalPLATO/Results/euroc_test_openvins_se23mse_vs_nll_rigorous_rerun.csv`
