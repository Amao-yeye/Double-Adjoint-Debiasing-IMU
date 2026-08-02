# Rotation-Aligned OpenVINS Metric

Source: `/home/amao/workspace/formalPLATO/Results/openvins_imu/openvins_results_rigorous_rerun`
GT root: `/home/amao/workspace/PLATO/external/OpenVINS/src/open_vins/ov_data/euroc_mav`
Association threshold: `0.020 s`

Definition: solve one constant SO(3) correction from GT/estimate orientations, apply it to positions,
then fit translation only and report position RMSE. Rotation residual is evaluated after the same
constant SO(3) correction.

## Per-Sequence

| method | sequence | matches | rot-aligned pos RMSE (m) | rot residual RMSE (deg) | constant rot corr (deg) |
|---|---|---:|---:|---:|---:|
| GT SE23-MSE | MH_02_easy | 22259 | 0.156894 | 0.760786 | 178.719137 |
| GT SE23-MSE | MH_04_difficult | 15856 | 0.576183 | 1.197170 | 39.450843 |
| GT SE23-MSE | V1_01_easy | 24314 | 0.067167 | 0.673742 | 13.788386 |
| GT SE23-MSE | V1_03_difficult | 19792 | 0.095875 | 0.874340 | 8.953054 |
| GT SE23-MSE | V2_02_medium | 22397 | 0.057579 | 1.260316 | 178.174966 |
| GT NLL | MH_02_easy | 22272 | 0.151481 | 1.005678 | 178.951888 |
| GT NLL | MH_04_difficult | 15850 | 1.193091 | 0.893725 | 39.768801 |
| GT NLL | V1_01_easy | 24317 | 0.064152 | 0.891030 | 12.908565 |
| GT NLL | V1_03_difficult | 19764 | 0.103367 | 0.903422 | 8.203771 |
| GT NLL | V2_02_medium | 22377 | 0.045075 | 1.234782 | 177.426822 |
| VO SE23-MSE | MH_02_easy | 22252 | 0.127711 | 0.796532 | 179.813959 |
| VO SE23-MSE | MH_04_difficult | 15855 | 0.546896 | 1.039646 | 39.047046 |
| VO SE23-MSE | V1_01_easy | 24369 | 0.054396 | 0.565808 | 13.707079 |
| VO SE23-MSE | V1_03_difficult | 19786 | 0.098422 | 0.779696 | 8.559558 |
| VO SE23-MSE | V2_02_medium | 22389 | 0.053219 | 1.201094 | 177.938329 |
| VO NLL | MH_02_easy | 22245 | 0.128765 | 0.801128 | 178.978010 |
| VO NLL | MH_04_difficult | 15867 | 1.208111 | 1.322547 | 41.112276 |
| VO NLL | V1_01_easy | 24333 | 0.047961 | 0.815787 | 12.996304 |
| VO NLL | V1_03_difficult | 19779 | 0.101462 | 0.827467 | 8.236575 |
| VO NLL | V2_02_medium | 22357 | 0.052559 | 1.037441 | 177.154380 |

## Method Average

| method | sequences | mean rot-aligned pos RMSE (m) | mean rot residual RMSE (deg) |
|---|---:|---:|---:|
| GT SE23-MSE | 5 | 0.190740 | 0.953271 |
| GT NLL | 5 | 0.311433 | 0.985727 |
| VO SE23-MSE | 5 | 0.176129 | 0.876555 |
| VO NLL | 5 | 0.307772 | 0.960874 |
