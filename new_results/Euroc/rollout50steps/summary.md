# Euroc 50-step Preintegration Drift

Divergence threshold: 1.000 m position error to GT at current time.

| role | method | segments | mean end err m | max err m | best accepted GT path m | mean accepted GT path m |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| test | ours learned bias | 15 | 0.017 | 0.155 | 0.839 | 0.421 |
| test | raw no bias | 15 | 0.020 | 0.156 | 0.839 | 0.421 |

## Per Segment

| sequence | role | seg | method | GT path m | end err m | max err m | diverge time s | accepted GT path m |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| MH_02_easy | test | 0 | raw no bias | 0.302 | 0.007 | 0.007 | not reached | 0.302 |
| MH_02_easy | test | 0 | ours learned bias | 0.302 | 0.001 | 0.001 | not reached | 0.302 |
| MH_02_easy | test | 1 | raw no bias | 0.291 | 0.011 | 0.011 | not reached | 0.291 |
| MH_02_easy | test | 1 | ours learned bias | 0.291 | 0.005 | 0.005 | not reached | 0.291 |
| MH_02_easy | test | 2 | raw no bias | 0.269 | 0.006 | 0.006 | not reached | 0.269 |
| MH_02_easy | test | 2 | ours learned bias | 0.269 | 0.002 | 0.002 | not reached | 0.269 |
| MH_04_difficult | test | 0 | raw no bias | 0.839 | 0.156 | 0.156 | not reached | 0.839 |
| MH_04_difficult | test | 0 | ours learned bias | 0.839 | 0.155 | 0.155 | not reached | 0.839 |
| MH_04_difficult | test | 1 | raw no bias | 0.666 | 0.021 | 0.021 | not reached | 0.666 |
| MH_04_difficult | test | 1 | ours learned bias | 0.666 | 0.021 | 0.021 | not reached | 0.666 |
| MH_04_difficult | test | 2 | raw no bias | 0.644 | 0.006 | 0.006 | not reached | 0.644 |
| MH_04_difficult | test | 2 | ours learned bias | 0.644 | 0.001 | 0.001 | not reached | 0.644 |
| V1_01_easy | test | 0 | raw no bias | 0.262 | 0.017 | 0.017 | not reached | 0.262 |
| V1_01_easy | test | 0 | ours learned bias | 0.262 | 0.011 | 0.011 | not reached | 0.262 |
| V1_01_easy | test | 1 | raw no bias | 0.217 | 0.017 | 0.017 | not reached | 0.217 |
| V1_01_easy | test | 1 | ours learned bias | 0.217 | 0.012 | 0.012 | not reached | 0.217 |
| V1_01_easy | test | 2 | raw no bias | 0.206 | 0.020 | 0.020 | not reached | 0.206 |
| V1_01_easy | test | 2 | ours learned bias | 0.206 | 0.014 | 0.014 | not reached | 0.206 |
| V1_03_difficult | test | 0 | raw no bias | 0.469 | 0.008 | 0.008 | not reached | 0.469 |
| V1_03_difficult | test | 0 | ours learned bias | 0.469 | 0.003 | 0.003 | not reached | 0.469 |
| V1_03_difficult | test | 1 | raw no bias | 0.433 | 0.006 | 0.006 | not reached | 0.433 |
| V1_03_difficult | test | 1 | ours learned bias | 0.433 | 0.003 | 0.003 | not reached | 0.433 |
| V1_03_difficult | test | 2 | raw no bias | 0.418 | 0.012 | 0.012 | not reached | 0.418 |
| V1_03_difficult | test | 2 | ours learned bias | 0.418 | 0.007 | 0.007 | not reached | 0.418 |
| V2_02_medium | test | 0 | raw no bias | 0.485 | 0.005 | 0.005 | not reached | 0.485 |
| V2_02_medium | test | 0 | ours learned bias | 0.485 | 0.004 | 0.004 | not reached | 0.485 |
| V2_02_medium | test | 1 | raw no bias | 0.411 | 0.005 | 0.005 | not reached | 0.411 |
| V2_02_medium | test | 1 | ours learned bias | 0.411 | 0.007 | 0.007 | not reached | 0.411 |
| V2_02_medium | test | 2 | raw no bias | 0.404 | 0.005 | 0.005 | not reached | 0.404 |
| V2_02_medium | test | 2 | ours learned bias | 0.404 | 0.003 | 0.003 | not reached | 0.404 |

## Files

- `summary.csv`: machine-readable metrics.
- `summary.json`: same metrics in JSON.
- `trajectories/*.csv`: common-origin GT/raw/GT-const/ours trajectories for each selected 50-step window.
- `html/index.html`: HTML report with GT retained in each plot.
