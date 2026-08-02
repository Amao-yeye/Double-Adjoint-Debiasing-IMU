# Euroc 100-step Preintegration Drift

Divergence threshold: 1.000 m position error to GT at current time.

| role | method | segments | mean end err m | max err m | best accepted GT path m | mean accepted GT path m |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| test | ours learned bias | 15 | 0.033 | 0.176 | 1.545 | 0.817 |
| test | raw no bias | 15 | 0.053 | 0.178 | 1.545 | 0.817 |

## Per Segment

| sequence | role | seg | method | GT path m | end err m | max err m | diverge time s | accepted GT path m |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| MH_02_easy | test | 0 | raw no bias | 0.601 | 0.034 | 0.034 | not reached | 0.601 |
| MH_02_easy | test | 0 | ours learned bias | 0.601 | 0.005 | 0.005 | not reached | 0.601 |
| MH_02_easy | test | 1 | raw no bias | 0.548 | 0.045 | 0.045 | not reached | 0.548 |
| MH_02_easy | test | 1 | ours learned bias | 0.548 | 0.016 | 0.016 | not reached | 0.548 |
| MH_02_easy | test | 2 | raw no bias | 0.529 | 0.031 | 0.031 | not reached | 0.529 |
| MH_02_easy | test | 2 | ours learned bias | 0.529 | 0.007 | 0.007 | not reached | 0.529 |
| MH_04_difficult | test | 0 | raw no bias | 1.545 | 0.178 | 0.178 | not reached | 1.545 |
| MH_04_difficult | test | 0 | ours learned bias | 1.545 | 0.176 | 0.176 | not reached | 1.545 |
| MH_04_difficult | test | 1 | raw no bias | 1.309 | 0.045 | 0.045 | not reached | 1.309 |
| MH_04_difficult | test | 1 | ours learned bias | 1.309 | 0.027 | 0.027 | not reached | 1.309 |
| MH_04_difficult | test | 2 | raw no bias | 1.293 | 0.031 | 0.031 | not reached | 1.293 |
| MH_04_difficult | test | 2 | ours learned bias | 1.293 | 0.005 | 0.005 | not reached | 1.293 |
| V1_01_easy | test | 0 | raw no bias | 0.496 | 0.074 | 0.074 | not reached | 0.496 |
| V1_01_easy | test | 0 | ours learned bias | 0.496 | 0.046 | 0.046 | not reached | 0.496 |
| V1_01_easy | test | 1 | raw no bias | 0.432 | 0.077 | 0.077 | not reached | 0.432 |
| V1_01_easy | test | 1 | ours learned bias | 0.432 | 0.047 | 0.047 | not reached | 0.432 |
| V1_01_easy | test | 2 | raw no bias | 0.413 | 0.088 | 0.088 | not reached | 0.413 |
| V1_01_easy | test | 2 | ours learned bias | 0.413 | 0.058 | 0.058 | not reached | 0.413 |
| V1_03_difficult | test | 0 | raw no bias | 0.892 | 0.034 | 0.034 | not reached | 0.892 |
| V1_03_difficult | test | 0 | ours learned bias | 0.892 | 0.016 | 0.016 | not reached | 0.892 |
| V1_03_difficult | test | 1 | raw no bias | 0.863 | 0.038 | 0.038 | not reached | 0.863 |
| V1_03_difficult | test | 1 | ours learned bias | 0.863 | 0.010 | 0.010 | not reached | 0.863 |
| V1_03_difficult | test | 2 | raw no bias | 0.795 | 0.049 | 0.049 | not reached | 0.795 |
| V1_03_difficult | test | 2 | ours learned bias | 0.795 | 0.021 | 0.021 | not reached | 0.795 |
| V2_02_medium | test | 0 | raw no bias | 0.953 | 0.029 | 0.029 | not reached | 0.953 |
| V2_02_medium | test | 0 | ours learned bias | 0.953 | 0.018 | 0.018 | not reached | 0.953 |
| V2_02_medium | test | 1 | raw no bias | 0.803 | 0.019 | 0.019 | not reached | 0.803 |
| V2_02_medium | test | 1 | ours learned bias | 0.803 | 0.034 | 0.034 | not reached | 0.803 |
| V2_02_medium | test | 2 | raw no bias | 0.782 | 0.028 | 0.028 | not reached | 0.782 |
| V2_02_medium | test | 2 | ours learned bias | 0.782 | 0.012 | 0.012 | not reached | 0.782 |

## Files

- `summary.csv`: machine-readable metrics.
- `summary.json`: same metrics in JSON.
- `trajectories/*.csv`: common-origin GT/raw/GT-const/ours trajectories for each selected 100-step window.
- `html/index.html`: HTML report with GT retained in each plot.
