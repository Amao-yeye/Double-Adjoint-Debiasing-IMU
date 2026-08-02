# Euroc 200-step Preintegration Drift

Divergence threshold: 1.000 m position error to GT at current time.

| role | method | segments | mean end err m | max err m | best accepted GT path m | mean accepted GT path m |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| test | ours learned bias | 15 | 0.086 | 0.236 | 2.964 | 1.572 |
| test | raw no bias | 15 | 0.222 | 0.418 | 2.964 | 1.572 |

## Per Segment

| sequence | role | seg | method | GT path m | end err m | max err m | diverge time s | accepted GT path m |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| MH_02_easy | test | 0 | raw no bias | 1.155 | 0.200 | 0.200 | not reached | 1.155 |
| MH_02_easy | test | 0 | ours learned bias | 1.155 | 0.016 | 0.016 | not reached | 1.155 |
| MH_02_easy | test | 1 | raw no bias | 1.075 | 0.213 | 0.213 | not reached | 1.075 |
| MH_02_easy | test | 1 | ours learned bias | 1.075 | 0.026 | 0.026 | not reached | 1.075 |
| MH_02_easy | test | 2 | raw no bias | 1.037 | 0.191 | 0.191 | not reached | 1.037 |
| MH_02_easy | test | 2 | ours learned bias | 1.037 | 0.014 | 0.014 | not reached | 1.037 |
| MH_04_difficult | test | 0 | raw no bias | 2.964 | 0.227 | 0.227 | not reached | 2.964 |
| MH_04_difficult | test | 0 | ours learned bias | 2.964 | 0.172 | 0.172 | not reached | 2.964 |
| MH_04_difficult | test | 1 | raw no bias | 2.617 | 0.191 | 0.191 | not reached | 2.617 |
| MH_04_difficult | test | 1 | ours learned bias | 2.617 | 0.019 | 0.030 | not reached | 2.617 |
| MH_04_difficult | test | 2 | raw no bias | 2.475 | 0.195 | 0.195 | not reached | 2.475 |
| MH_04_difficult | test | 2 | ours learned bias | 2.475 | 0.056 | 0.056 | not reached | 2.475 |
| V1_01_easy | test | 0 | raw no bias | 0.990 | 0.352 | 0.352 | not reached | 0.990 |
| V1_01_easy | test | 0 | ours learned bias | 0.990 | 0.174 | 0.174 | not reached | 0.990 |
| V1_01_easy | test | 1 | raw no bias | 0.837 | 0.372 | 0.372 | not reached | 0.837 |
| V1_01_easy | test | 1 | ours learned bias | 0.837 | 0.189 | 0.189 | not reached | 0.837 |
| V1_01_easy | test | 2 | raw no bias | 0.815 | 0.418 | 0.418 | not reached | 0.815 |
| V1_01_easy | test | 2 | ours learned bias | 0.815 | 0.236 | 0.236 | not reached | 0.815 |
| V1_03_difficult | test | 0 | raw no bias | 1.714 | 0.195 | 0.195 | not reached | 1.714 |
| V1_03_difficult | test | 0 | ours learned bias | 1.714 | 0.030 | 0.030 | not reached | 1.714 |
| V1_03_difficult | test | 1 | raw no bias | 1.558 | 0.146 | 0.146 | not reached | 1.558 |
| V1_03_difficult | test | 1 | ours learned bias | 1.558 | 0.045 | 0.045 | not reached | 1.558 |
| V1_03_difficult | test | 2 | raw no bias | 1.486 | 0.220 | 0.220 | not reached | 1.486 |
| V1_03_difficult | test | 2 | ours learned bias | 1.486 | 0.057 | 0.057 | not reached | 1.486 |
| V2_02_medium | test | 0 | raw no bias | 1.748 | 0.163 | 0.163 | not reached | 1.748 |
| V2_02_medium | test | 0 | ours learned bias | 1.748 | 0.071 | 0.071 | not reached | 1.748 |
| V2_02_medium | test | 1 | raw no bias | 1.592 | 0.076 | 0.076 | not reached | 1.592 |
| V2_02_medium | test | 1 | ours learned bias | 1.592 | 0.132 | 0.132 | not reached | 1.592 |
| V2_02_medium | test | 2 | raw no bias | 1.516 | 0.179 | 0.179 | not reached | 1.516 |
| V2_02_medium | test | 2 | ours learned bias | 1.516 | 0.052 | 0.052 | not reached | 1.516 |

## Files

- `summary.csv`: machine-readable metrics.
- `summary.json`: same metrics in JSON.
- `trajectories/*.csv`: common-origin GT/raw/GT-const/ours trajectories for each selected 200-step window.
- `html/index.html`: HTML report with GT retained in each plot.
