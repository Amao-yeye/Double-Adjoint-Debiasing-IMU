# Euroc 150-step Preintegration Drift

Divergence threshold: 1.000 m position error to GT at current time.

| role | method | segments | mean end err m | max err m | best accepted GT path m | mean accepted GT path m |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| test | ours learned bias | 15 | 0.056 | 0.152 | 2.258 | 1.211 |
| test | raw no bias | 15 | 0.112 | 0.216 | 2.258 | 1.211 |

## Per Segment

| sequence | role | seg | method | GT path m | end err m | max err m | diverge time s | accepted GT path m |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| MH_02_easy | test | 0 | raw no bias | 0.874 | 0.090 | 0.090 | not reached | 0.874 |
| MH_02_easy | test | 0 | ours learned bias | 0.874 | 0.009 | 0.009 | not reached | 0.874 |
| MH_02_easy | test | 1 | raw no bias | 0.817 | 0.104 | 0.104 | not reached | 0.817 |
| MH_02_easy | test | 1 | ours learned bias | 0.817 | 0.013 | 0.013 | not reached | 0.817 |
| MH_02_easy | test | 2 | raw no bias | 0.790 | 0.096 | 0.096 | not reached | 0.790 |
| MH_02_easy | test | 2 | ours learned bias | 0.790 | 0.032 | 0.032 | not reached | 0.790 |
| MH_04_difficult | test | 0 | raw no bias | 2.258 | 0.164 | 0.164 | not reached | 2.258 |
| MH_04_difficult | test | 0 | ours learned bias | 2.258 | 0.152 | 0.152 | not reached | 2.258 |
| MH_04_difficult | test | 1 | raw no bias | 1.975 | 0.092 | 0.092 | not reached | 1.975 |
| MH_04_difficult | test | 1 | ours learned bias | 1.975 | 0.016 | 0.030 | not reached | 1.975 |
| MH_04_difficult | test | 2 | raw no bias | 1.911 | 0.091 | 0.091 | not reached | 1.911 |
| MH_04_difficult | test | 2 | ours learned bias | 1.911 | 0.038 | 0.038 | not reached | 1.911 |
| V1_01_easy | test | 0 | raw no bias | 0.756 | 0.185 | 0.185 | not reached | 0.756 |
| V1_01_easy | test | 0 | ours learned bias | 0.756 | 0.099 | 0.099 | not reached | 0.756 |
| V1_01_easy | test | 1 | raw no bias | 0.638 | 0.190 | 0.190 | not reached | 0.638 |
| V1_01_easy | test | 1 | ours learned bias | 0.638 | 0.106 | 0.106 | not reached | 0.638 |
| V1_01_easy | test | 2 | raw no bias | 0.618 | 0.216 | 0.216 | not reached | 0.618 |
| V1_01_easy | test | 2 | ours learned bias | 0.618 | 0.131 | 0.131 | not reached | 0.618 |
| V1_03_difficult | test | 0 | raw no bias | 1.297 | 0.098 | 0.098 | not reached | 1.297 |
| V1_03_difficult | test | 0 | ours learned bias | 1.297 | 0.019 | 0.019 | not reached | 1.297 |
| V1_03_difficult | test | 1 | raw no bias | 1.273 | 0.104 | 0.104 | not reached | 1.273 |
| V1_03_difficult | test | 1 | ours learned bias | 1.273 | 0.041 | 0.041 | not reached | 1.273 |
| V1_03_difficult | test | 2 | raw no bias | 1.240 | 0.049 | 0.049 | not reached | 1.240 |
| V1_03_difficult | test | 2 | ours learned bias | 1.240 | 0.036 | 0.036 | not reached | 1.240 |
| V2_02_medium | test | 0 | raw no bias | 1.367 | 0.078 | 0.078 | not reached | 1.367 |
| V2_02_medium | test | 0 | ours learned bias | 1.367 | 0.042 | 0.042 | not reached | 1.367 |
| V2_02_medium | test | 1 | raw no bias | 1.215 | 0.040 | 0.040 | not reached | 1.215 |
| V2_02_medium | test | 1 | ours learned bias | 1.215 | 0.075 | 0.075 | not reached | 1.215 |
| V2_02_medium | test | 2 | raw no bias | 1.139 | 0.081 | 0.081 | not reached | 1.139 |
| V2_02_medium | test | 2 | ours learned bias | 1.139 | 0.027 | 0.027 | not reached | 1.139 |

## Files

- `summary.csv`: machine-readable metrics.
- `summary.json`: same metrics in JSON.
- `trajectories/*.csv`: common-origin GT/raw/GT-const/ours trajectories for each selected 150-step window.
- `html/index.html`: HTML report with GT retained in each plot.
