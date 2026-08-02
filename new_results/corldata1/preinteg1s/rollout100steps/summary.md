# corldata 100-step Preintegration Drift

Divergence threshold: 1.000 m position error to GT at current time.

| role | method | segments | mean end err m | max err m | best accepted GT path m | mean accepted GT path m |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| test | GT-diff const bias | 4 | 0.018 | 0.021 | 0.024 | 0.023 |
| test | ours learned bias | 4 | 0.001 | 0.002 | 0.024 | 0.023 |
| test | raw no bias | 4 | 0.002 | 0.003 | 0.024 | 0.023 |
| train | GT-diff const bias | 4 | 0.010 | 0.024 | 0.031 | 0.027 |
| train | ours learned bias | 4 | 0.009 | 0.020 | 0.031 | 0.027 |
| train | raw no bias | 4 | 0.008 | 0.020 | 0.031 | 0.027 |

## Per Segment

| sequence | role | seg | method | GT path m | end err m | max err m | diverge time s | accepted GT path m |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| S00 | train | 0 | raw no bias | 0.031 | 0.020 | 0.020 | not reached | 0.031 |
| S00 | train | 0 | GT-diff const bias | 0.031 | 0.024 | 0.024 | not reached | 0.031 |
| S00 | train | 0 | ours learned bias | 0.031 | 0.020 | 0.020 | not reached | 0.031 |
| S00 | train | 1 | raw no bias | 0.027 | 0.004 | 0.004 | not reached | 0.027 |
| S00 | train | 1 | GT-diff const bias | 0.027 | 0.005 | 0.005 | not reached | 0.027 |
| S00 | train | 1 | ours learned bias | 0.027 | 0.005 | 0.005 | not reached | 0.027 |
| S00 | train | 2 | raw no bias | 0.026 | 0.007 | 0.007 | not reached | 0.026 |
| S00 | train | 2 | GT-diff const bias | 0.026 | 0.005 | 0.005 | not reached | 0.026 |
| S00 | train | 2 | ours learned bias | 0.026 | 0.008 | 0.008 | not reached | 0.026 |
| S00 | train | 3 | raw no bias | 0.023 | 0.002 | 0.002 | not reached | 0.023 |
| S00 | train | 3 | GT-diff const bias | 0.023 | 0.008 | 0.008 | not reached | 0.023 |
| S00 | train | 3 | ours learned bias | 0.023 | 0.002 | 0.002 | not reached | 0.023 |
| S13 | test | 0 | raw no bias | 0.024 | 0.002 | 0.002 | not reached | 0.024 |
| S13 | test | 0 | GT-diff const bias | 0.024 | 0.019 | 0.019 | not reached | 0.024 |
| S13 | test | 0 | ours learned bias | 0.024 | 0.001 | 0.001 | not reached | 0.024 |
| S13 | test | 1 | raw no bias | 0.024 | 0.003 | 0.003 | not reached | 0.024 |
| S13 | test | 1 | GT-diff const bias | 0.024 | 0.011 | 0.011 | not reached | 0.024 |
| S13 | test | 1 | ours learned bias | 0.024 | 0.002 | 0.002 | not reached | 0.024 |
| S13 | test | 2 | raw no bias | 0.023 | 0.003 | 0.003 | not reached | 0.023 |
| S13 | test | 2 | GT-diff const bias | 0.023 | 0.020 | 0.020 | not reached | 0.023 |
| S13 | test | 2 | ours learned bias | 0.023 | 0.002 | 0.002 | not reached | 0.023 |
| S13 | test | 3 | raw no bias | 0.023 | 0.001 | 0.001 | not reached | 0.023 |
| S13 | test | 3 | GT-diff const bias | 0.023 | 0.021 | 0.021 | not reached | 0.023 |
| S13 | test | 3 | ours learned bias | 0.023 | 0.000 | 0.001 | not reached | 0.023 |

## Files

- `summary.csv`: machine-readable metrics.
- `summary.json`: same metrics in JSON.
- `trajectories/*.csv`: common-origin GT/raw/GT-const/ours trajectories for each selected 100-step window.
- `html/index.html`: HTML report with GT retained in each plot.
