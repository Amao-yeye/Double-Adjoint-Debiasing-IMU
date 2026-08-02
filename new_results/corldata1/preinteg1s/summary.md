# corldata 1 s Preintegration Drift

Divergence threshold: 1.000 m position error to GT at current time.

| role | method | segments | mean end err m | max err m | best accepted GT path m | mean accepted GT path m |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| test | GT-diff const bias | 4 | 1.928 | 2.480 | 0.194 | 0.165 |
| test | ours learned bias | 4 | 0.114 | 0.159 | 0.231 | 0.220 |
| test | raw no bias | 4 | 0.187 | 0.258 | 0.231 | 0.220 |
| train | GT-diff const bias | 4 | 0.484 | 0.656 | 0.240 | 0.220 |
| train | ours learned bias | 4 | 0.223 | 0.303 | 0.240 | 0.220 |
| train | raw no bias | 4 | 0.191 | 0.348 | 0.240 | 0.220 |

## Per Segment

| sequence | role | seg | method | GT path m | end err m | max err m | diverge time s | accepted GT path m |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| S00 | train | 0 | raw no bias | 0.240 | 0.123 | 0.123 | not reached | 0.240 |
| S00 | train | 0 | GT-diff const bias | 0.240 | 0.446 | 0.446 | not reached | 0.240 |
| S00 | train | 0 | ours learned bias | 0.240 | 0.303 | 0.303 | not reached | 0.240 |
| S00 | train | 1 | raw no bias | 0.232 | 0.186 | 0.186 | not reached | 0.232 |
| S00 | train | 1 | GT-diff const bias | 0.232 | 0.258 | 0.258 | not reached | 0.232 |
| S00 | train | 1 | ours learned bias | 0.232 | 0.297 | 0.297 | not reached | 0.232 |
| S00 | train | 2 | raw no bias | 0.213 | 0.107 | 0.107 | not reached | 0.213 |
| S00 | train | 2 | GT-diff const bias | 0.213 | 0.576 | 0.576 | not reached | 0.213 |
| S00 | train | 2 | ours learned bias | 0.213 | 0.051 | 0.051 | not reached | 0.213 |
| S00 | train | 3 | raw no bias | 0.194 | 0.348 | 0.348 | not reached | 0.194 |
| S00 | train | 3 | GT-diff const bias | 0.194 | 0.656 | 0.656 | not reached | 0.194 |
| S00 | train | 3 | ours learned bias | 0.194 | 0.240 | 0.240 | not reached | 0.194 |
| S13 | test | 0 | raw no bias | 0.231 | 0.258 | 0.258 | not reached | 0.231 |
| S13 | test | 0 | GT-diff const bias | 0.231 | 2.439 | 2.439 | 0.646 | 0.143 |
| S13 | test | 0 | ours learned bias | 0.231 | 0.122 | 0.122 | not reached | 0.231 |
| S13 | test | 1 | raw no bias | 0.229 | 0.185 | 0.185 | not reached | 0.229 |
| S13 | test | 1 | GT-diff const bias | 0.229 | 1.561 | 1.561 | 0.799 | 0.186 |
| S13 | test | 1 | ours learned bias | 0.229 | 0.113 | 0.113 | not reached | 0.229 |
| S13 | test | 2 | raw no bias | 0.216 | 0.128 | 0.128 | not reached | 0.216 |
| S13 | test | 2 | GT-diff const bias | 0.216 | 1.232 | 1.232 | 0.902 | 0.194 |
| S13 | test | 2 | ours learned bias | 0.216 | 0.062 | 0.062 | not reached | 0.216 |
| S13 | test | 3 | raw no bias | 0.206 | 0.177 | 0.177 | not reached | 0.206 |
| S13 | test | 3 | GT-diff const bias | 0.206 | 2.480 | 2.480 | 0.634 | 0.138 |
| S13 | test | 3 | ours learned bias | 0.206 | 0.159 | 0.159 | not reached | 0.206 |

## Files

- `summary.csv`: machine-readable metrics.
- `summary.json`: same metrics in JSON.
- `trajectories/*.csv`: common-origin GT/raw/GT-const/ours trajectories for each selected 1 s window.
- `html/index.html`: HTML report with GT retained in each plot.
