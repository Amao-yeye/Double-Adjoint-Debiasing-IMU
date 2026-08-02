# Euroc Rollout Error Summary

Endpoint errors averaged over 15 sampled rollouts per L. Rotation error is `||Log(R_gt^T R_hat)||`.

| L steps | method | segments | mean rot rad | mean rot deg | mean vel m/s | mean pos m |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 50 | raw | 15 | 0.020394 | 1.1685 | 0.0878 | 0.0201 |
| 50 | ours | 15 | 0.001299 | 0.0745 | 0.0507 | 0.0167 |
| 100 | raw | 15 | 0.040592 | 2.3257 | 0.1994 | 0.0533 |
| 100 | ours | 15 | 0.001883 | 0.1079 | 0.0839 | 0.0331 |
| 150 | raw | 15 | 0.060249 | 3.4520 | 0.3543 | 0.1119 |
| 150 | ours | 15 | 0.002884 | 0.1653 | 0.1249 | 0.0557 |
| 200 | raw | 15 | 0.080031 | 4.5855 | 0.5625 | 0.2224 |
| 200 | ours | 15 | 0.003579 | 0.2051 | 0.1539 | 0.0859 |
