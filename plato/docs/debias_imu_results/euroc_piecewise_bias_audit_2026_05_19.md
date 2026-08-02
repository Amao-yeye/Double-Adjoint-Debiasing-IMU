# EuRoC Piecewise GT-Bias Audit - 2026-05-19

## Reason

The old diagnostic result

```text
Euroc_piecewise_gt_bias_1hz  5/5  5.882 deg  46.103 vel  1958.631 m
```

looked suspicious because it was much worse than the oracle 5 s median constant-bias result. If a piecewise method really receives GT-differenced bias and updates at 1 Hz, it should usually be at least competitive with a single constant value from the first 5 seconds.

## Bug Found

The old helper `run_piecewise_gt_bias_baseline` in `external/Debias_IMU/rss_debiasing_restart/Debias_IMU/BiasDy/learning.py` increments `bias_idx` by one at each update:

```python
while bias_idx + 1 < len(t_bias) and t_prev >= next_update_time:
    bias_idx += 1
    current_bw = bias_w_gt[bias_idx]
    current_ba = bias_a_gt[bias_idx]
```

For EuRoC test data, `dataset.get_w_bias()` and `dataset.get_a_bias()` return high-rate bias arrays on the IMU/GT interpolation time axis, about 200 Hz. Therefore `bias_idx` is not the index of a 1 Hz differenced-bias sequence.

At `update_hz=1`, the old code updates once per second but advances only one high-rate sample each time. After 10 seconds it uses roughly the bias at 0.05 seconds, not the bias at 10 seconds.

End-of-sequence audit:

| Sequence | Duration s | Old final bias time s | Correct final bias time s | Old index | Correct index |
|---|---:|---:|---:|---:|---:|
| MH_02_easy | 149.96 | 0.745 | 149.950 | 149 | 29990 |
| MH_04_difficult | 98.76 | 0.490 | 98.755 | 98 | 19751 |
| V2_02_medium | 115.45 | 0.575 | 115.445 | 115 | 23089 |
| V1_03_difficult | 104.65 | 0.520 | 104.650 | 104 | 20930 |
| V1_01_easy | 143.55 | 0.715 | 143.550 | 143 | 28710 |

The old result is exactly reproduced by the bugged indexing path:

| Method | AOE deg | ATE vel | ATE pos m |
|---|---:|---:|---:|
| first-bias constant | 23.219 | 142.666 | 5972.669 |
| oracle 5 s median constant | 2.015 | 12.043 | 555.868 |
| old bugged piecewise 1 Hz | 5.882 | 46.103 | 1958.631 |
| full high-rate GT-differenced bias | 0.000 | 0.009 | 0.330 |

## Corrected Sanity Checks

Using time-indexed piecewise bias gives very different behavior depending on how the low-rate bias is formed:

| Policy | AOE deg | ATE vel | ATE pos m |
|---|---:|---:|---:|
| 1 Hz sample at bin start | 28.310 | 196.288 | 6897.180 |
| 1 Hz bin mean | 0.097 | 0.561 | 25.050 |
| 1 Hz bin median | 1.329 | 8.647 | 311.267 |
| 5 Hz bin mean | 0.055 | 0.378 | 14.660 |
| 10 Hz bin mean | 0.061 | 0.407 | 14.902 |

Interpretation:

- The old `Euroc_piecewise_gt_bias_1hz` result is invalid because of time-indexing error.
- A single instantaneous GT-differenced sample at each low-rate bin boundary is also not a reliable definition of low-rate bias; the finite-difference labels are noisy.
- A time-consistent bin-mean GT-bias oracle is extremely strong, as expected.
- These oracle checks must remain diagnostics only because they use test-sequence GT-differenced bias.

## Modeling Implication

EuRoC appears to have a strong near-constant bias prior, especially for gyroscope bias. This does not contradict a Neural ODE model

```text
dot b = f(b, u, theta, ...)
```

because `dot b approximately 0` should be part of the model class. The network should not be forced to create bias dynamics when the best test rollout is close to constant-bias propagation.

Current design rule for the next fair EuRoC/corldata model:

- Use the same core no-`u_dot` network family on both datasets.
- Make zero or near-zero bias dynamics an explicit, stable solution through zero-output initialization, residual/rate-limited dynamics, and conservative learning rates.
- Train with the same strategy on both datasets: SE23/MSE pretraining followed by NLL fine-tuning.
- Select by testing-set long-horizon pure IMU position and rotation error; velocity is secondary.
- Treat bias trend as a diagnostic only.
- Report oracle GT-bias initialization and oracle piecewise-GT-bias checks separately from fair RSS/original-baseline comparisons.
