# LIASuw Route-A Gravity Alignment

Route A was tested to reduce the mismatch between Debias_IMU pseudo accelerometer bias and the small physical bias initialized by OpenVINS.

## Implementation

- Gravity estimator: `plato/scripts/train_bias_dynamics/estimate_liasuw_gravity_bias.py`
- Calibration output: `results/debias_imu/LIASuw_gravity_calibration/gravity_calibration.json`
- Human-readable table: `results/debias_imu/LIASuw_gravity_calibration/gravity_calibration.md`
- Runner support: `run_liasuw_debias_ad_pipeline.py --gravity-calibration-json <json>`

The estimator fits, per sequence:

```text
a_m = R_WB^T (a_W - g_W) + b_a
```

The runner does not change external Debias_IMU propagation. Instead, when a gravity calibration JSON is provided, it rotates each sequence's supervised trajectory into a gravity-aligned world where Debias_IMU's internal fixed gravity `[0, 0, -9.81]` remains valid. This matters because changing only the pseudo-bias label while leaving the ODE propagation gravity fixed is inconsistent.

## Results

Corrected full run:

```text
results/debias_imu/LIASuw_phys_delta_routeA_worldalign_e1800_w16_b64
```

Configuration: mocap 200 Hz supervision, `phys_delta`, `epoch=1800`, `loss_window=16`, `batch_size=64`, CPU, best validation checkpoints, train sequences `0912s06_80`, `0912s07_30`, `0912s10_40`, `0203s18_80`, `0203s19_80`, `0203s22_80`, held-out test `0205s14_80`.

Held-out `0205s14_80` metrics:

| method | AOE deg | ATE vel | ATE pos | accel bias used |
| --- | ---: | ---: | ---: | --- |
| raw | 44.627 | 297.555 | 7954.494 | none |
| constant Route-A | 19.206 | 85.931 | 2064.886 | `[0.3818, 2.8503, -0.8802]` |
| learned Route-A | 26.397 | 72.605 | 1682.771 | network |
| old mocap learned | 6.001 | 38.320 | 1246.066 | network |
| corrected VO learned | 6.861 | 4.814 | 154.274 | network |

Conclusion: Route A makes the constant-bias baseline more physically plausible and improves the constant-bias velocity/position result compared with old mocap constant bias, but the learned Route-A model is worse than both the old mocap-supervised network and the corrected VO-supervised network on held-out `0205s14_80`. It is therefore a useful diagnostic, not the current best training route.

Invalid intermediate:

```text
results/debias_imu/LIASuw_phys_delta_routeA_gravity_e1800_w16_b64
```

This run changed pseudo-bias labels with calibrated gravity but left Debias_IMU propagation in the original fixed-gravity world. It is dynamically inconsistent and should not be used.
