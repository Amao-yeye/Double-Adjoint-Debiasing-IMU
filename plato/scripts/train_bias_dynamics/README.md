# Bias Dynamics Training Scripts

PLATO-owned Debias_IMU training wrappers live here.  They reuse the external
Debias_IMU model/training code but keep dataset-specific glue outside
`external/`.

## LIASuw AD Pipeline

`run_liasuw_debias_ad_pipeline.py` runs the automatic-differentiation path:

1. read LIASuw 200 Hz `mav0/imu0` and `mav0/mocap0` body-GT data,
2. derive gyro/accelerometer bias labels from the mocap body trajectory,
3. train `Gyro_train`,
4. train `Acc_train_SE23`,
5. run Debias_IMU `Test` and write `FullTra_result.p`.

Use the `debiasimu` conda environment:

```bash
/home/amao/anaconda3/envs/debiasimu/bin/python \
  plato/scripts/train_bias_dynamics/run_liasuw_debias_ad_pipeline.py \
  --device cpu \
  --test-device cpu
```

The default split trains on `0912s06_80`, `0912s07_30`, `0912s10_40`,
`0203s18_80`, `0203s19_80`, and `0203s22_80`, then tests on `0205s14_80`.
Defaults are intentionally smoke-test sized (`--epoch 1`) so a real experiment
must set a larger epoch count and suitable window/batch settings explicitly.

To verify that the supervision files are the world-frame IMU/body GT and not
camera, marker, or first-frame-normalized poses:

```bash
/home/amao/anaconda3/envs/debiasimu/bin/python \
  plato/scripts/train_bias_dynamics/verify_liasuw_supervision_frames.py
```

The model/frame confirmation is documented in
`plato/docs/debias_imu_liasuw_supervision_check.md`.

To draw an interactive local 3D HTML check for one 32-sample window using
differenced pseudo-bias GT and raw IMU directly from aggregate CSV files:

```bash
/home/amao/anaconda3/envs/debiasimu/bin/python \
  plato/scripts/train_bias_dynamics/plot_liasuw_bias_preintegration_window.py \
  --source csv \
  --sequence 0912s10_40 \
  --window 32 \
  --start-index auto
```

The output is a self-contained HTML file under
`results/debias_imu/preintegration_checks/`; open it in a browser and drag to
rotate the 3D view.
