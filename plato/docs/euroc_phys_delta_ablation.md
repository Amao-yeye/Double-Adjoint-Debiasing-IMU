# EuRoC Phys-Delta Bias Dynamics Ablation

Date: 2026-05-09

## Goal

Validate whether the PLATO `phys_delta` bias-dynamics network can run as a drop-in replacement for the original Debias_IMU bias networks on the standard EuRoC training setup.

This is a feasibility ablation.  The controlled variable is the network structure:

- Gyro: `bw_func_phys_delta`
- Accel: `ba_func_phys_delta`

The dataset, train/test split, supervision source, loss window, batch size, optimizer settings, and training length were kept aligned with the existing EuRoC `w16` Debias_IMU runs.

## Supervision

Training and validation supervision used EuRoC ground truth:

- Source: `mav0/state_groundtruth_estimate0/data.csv`
- Runner setting: `trainval_supervision_source=state_groundtruth_estimate0`
- Test source: `state_groundtruth_estimate0`

This run did not use VO or pseudo-GT supervision.

Train sequences:

- `MH_01_easy`
- `MH_03_medium`
- `MH_05_difficult`
- `V1_02_medium`
- `V2_01_easy`
- `V2_03_difficult`

Held-out test sequences:

- `MH_02_easy`
- `MH_04_difficult`
- `V2_02_medium`
- `V1_03_difficult`
- `V1_01_easy`

## Runner

PLATO runner:

- `plato/scripts/train_bias_dynamics/run_euroc_debias_phys_delta_pipeline.py`

Result directory:

- `results/debias_imu/EUROC_phys_delta_e1801_w16_b1000/`

The runner reuses external Debias_IMU's `EUROCDataset`, losses, training structure, and evaluation functions, but imports PLATO-owned network choices from `m2_bias_networks.py`.  It also reuses the readable validation metric and delta-IMU auxiliary input handling already validated for LIASuw.  No external repository files were modified.

Command:

```bash
/home/amao/anaconda3/envs/debiasimu/bin/python plato/scripts/train_bias_dynamics/run_euroc_debias_phys_delta_pipeline.py \
  --outputdir results/debias_imu/EUROC_phys_delta_e1801_w16_b1000 \
  --cache-dir results/debias_imu/cache/EUROC_gt_w16 \
  --test-cache-dir results/debias_imu/cache/EUROC_test_gt \
  --epoch 1801 \
  --batch-size 1000 \
  --val_freq 100 \
  --save_freq 100 \
  --device cpu \
  --test-device cpu \
  --test-recompute
```

Environment:

- Conda env: `/home/amao/anaconda3/envs/debiasimu`
- PyTorch: `2.4.0+cu121`
- CUDA availability in this session: `False`, so the run used CPU.

## Training Behavior

Training completed both stages without numerical instability.

Gyro stage:

- Epoch 0 validation rotation RMSE: about `0.155905 deg`
- Epoch 1800 validation rotation RMSE: about `0.150670 deg`
- Best checkpoint: epoch `1800`, validation loss `9.543348252773285`

Accel stage:

- Epoch 0 validation SE23 loss: `2181.4629669189453`
- Epoch 1700 best validation SE23 loss: `2098.1941528320312`
- Epoch 1800 validation SE23 loss: `2098.218551635742`
- Epoch 1800 validation velocity RMSE: about `0.0549603 m/s`
- Epoch 1800 validation position RMSE: about `0.00155695 m`

Training profile:

- Total training time: `5620.59 s`
- Gyro average epoch time: `1.5475 s`
- Accel average epoch time: `1.5730 s`
- GPU memory: `0 MB` because this run used CPU.

## Held-Out Test Results

Main result uses the last checkpoint, matching the original EuRoC runner's default checkpoint convention:

- Gyro checkpoint: `Gyro_weights/model_epoch_1800.pt`
- Accel checkpoint: `Acc_weights/model_epoch_1800.pt`

| Sequence | AOE deg | ATE vel | ATE pos |
|---|---:|---:|---:|
| `MH_02_easy` | 1.977 | 17.835 | 1027.200 |
| `MH_04_difficult` | 3.326 | 14.958 | 473.157 |
| `V1_01_easy` | 5.608 | 19.418 | 1028.183 |
| `V1_03_difficult` | 3.035 | 11.030 | 418.376 |
| `V2_02_medium` | 3.421 | 18.149 | 725.455 |

Mean over the five held-out sequences:

| Method | Mean AOE deg | Mean ATE vel | Mean ATE pos |
|---|---:|---:|---:|
| `phys_delta` | 3.473 | 16.278 | 734.474 |
| existing `Euroc_SE23_w16` | 2.713 | 18.068 | 779.317 |
| existing `Euroc_baseline_w16` | 2.713 | 17.871 | 771.703 |

Compared with existing `Euroc_SE23_w16`, `phys_delta` changes by:

| Sequence | Delta AOE deg | Delta ATE vel | Delta ATE pos |
|---|---:|---:|---:|
| `MH_02_easy` | -1.146 | +13.413 | +782.424 |
| `MH_04_difficult` | +2.272 | +10.364 | +319.880 |
| `V1_01_easy` | +2.749 | -12.795 | -723.424 |
| `V1_03_difficult` | +0.774 | -6.775 | -210.196 |
| `V2_02_medium` | -0.849 | -13.159 | -392.898 |

## Conclusion

The new `phys_delta` structure is feasible on EuRoC: it trains stably for the full 1801-epoch two-stage setup, validation losses decrease, checkpoints load correctly, and the standard full-trajectory held-out evaluation runs to completion.

The result is not uniformly better than the original networks.  On average, `phys_delta` improves held-out velocity and position error versus `Euroc_SE23_w16`, but worsens average AOE.  It improves translation/velocity strongly on `V1_01_easy`, `V1_03_difficult`, and `V2_02_medium`, while degrading position and velocity on `MH_02_easy` and `MH_04_difficult`.  Treat this as a positive feasibility result and a mixed performance result, not a final architecture win.
