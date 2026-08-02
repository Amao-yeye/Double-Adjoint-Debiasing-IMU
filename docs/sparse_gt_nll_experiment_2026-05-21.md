# Sparse GT SE23 NLL experiment log (2026-05-21)

## Status

This is a historical sparse GT NLL search log. It records attempts made before
the strict paper branch was fixed to:

```text
mse_residual_mode = right
nll_gt_residual_mode = right
r = Log(X_gt * X_pred^{-1})
```

For the current strict-right NLL result, use:

```text
docs/README.md
docs/SE23_strict_definition_memory.md
docs/Euroc_double_adjoint_training_eval.md
```

The `log` and `hybrid` NLL/MSE-anchor experiments below are useful ablations,
not the paper mainline.

## Implemented cost function

Added sparse transition-factor GT NLL for full-state `SE_2(3)` supervision.

Implementation files:

- `external/Debias_IMU/BiasDy/plato/nll.py`
  - `sparse_full_state_nll(...)`
  - uses transition residual `d_i = tilde_r_{i+1} - Phi_i tilde_r_i`
  - uses Cholesky solves and logdet of each sparse interval covariance `Q_i`
  - does not build dense trajectory covariance / dense precision
- `external/Debias_IMU/BiasDy/plato/losses.py`
  - `se23_gt_right_residual(...) = Log(X_gt X_pred^{-1})`
- `external/Debias_IMU/BiasDy/plato/training.py`
  - `--nll_gt_mode sparse`
  - `--nll_gt_residual_mode {right,log,hybrid_global}`
- `external/Debias_IMU/BiasDy/mainEurocPLATO.py`
  - CLI switches for sparse NLL mode/residual mode
- `external/Debias_IMU/BiasDy/plato/nll_smoke_test.py`
  - sparse NLL finite/gradient smoke test

Verification:

```bash
/home/amao/anaconda3/envs/debiasimu/bin/python -m py_compile \
  external/Debias_IMU/BiasDy/plato/nll.py \
  external/Debias_IMU/BiasDy/plato/losses.py \
  external/Debias_IMU/BiasDy/plato/training.py \
  external/Debias_IMU/BiasDy/mainEurocPLATO.py \
  external/Debias_IMU/BiasDy/network_for_BiadDy.py

cd external/Debias_IMU
/home/amao/anaconda3/envs/debiasimu/bin/python BiasDy/plato/nll_smoke_test.py
```

Both passed.

## Reference MSE result

Main SE23-MSE result:

`external/Debias_IMU/results/Euroc_PLATO_bg_damped145e3_ba_init_bagate22e2_oldV1P10_eval`

Configuration summary:

- BG: `bw_func_net_damped_145e3`
  - loads `results/Euroc_baseline_w64/Gyro_weights/model_epoch_1800.pt`
  - applies `dot bg = f(u,bg) - 0.145 bg`
- BA: `ba_func_net_init_bagate_22e2`
  - loads `results/Euroc_PLATO_acc_se23_hybrid_V1P10_w64_b1000/Acc_weights/best_val_model.pt`
  - freezes BA dynamics if initial `||ba0|| >= 0.22`

Mean test metrics:

- AOE: `3.0437306881 deg`
- velocity RMSE: `15.5518524170`
- APE / position RMSE: `610.5109115601 m`

The requested `>3%` APE improvement target over this MSE result is:

`610.5109115601 * 0.97 = 592.1955842133 m`

## Best sparse NLL result found

Best APE found in this search:

`external/Debias_IMU/results/Euroc_PLATO_sparse_gt_nll_logres_from_bg12_initgate22_w64_b1000_lr1e6_anchor001_freezeA_eval_e40`

Configuration:

- sparse GT NLL
- residual mode: `log`
  - This is the old `Log(X_gt^{-1} X_pred)` residual path, not the exact right residual.
  - It gave better APE than exact right residual in these runs.
- start from `bg=0.14` / initial BA gate setup
- train stage: freeze BA, train BG + IMU covariance
- `lr_NLL=1e-6`
- `nll_mse_anchor_weight=0.001`
- MSE anchor: `hybrid`, rot `1`, vel `1`, pos `10`
- checkpoint: `NLL_weights/model_epoch_40.pt`
- evaluation-only damping sweep used `bw_func_net_damped_12e2`

Mean test metrics:

- AOE: `3.2290480852 deg`
- velocity RMSE: `14.7629238129`
- APE / position RMSE: `592.2882736206 m`

Relative to main MSE:

- APE improved by `2.984818%`
- velocity RMSE improved by `5.072892%`
- AOE worsened by `6.088495%`

This misses the strict `>3%` APE target by about `0.092689 m`, and it does not preserve the previous AOE `>10%` improvement over baseline. Relative to baseline AOE, this result is only `4.824234%` better.

## Attempts and conclusions

Failed / not enough:

- exact sparse right-residual NLL, freeze BG, train BA:
  - APE worsened to about `613.8248 m`
- sparse hybrid-global residual NLL, freeze BG, train BA:
  - APE worsened to about `613.7611 m`
- sparse NLL with BA train and MSE anchor:
  - APE stayed worse than MSE, about `612-617 m`
- sparse NLL freeze BA, train BG, best-val checkpoint:
  - final/best-val APE worsened, but intermediate checkpoints improved APE
- sparse NLL joint BG+BA:
  - APE and AOE both worsened (`687.9271 m`, `3.5990 deg`)
- stronger rotation anchor (`rot=10`):
  - did not preserve AOE enough and only gave small APE improvement (`607.9195 m`)
- BA gate threshold `0.25`:
  - worsened V1_03 and overall APE
- damping sweep around the best intermediate checkpoint:
  - best was around `0.12`, but still just short of `3%`

Useful:

- sparse NLL can improve position when used as a small BG fine-tune with BA frozen and early stopping.
- NLL validation loss is not a reliable selector for final full-trajectory APE. Best validation checkpoint was worse than intermediate checkpoints.
- The best APE checkpoint was around epoch `40`, not final or best validation.

Current status:

- sparse GT NLL cost function is implemented and usable.
- APE improvement is almost at `3%` but not strictly above it.
- AOE is currently not preserved in the APE-improving sparse NLL branch.

Recommended next direction:

Use a validation selector closer to final metrics, or add an explicit rotation-preserving constraint that does not kill the BG-induced position/velocity improvement. The most promising branch is:

```text
NLL residual mode: log
freeze BA
train BG + covariance
lr_NLL = 1e-6
anchor = 0.001
early checkpoint around epoch 35-45
BG damping around 0.12-0.14
```
