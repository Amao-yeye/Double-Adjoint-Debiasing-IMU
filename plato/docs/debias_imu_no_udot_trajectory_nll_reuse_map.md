# Debias_IMU No-u_dot Trajectory-State NLL Reuse Map

Date: 2026-05-18

This note records the mandatory implementation audit for
`plato_debias_imu_codex_plan.md`. The implementation keeps new PLATO behavior
behind explicit flags and avoids modifying external repositories.

| Component | Existing file/module | Reuse directly? | Need modification? | Notes |
|---|---|---:|---:|---|
| IMU preintegration / ODE rollout | `external/Debias_IMU/rss_debiasing_restart/Debias_IMU/BiasDy/network_for_BiadDy.py`, `learning.odeint_SO3` | Yes | No | Reused nominal state order `[xi, t, b_g, v, p, b_a]` and ODE solver path. |
| Lie group ops | `external/Debias_IMU/rss_debiasing_restart/Debias_IMU/BiasDy/lie_algebra.py` | Yes | No | Reused `SO3log`, `SEn3log`, `SEn3inverse`, `SEn3fromSO3Rn`; no duplicate Lie utilities added. |
| Pose residual | `BiasDy/BiasDy_loss.py::SE23_loss` convention | Yes | Minimal | Trajectory-state NLL uses the same residual convention `Log(inv(X_gt) @ X_pred)` for consistency with existing SE23 MSE/NLL. |
| Covariance propagation | `BiasDy/covariance.py::paper_invariant_error_matrices` and PLATO interval covariance helpers | Partial | Yes | Reused invariant error matrices. Added PLATO-side joint supervised-time covariance block construction for trajectory-state NLL. |
| Cholesky / Gaussian solve | `BiasDy/BiasDy_loss.py::SE23_nll_loss` pattern | Yes | Minimal | Reused Cholesky/`cholesky_solve` pattern. New loss avoids explicit inverse and logs quad/logdet/cholesky diagnostics. |
| MSE loss | `BiasDy/BiasDy_loss.py::SE23_loss`, PLATO `Acc_train_SE23_with_readable_metrics` | Yes | No | Kept as Stage-1 pretraining/baseline. |
| Existing NLL loss | `BiasDy/BiasDy_loss.py::SE23_nll_loss`, PLATO interval NLL in `run_liasuw_debias_ad_pipeline.py` | Partial | Yes | Existing per-state and interval-factor NLL remain runnable. New `trajectory_nll` is a separate explicit loss type. |
| Bias model / Neural ODE | `BiasDy/network_for_BiadDy.py`, `plato/scripts/train_bias_dynamics/m2_bias_networks.py` | Yes | No | No-u_dot networks are selected by config, e.g. `bw_func_net_no_udot` / `ba_func_net_no_udot`. |
| EuRoC dataset loader | `BiasDy/dataset.py::EUROCDataset`, PLATO EuRoC runner | Yes | Minimal | Reused existing loader and source-timing support. New loss is exposed through the PLATO EuRoC runner. |
| corldata dataset loader | `plato/scripts/train_bias_dynamics/run_liasuw_debias_ad_pipeline.py::LIASuwDebiasDataset` | Yes | Minimal | Reused corldata/LIASuw non-interpolated supervision and source-timing paths. |
| Timestamp synchronization | Existing dataset cache construction and `eval_indices` | Yes | No | Trajectory NLL uses `eval_indices` to project supervised timestamps into ODE integration indices. |
| Coordinate-frame alignment | Existing pseudo-GT/export/alignment scripts and dataset cache fields | Yes | No | Loss assumes cached supervision already in the body/IMU frame, matching existing SE23 training. |
| Normalization / smoothing | Existing Savitzky-Golay bias generation in dataset caches | Yes | No | No new normalization path added. |
| Evaluation metrics | Existing `Test`, `simple_visualization`, `ATE/AOE` summaries | Yes | No | Full-sequence tests remain the same. |
| Bias visualization | Existing `learning.plot_bias`, PLATO `compare_euroc_predicted_bias_to_gt.py` | Partial | Yes | EuRoC trend metrics exist; corldata should get the same trend comparison before final reporting. |

Implementation status:

- `acc_loss_type=trajectory_nll` is now available in the PLATO corldata/LIASuw
  and EuRoC runners.
- The older `acc_loss_type=interval_nll` remains available and is still labeled
  as an independent interval-factor baseline.
- `trajectory_nll` supports `trajectory_supervision_mode=sparse|endpoint`,
  `trajectory_supervision_stride`, and `trajectory_max_supervised_states`.
- The initial supervised state is excluded from the stacked likelihood because
  its covariance is zero by construction.
- EuRoC and corldata have different IMU rates. Fair window comparisons should
  use physical duration `(loss_window - 1) * dt`, not raw step count. For
  example, EuRoC `loss_window=64, dt=0.005` spans about `0.315 s`, while a
  corldata 1000 Hz run needs about `loss_window=320` for the same duration.

Known limitation:

- The current trajectory-state covariance uses dense block covariance over
  selected supervised timestamps. This is correct for the current sparse-window
  experiments, but a sparse/block-tridiagonal precision implementation may be
  needed for much longer windows.
