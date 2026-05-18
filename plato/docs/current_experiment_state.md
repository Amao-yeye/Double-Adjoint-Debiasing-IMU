# Current Experiment State

Date: 2026-05-17

This note is the compact handoff for the current corldata Debias_IMU / PLATO likelihood experiments. Use it before launching the next long runs.

## Dataset And Split

- Dataset root: `results/debias_imu/corldata_IMU_aligned_gt_vo_ref_sparse_gt`
- Supervision: `mocap_noninterp`
- Train: `S00`, `S01`, `S02`, `S03`
- Test: `S13`, `S15`
- Test role: visually degraded corldata sequences; keep `S13+` out of the training pool unless the split is explicitly changed.
- Current network family: no-`u_dot`
  - Gyro: `bw_func_net_no_udot`
  - Acc: `ba_func_net_no_udot`
- Test initial bias policy: `median_window`, `5.0 s`

Sparse GT timing:

- `loss_window=128` sparse poses is about `1.06 s`.
- `loss_window=256` sparse poses is about `2.13 s`.

## Current Best Mean Model

Current best full-test trajectory model:

- Config: `plato/configs/debias_imu/corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_interval_nll_joint_lr1e5_from_covonly_continue.yaml`
- Output: `results/debias_imu/corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_interval_nll_joint_lr1e5_from_covonly_continue/`
- Gyro checkpoint: `Gyro_weights/best_val_model.pt`
- Acc checkpoint: `Acc_weights/best_val_model.pt`
- Summary: `results/debias_imu/corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_interval_nll_joint_lr1e5_from_covonly_continue/interval_nll_result_summary.md`

Training:

| Stage | Best Epoch | Best Val | Learned Sigma A | Learned Sigma G |
| --- | ---: | ---: | ---: | ---: |
| Interval NLL joint bg/ba | 30 | 3376.063 | 0.004065556 | 0.000355808 |

Full-test metrics against `IMU-aligned GT`:

| Sequence | AOE deg | ATE vel | ATE pos m |
| --- | ---: | ---: | ---: |
| S13 | 7.166 | 15.995 | 181.865 |
| S15 | 6.534 | 18.101 | 290.986 |

Improvement over the previous SE23 e301 best:

| Sequence | Delta AOE deg | Delta ATE vel | Delta ATE pos m |
| --- | ---: | ---: | ---: |
| S13 | -0.787 | -2.344 | -28.059 |
| S15 | -0.703 | -0.609 | -2.840 |

Interpretation:

- This is the first NLL joint run in this thread that improves both S13 and S15 full-test metrics.
- Treat it as the current best corldata S13/S15 trajectory model, while still validating before broader claims.

Previous SE23 MSE run:

- Config: `plato/configs/debias_imu/corldata_sparse_endpoint_no_udot_w128_sg101_initmedian5_e301_s13s15.yaml`
- Output: `results/debias_imu/corldata_sparse_endpoint_no_udot_w128_sg101_initmedian5_e301_s13s15/`
- Gyro checkpoint: `Gyro_weights/best_val_model.pt`
- Acc checkpoint: `Acc_weights/best_val_model.pt`

Validation:

| Stage | Best Epoch | Best Val | Last Epoch | Last Val |
| --- | ---: | ---: | ---: | ---: |
| Gyro | 275 | 476.553 | 300 | 496.371 |
| Acc | 300 | 12396.594 | 300 | 12396.594 |

Full-test metrics against `IMU-aligned GT`:

| Sequence | AOE deg | ATE vel | ATE pos m |
| --- | ---: | ---: | ---: |
| S13 | 7.953 | 18.339 | 209.924 |
| S15 | 7.237 | 18.710 | 293.826 |

Baselines from the same setup:

| Sequence | Raw AOE deg | Raw ATE vel | Raw ATE pos m | Const AOE deg | Const ATE vel | Const ATE pos m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S13 | 33.440 | 132.043 | 2938.829 | 43.242 | 195.411 | 4281.255 |
| S15 | 21.189 | 55.579 | 798.635 | 25.966 | 78.852 | 1143.869 |

Interpretation:

- This was the previous best trajectory-mean model before interval-NLL joint.
- Acc-only continuation improved validation but worsened S13/S15 full-test drift, so the e301 run should not replace the interval-NLL joint model.
- The old SE23 gyro best is epoch 275; the current interval-NLL joint run has its own jointly fine-tuned gyro checkpoint.

## Likelihood Results

Per-state NLL ablation:

- Output: `results/debias_imu/corldata_sparse_endpoint_no_udot_w128_sg101_initmedian5_e301_s13s15_nll_optq/`
- Best Acc NLL epoch: 250
- Learned `sigma_a=0.002174545`, `sigma_g=0.000187925`
- Full-test metrics worsened relative to SE23:
  - S13: `7.953 / 27.319 / 454.737`
  - S15: `7.237 / 24.210 / 390.447`
- Conclusion: negative ablation. Do not use as current best.

Endpoint NLL with unrestricted Acc-network updates:

- Output: `results/debias_imu/corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_nll_endpoint_finetune/`
- Stopped at epoch 40.
- Validation got worse after allowing Acc updates:
  - epoch 0: `val_loss=2464.83`, endpoint vel/pos `0.7529 / 0.7629`
  - epoch 40: `val_loss=2828.55`, endpoint vel/pos `0.8218 / 0.8355`
- Conclusion: unrestricted NLL fine-tune damages the SE23-pretrained mean.

Endpoint NLL q-only calibration:

- Config: `plato/configs/debias_imu/corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_nll_endpoint_qonly.yaml`
- Output: `results/debias_imu/corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_nll_endpoint_qonly/`
- Summary: `results/debias_imu/corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_nll_endpoint_qonly/qonly_result_summary.md`
- Best epoch: 50
- Best endpoint NLL: `2552.699646`
- Learned `sigma_a=0.002793442`, `sigma_g=0.000241539`
- Full-test metrics match SE23 best because the mean network was frozen:
  - S13: `7.953 / 18.339 / 209.924`
  - S15: `7.237 / 18.710 / 293.826`

Interpretation:

- q-only likelihood calibration is stable and useful for `n_a/n_g`.
- It does not improve AOE/ATE by design.
- Use this as the covariance starting point. Current dynamics fine-tunes from it did not improve full-test metrics.

Acc-only SE23 extension from e301:

- Config: `plato/configs/debias_imu/corldata_sparse_endpoint_no_udot_w128_sg101_initmedian5_acc601_from_e301.yaml`
- Output: `results/debias_imu/corldata_sparse_endpoint_no_udot_w128_sg101_initmedian5_acc601_from_e301/`
- Gyro was skipped and loaded from current best.
- Acc was initialized from current best e301 Acc checkpoint, not from scratch.
- Training was stopped after epoch 225 because best validation had not improved since epoch 125.
- Best Acc validation: epoch 125, `10411.8`.
- Full-test metrics worsened relative to current SE23 best:
  - S13: `7.953 / 24.158 / 373.069`
  - S15: `7.237 / 22.355 / 359.907`
- Conclusion: validation improved but full-trajectory generalization worsened. Do not use this checkpoint as the current mean model or as the NLL starting mean.

Constrained joint bg/ba endpoint NLL:

- Config: `plato/configs/debias_imu/corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_nll_endpoint_joint_lr5e5_from_qonly.yaml`
- Output: `results/debias_imu/corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_nll_endpoint_joint_lr5e5_from_qonly/`
- Implementation note: PLATO runner now supports default-off `--nll-train-gyro`, saving joint fine-tuned `Gyro_weights` and `Acc_weights` in the run directory.
- Started from current SE23 e301 mean and q-only covariance, fixed `sigma_a/sigma_g`, used endpoint NLL, and jointly updated bg/ba with `lr_Acc=5e-5`.
- Validation improved from epoch 0 `2526.87` to epoch 15 `2497.08`, then regressed by epoch 25; stopped early and tested epoch 15 best.
- Full-test metrics were worse:
  - S13: `12.016 / 41.424 / 718.908`
  - S15: `9.496 / 28.107 / 420.514`
- Conclusion: joint NLL lowered short-window endpoint NLL but damaged long-horizon orientation and drift. Do not use.

Acc-only endpoint NLL with mean anchor:

- Config: `plato/configs/debias_imu/corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_nll_endpoint_acconly_anchor_lr1e5_from_qonly.yaml`
- Output: `results/debias_imu/corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_nll_endpoint_acconly_anchor_lr1e5_from_qonly/`
- Implementation note: PLATO runner now supports default-off `--nll-mean-anchor-weight` and `--nll-mean-anchor-mode`.
- Gyro was fixed, q-only covariance was fixed, Acc used `lr_Acc=1e-5`, and endpoint NLL was anchored to the initialized SE23 mean.
- Validation stayed near q-only: epoch 0 combined `2552.96`, epoch 5 `2553.06`, epoch 10 `2553.20`; stopped early and tested epoch 0 best.
- Full-test metrics were essentially preserved but slightly worse than current SE23 best:
  - S13: `7.953 / 18.435 / 212.302`
  - S15: `7.237 / 18.762 / 294.649`
- Conclusion: mean anchor prevents the severe endpoint-NLL drift, but this endpoint-NLL dynamics fine-tune did not beat the frozen SE23 mean. It has now been superseded by interval-NLL joint.

Interval-level preintegration NLL implementation:

- Implemented on 2026-05-17 in `plato/scripts/train_bias_dynamics/run_liasuw_debias_ad_pipeline.py`; this is separate from the older per-state/endpoint trajectory NLL.
- The residual is an interval preintegration factor over each low-rate pair `[s_m, s_{m+1}]`: `e_R=Log(DeltaR_hat.T @ DeltaR_star)`, `e_v=Deltav_star-Deltav_hat`, `e_p=Deltap_star-Deltap_hat`.
- Covariance is reset per interval and uses learnable scalar IMU measurement-noise variances `sigma_g2` and `sigma_a2`, not bias covariances.
- New modes: `mse_only`, `nll_cov_only`, and `nll_joint`; covariance linearization defaults to `detach_covariance_theta=true` so covariance weighting does not backpropagate into theta, while residuals still do.
- Covariance-parameter sensitivity now has a switchable backend: `interval_covariance_gradient_mode=analytic_sensitivity` propagates unit `D_g/D_a=dP/dalpha` and builds `S=alpha_g D_g + alpha_a D_a + floor`; `ad` propagates the alpha-scaled covariance recurrence directly. A synthetic smoke confirmed identical loss/raw-sigma gradients, and `interval_debug_covariance_gradients=true` can print closed-form scalar gradients for `alpha_g/alpha_a`.
- Prepared configs:
  - `plato/configs/debias_imu/corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_interval_nll_covonly_from_e301.yaml`
  - `plato/configs/debias_imu/corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_interval_nll_joint_lr1e5_from_covonly.yaml`
- Verification: Python compile, helper unit smoke, CLI help, wrapper dry-runs, cov-only training, cov-only continuation, and low-LR joint training all completed.

Interval NLL covariance-only and joint tests:

- First cov-only run:
  - Config: `plato/configs/debias_imu/corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_interval_nll_covonly_from_e301.yaml`
  - Output: `results/debias_imu/corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_interval_nll_covonly_from_e301/`
  - Best epoch: 50
  - Best validation interval NLL: `5470.974`
  - Learned `sigma_a=0.003163467`, `sigma_g=0.000273187`
  - Full-test matched the previous SE23 best because the mean was frozen.
- Continued cov-only run:
  - Config: `plato/configs/debias_imu/corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_interval_nll_covonly_continue_from_covonly.yaml`
  - Output: `results/debias_imu/corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_interval_nll_covonly_continue_from_covonly/`
  - Best epoch: 100
  - Best validation interval NLL: `3486.279`
  - Learned `sigma_a=0.004003486`, `sigma_g=0.000350775`
  - Full-test still matched the previous SE23 best because the mean was frozen.
- Low-LR interval-NLL joint:
  - Config: `plato/configs/debias_imu/corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_interval_nll_joint_lr1e5_from_covonly_continue.yaml`
  - Output: `results/debias_imu/corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_interval_nll_joint_lr1e5_from_covonly_continue/`
  - Best epoch: 30
  - Best validation interval NLL: `3376.063`
  - Learned `sigma_a=0.004065556`, `sigma_g=0.000355808`
  - Full-test improved both S13 and S15; this is the current best.

S20-S23 only-test generalization check:

- Interval NLL joint output: `results/debias_imu/corldata_interval_nll_joint_best_onlytest_s20s21s22s23/`
- SE23 e301 comparison output: `results/debias_imu/corldata_se23_e301_best_onlytest_s20s21s22s23/`
- Summary: `results/debias_imu/corldata_interval_nll_joint_best_onlytest_s20s21s22s23/s20_s23_onlytest_comparison.md`
- Full-trajectory rollout durations are about `60.017 s` for S20, `60.772 s` for S21, `62.708 s` for S22, and `63.231 s` for S23.
- Interval NLL joint vs previous SE23 e301:

| Seq | Delta AOE deg | Delta ATE vel | Delta ATE pos m |
| --- | ---: | ---: | ---: |
| S20 | -0.343 | -1.289 | -12.160 |
| S21 | -0.246 | -1.894 | -24.667 |
| S22 | -0.393 | -1.130 | -11.511 |
| S23 | -0.653 | -1.159 | -3.345 |

- Mean over high-confidence S20/S21/S23: delta AOE `-0.414 deg`, delta velocity `-1.447`, delta position `-13.390 m`.

## OpenVINS Natural-Light Control

Raw self-initialized OpenVINS was run on the original corldata S13/S15 bags, with no artificial dropout:

- Runner: `plato/scripts/openvins_eval/run_corldata_openvins_natural_trial.sh`
- Config: `plato/configs/openvins/corldata_0511/estimator_config_dynamic.yaml`
- Results: `results/openvins_eval/corldata/natural/raw/`
- Summary: `results/openvins_eval/corldata/natural/raw/summary.md`
- Evaluator: `plato/scripts/openvins_eval/summarize_corldata_openvins_natural.py`

Both OpenVINS runs wrote total-state trajectories and then exited with the known shutdown status `134`.

Using the first natural light-on interval as the SE(3) no-scale alignment window:

| Seq | Initial light-on RMSE m | First light-off RMSE / end m | Second light-off RMSE / end m | Final pos error m |
| --- | ---: | ---: | ---: | ---: |
| S13 | 0.103 | 4.986 / 10.796 | 19.470 / 46.197 | 215.440 |
| S15 | 0.062 | 4.695 / 9.710 | 20.641 / 46.976 | 290.399 |

Interpretation:

- Plain OpenVINS initializes and is initially close to the aligned GT.
- It drifts badly through the natural dark intervals and does not recover by the final light-on segment.
- These are raw OpenVINS control results only; learned-IMU/corrected-IMU OpenVINS S13/S15 results are below.

Learned-corrected OpenVINS was then run from the current best interval-NLL bias predictions:

- Corrected input export: `plato/scripts/openvins_eval/export_corldata_openvins_corrected_inputs.py`
- Apples-to-apples raw control from the same repacked PNG+IMU pipeline: `results/openvins_eval/corldata/natural/repacked_raw/`
- Generated corrected-input bags and negative run summaries were deleted after being summarized in `plato/docs/corldata_openvins_learned_imu_lessons.md`.

Using first-light-on alignment:

| Method | S13 final m | S13 light-off RMSE m | S15 final m | S15 light-off RMSE m |
| --- | ---: | ---: | ---: | ---: |
| raw repacked | 105.420 | 15.162 | 168.301 | 11.531 |
| full network corrected, default OpenVINS bias | 326.974 | 45.047 | 381.959 | 62.097 |
| full network corrected, frozen zero bias | 816.787 | 166.168 | 357.390 | 44.368 |
| network gyro-only corrected | 454.822 | 86.954 | 397.619 | 72.150 |
| network accel-only corrected | 656.233 | 93.218 | 151.146 | 11.047 |

Notes:

- S15 full-network and gyro-only corrected streams fail OpenVINS dynamic initialization under the original `init_dyn_min_deg=5.0`; the S15 corrected rows above use explicit corrected-init reruns with `init_dyn_min_deg=1.0`.
- Full learned bias correction is a negative OpenVINS result, despite improving offline Debias_IMU full-trajectory metrics.
- The main failure source appears to be learned gyro correction. Accel-only correction has a small positive signal on S15 but fails on S13, so it is not yet a robust solution.
- Do not use full raw-minus-network-bias injection as the current OpenVINS claim. Next OpenVINS correction work should isolate gyro frame/sign/initialization compatibility and test a constrained policy: no learned gyro correction initially, conservative accel correction/gating, and any per-sequence light-on adaptation reported separately from strict generalization.

Follow-up gated residual OpenVINS correction was implemented and tested:

- Exporter: `plato/scripts/openvins_eval/export_corldata_openvins_gated_residual_inputs.py`
- Comparison helper: `plato/scripts/openvins_eval/compare_corldata_openvins_natural_runs.py`
- Generated residual/delta bags and negative run summaries were deleted after being summarized in `plato/docs/corldata_openvins_learned_imu_lessons.md`.

The tested policies were:

1. absolute learned accel residual relative to first-light-on raw OpenVINS bias;
2. absolute learned accel residual relative to the raw OpenVINS bias at each light-off start;
3. learned accel bias change within each light-off interval, `b_learned(t)-b_learned(t_dark_start)`;
4. the same learned-delta policy with opposite sign.

All policies used no learned gyro correction (`gamma_g=0`), light-off-only gating, and a 1 s ramp. Key first-light-on-aligned position results:

| Method | S13 rows/dur s | S13 final m | S13 light-off RMSE m | S15 rows/dur s | S15 final m | S15 light-off RMSE m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw original | 1403 / 73.860 | 215.440 | 15.161 | 928 / 48.836 | 290.399 | 14.925 |
| raw repacked | 1277 / 67.222 | 105.420 | 15.162 | 928 / 48.836 | 168.301 | 11.531 |
| accel-only full bias | 1402 / 73.807 | 656.233 | 93.218 | 928 / 48.836 | 151.146 | 11.047 |
| learned-delta, `gamma_a=-0.10` | 1402 / 73.807 | 212.919 | 15.443 | 928 / 48.836 | 163.635 | 11.298 |
| learned-delta, `gamma_a=-2.00` | 905 / 47.624 | 68.518 | 28.431 | 928 / 48.836 | 1.752 | 2.083 |

Interpretation:

- The conservative `gamma_a=-0.10` learned-delta correction is the only tested policy that is roughly stable across both S13 and S15: S15 improves over raw-repacked, while S13 is near raw-original but does not improve light-off RMSE.
- The strong `gamma_a=-2.00` learned-delta policy is excellent on S15 position but causes S13 to terminate early and is not a valid generalized claim.
- The sign reversal giving the only strong positive signal suggests the learned accelerometer-bias trajectory and OpenVINS pre-corrected-IMU convention still have a sign/frame/offset compatibility issue.
- Current OpenVINS status: the learned model contains useful accel-trend information, but robust S13/S15 deployment through corrected IMU is not solved yet. Do not report the high-gain S15-only result as the main method.

Pure light-off preintegration was then tested without OpenVINS:

- Script: `plato/scripts/openvins_eval/evaluate_corldata_light_off_preintegration.py`
- Output: `results/openvins_eval/corldata/natural/light_off_preintegration/summary.md`
- Method: for every natural light-off interval, initialize from GT `R/v/p` at the interval start, integrate raw IMU with the tested bias correction only through the dark interval, and compare against `IMU-aligned-gt`.

Mean over the two light-off intervals per sequence:

| Method | S13 pos RMSE m | S13 end mean/max m | S13 rot RMSE deg | S15 pos RMSE m | S15 end mean/max m | S15 rot RMSE deg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 36.491 | 92.910 / 139.001 | 7.864 | 10.029 | 23.592 / 24.666 | 5.181 |
| learned full bias | 19.889 | 40.792 / 67.762 | 2.778 | 9.049 | 19.609 / 21.162 | 3.957 |
| learned gyro only | 16.641 | 37.309 / 49.639 | 2.778 | 10.192 | 22.317 / 24.124 | 3.957 |
| learned accel only | 50.710 | 125.717 / 183.723 | 7.864 | 10.702 | 25.679 / 26.027 | 5.181 |

Interpretation:

- In a controlled GT-initialized pure-IMU dark-interval test, the learned full bias improves over raw on S13 and S15.
- The improvement is dominated by learned gyro bias. Learned accel-only does not help in this test and is clearly harmful on S13.
- This contradicts the self-initialized OpenVINS corrected-IMU result where learned gyro correction is harmful, so the main OpenVINS issue is likely state/sign/frame/linearization compatibility rather than the learned gyro trajectory being useless.

The first successful self-initialized OpenVINS learned-IMU strategy was then tested:

- Input/export root: `results/openvins_eval/corldata/natural/gated_gyro_ref_inputs/`
- Run root: `results/openvins_eval/corldata/natural/gated_gyro_ref_runs/`
- Comparison: `results/openvins_eval/corldata/natural/gated_gyro_ref_comparison.md`
- Policy: keep light-on IMU raw; at each light-off start take raw OpenVINS `b_g_ref`; during light-off subtract only `gamma_g * clip(b_g_learned - b_g_ref, 0.05 rad/s)` with a 1 s ramp and no learned accel correction. This makes the effective dark-interval gyro bias move toward the learned gyro while preserving OpenVINS's light-on bias state.

First-light-on-aligned OpenVINS position results:

| Method | S13 duration s | S13 final m | S13 light-off RMSE m | S13 off1/off2 end m | S15 duration s | S15 final m | S15 light-off RMSE m | S15 off1/off2 end m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw original | 73.860 | 215.440 | 15.161 | 10.796 / 46.197 | 48.836 | 290.399 | 14.925 | 9.710 / 46.976 |
| raw repacked | 67.222 | 105.420 | 15.162 | 10.796 / 46.199 | 48.836 | 168.301 | 11.531 | 6.731 / 36.259 |
| full learned correction | 73.807 | 326.974 | 45.047 | 21.264 / 89.999 | S15 init-sensitive | 381.959 | 62.097 | 34.364 / 110.302 |
| dark gyro-ref `gamma_g=0.5` | 73.807 | 111.676 | 8.997 | 6.749 / 16.899 | 48.836 | 127.761 | 9.861 | 4.500 / 34.336 |
| dark gyro-ref `gamma_g=1.0` | 73.807 | 126.477 | 7.293 | 6.316 / 22.886 | 48.836 | 155.673 | 21.202 | 5.280 / 44.429 |

Interpretation:

- `gamma_g=0.5` is the best current self-initialized OpenVINS learned-IMU strategy: it sharply improves dark-interval endpoint errors on both S13 and S15 and reduces S15 final error from raw-repacked `168.301 m` to `127.761 m`.
- On S13, `gamma_g=0.5` reduces the second dark interval endpoint error from about `46.2 m` to `16.9 m` and cuts full-duration final error roughly in half compared with the original raw bag (`215.4 m` to `111.7 m`).
- `gamma_g=1.0` overcorrects S15, so partial blending is safer than directly replacing OpenVINS's gyro bias with learned gyro.
- This resolves the earlier contradiction: learned gyro is useful, but only when injected as a dark-interval residual relative to OpenVINS's current bias state, not as an all-sequence raw-minus-learned-bias stream.
- A simpler dark-only learned gyro-delta policy that subtracts `b_g_learned(t)-b_g_learned(t_dark_start)` without OpenVINS `b_g_ref` was also tested; it failed badly (`S13` light-off RMSE `174.593 m` at `gamma_g=0.5`, `S15` `168.037 m` and early termination). Therefore the OpenVINS current-bias reference is essential.

A finer dark gyro-ref gamma sweep was then run:

- The generated sweep inputs/runs were deleted after the conclusions were consolidated into `plato/docs/corldata_openvins_learned_imu_lessons.md`.

Key first-light-on-aligned results:

| Method | S13 rows/dur s | S13 final m | S13 light-off RMSE m | S13 off1/off2 end m | S15 final m | S15 light-off RMSE m | S15 off1/off2 end m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw original | 1403 / 73.860 | 215.440 | 15.161 | 10.796 / 46.197 | 290.399 | 14.925 | 9.710 / 46.976 |
| raw repacked | 1277 / 67.222 | 105.420 | 15.162 | 10.796 / 46.199 | 168.301 | 11.531 | 6.731 / 36.259 |
| gyro-ref `gamma_g=0.40` | 1192 / 62.744 | 56.453 | 15.534 | 2.162 / 47.440 | 38.047 | 4.495 | 2.755 / 9.500 |
| gyro-ref `gamma_g=0.50` | 1402 / 73.807 | 111.676 | 8.997 | 6.749 / 16.899 | 127.761 | 9.861 | 4.500 / 34.336 |
| gyro-ref `gamma_g=0.60` | 1402 / 73.807 | 136.619 | 14.687 | 8.898 / 23.649 | 35.691 | 4.732 | 2.600 / 10.794 |
| gyro-ref `gamma_g=0.75` | 1402 / 73.807 | 234.821 | 18.677 | 11.378 / 47.196 | 3.710 | 2.841 | 2.761 / 3.773 |

Interpretation:

- `gamma_g=0.75` gives a very strong S15 result, but overcorrects S13's longer dark intervals.
- `gamma_g=0.50` is the best full-duration S13/S15 robust single-gamma setting so far.
- A duration-aware policy is now the most plausible path to stronger generalized effect: use higher `gamma_g` for short dark intervals and lower `gamma_g` for longer dark intervals. Based on this sweep, a concrete first rule is `gamma_g=0.75` for dark intervals shorter than about `10 s`, and `gamma_g=0.50` for longer intervals, with the same OpenVINS `b_g_ref`, `0.05 rad/s` clip, and 1 s ramp.

The first duration-aware OpenVINS run was then implemented and tested:

- Exported inputs: `results/openvins_eval/corldata/natural/gated_gyro_ref_duration_inputs/`
- Runs: `results/openvins_eval/corldata/natural/gated_gyro_ref_duration_runs/`
- Comparison: `results/openvins_eval/corldata/natural/gated_gyro_ref_duration_comparison.md`
- Policy: for each light-off interval, use `gamma_g=0.75` if duration `< 10 s`, otherwise `gamma_g=0.50`; keep OpenVINS `b_g_ref`, `0.05 rad/s` clipping, no learned accelerometer correction, and a 1 s ramp.

Observed first-light-on-aligned results:

| Method | S13 final m | S13 light-off RMSE m | S13 off1/off2 end m | S15 final m | S15 light-off RMSE m | S15 off1/off2 end m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw original | 215.440 | 15.161 | 10.796 / 46.197 | 290.399 | 14.925 | 9.710 / 46.976 |
| raw repacked | 105.420 | 15.162 | 10.796 / 46.199 | 168.301 | 11.531 | 6.731 / 36.259 |
| duration-aware gyro-ref | 111.672 | 8.997 | 6.749 / 16.898 | 3.710 | 2.841 | 2.761 / 3.773 |

This is the strongest current self-initialized OpenVINS learned-IMU evidence. It preserves the robust S13 `gamma_g=0.50` behavior and captures the strong S15 short-dark-interval `gamma_g=0.75` gain.

Follow-up whole-trajectory/re-entry diagnostics:

- `summarize_corldata_openvins_natural.py` now reports `post_dark_light_on`, `last_light_on`, and `reentry_3s` groups.
- New common-horizon comparison: `results/openvins_eval/corldata/natural/gated_gyro_ref_duration_common_horizon.md`
- At the S13 raw-repacked stopping time, duration-aware learned has common final error `44.748 m` versus raw-repacked `105.420 m`; the apparent worse full final (`111.672 m` vs `105.420 m`) is because learned runs about `6.6 s` longer.
- S13 remains weak after the second dark interval: duration-aware learned last-light-on RMSE is `61.589 m`, so the remaining problem is re-entry/post-dark light-on consistency, not the dark preintegration itself.

Two re-entry variants were tested and rejected:

- Post-light-on correction tail: `export_corldata_openvins_gated_residual_inputs.py` supports `--gate-tail-s`, but the tested `3 s` tail was harmful. S13/S15 final worsened to `348.775/31.416 m`.
- Explicit dark image dropping: `make_liasuw_openvins_bags.py` supports `--light-segments`, `--sequence`, and `--drop-image-states light_off`, but dropping all light-off camera messages was harmful with the current OpenVINS config. Raw-darkdrop terminates early, and learned-darkdrop diverges badly on S13. Current evidence says to keep the natural camera stream and apply learned gyro residual only to IMU propagation during detected dark intervals.
- Detailed useful/useless attempt memory is now `plato/docs/corldata_openvins_learned_imu_lessons.md`. Generated artifacts from rejected corrected-IMU, residual, gyro-delta, darkdrop, tail, fixed-gamma sweep, and intermediate re-entry ablations were deleted after their conclusions were summarized there.

S13 re-entry stability was then improved with an opt-in OpenVINS re-entry hook:

- External OpenVINS files changed after user request: `ov_core/src/track/TrackBase.{h,cpp}` and `TrackKLT.{h,cpp}` add a tracker/database reset API; `ov_msckf/src/state/StateHelper.{h,cpp}` adds diagonal covariance inflation for selected state variables; `ov_msckf/src/core/VioManagerOptions.h`, `VioManager.h`, and `VioManager.cpp` add the PLATO dark-to-light state machine.
- Default behavior is unchanged because `plato_reentry_enable` defaults false. The corldata opt-in config is `plato/configs/openvins/corldata_0511/estimator_config_dynamic_reentry.yaml`.
- The final policy keeps the natural camera stream, clears stale KLT/ARUCO tracks on confirmed dark-to-light re-entry, inflates current IMU/clone covariance, and uses adaptive light confirmation: base dark confirm `0.5 s`, base light confirm `0.5 s`, fast light confirm `0.2 s` for dark intervals at least `10 s` or gray-dark confirmed images with mean intensity at least `5.0`; brightness/light threshold `25.0`.
- Negative ablations: clearing tracks without covariance was unstable, covariance without clearing tracks was poor on S13, `light_threshold=35` missed useful S13 gray recovery, and global immediate/0.2 s light confirmation helped S13 but hurt S15.
- Final results are in `results/openvins_eval/corldata/natural/gated_gyro_ref_duration_reentry_adaptive_gray_runs/`, with common-horizon comparisons `gated_gyro_ref_reentry_adaptive_gray_first_light_common_horizon.md` and `_all_common_horizon.md`.
- First-light-on-aligned common-horizon final error: S13 raw `214.396 m`, duration-aware learned `111.672 m`, final re-entry `3.615 m`; S15 raw `288.644 m`, duration-aware learned `3.710 m`, final re-entry `2.139 m`.
- First-light-on-aligned last-light-on RMSE: S13 duration-aware `61.589 m` to final re-entry `3.692 m`; S15 duration-aware `3.627 m` to final re-entry `2.290 m`.
- All-aligned S13 final error also improves strongly, `105.598 m` to `4.317 m`; S15 all-aligned final error is worse than duration-only (`1.325 m` to `3.042 m`), but still meter-level and stable. Use first-light-on alignment as the deployment-style re-entry metric because it fixes the initial good-vision segment before the dark intervals.

Offline Debias_IMU only-test generalization reminder for the same interval-NLL checkpoint:

- Mean over all S20-S23: delta AOE `-0.409 deg`, delta velocity `-1.368`, delta position `-12.920 m`.
- Interpretation: improvements are modest but consistent on held-out later testing sequences, so the interval-NLL gain is not only S13/S15 adaptation. S22 remains lower-confidence/stress due to prior VO/GT quality concerns.

## Code State

Important runner:

- `plato/scripts/train_bias_dynamics/run_liasuw_debias_ad_pipeline.py`
- `plato/scripts/train_bias_dynamics/run_corldata_gt_supervision_experiment.py`
- `plato/scripts/openvins_eval/summarize_corldata_openvins_natural.py`
- `plato/scripts/openvins_eval/compare_corldata_openvins_common_horizon.py`
- `plato/scripts/openvins_eval/plot_corldata_openvins_reentry_trajectory.py`
- `plato/configs/openvins/corldata_0511/estimator_config_dynamic_reentry.yaml`
- `plato/docs/corldata_openvins_learned_imu_lessons.md`

Current useful flags:

- `--ba-init-model-path`: initializes Acc dynamics from a checkpoint. It now works for PLATO readable SE23 and NLL trainers.
- `--acc-loss-type se23`: SE23 MSE trajectory-mean training.
- `--acc-loss-type nll`: endpoint/per-state likelihood.
- `--acc-loss-type interval_nll`: interval-level preintegration-factor likelihood, not per-state trajectory NLL.
- `--interval-nll-mode {mse_only,nll_cov_only,nll_joint}`: staged MSE/covariance-only/joint interval-NLL training.
- `--interval-detach-covariance-theta`: default true; detaches `Phi/G` covariance propagation from theta but keeps residual gradients to theta.
- `--interval-covariance-gradient-mode {analytic_sensitivity,ad}`: switch between explicit unit sensitivity propagation and the direct AD covariance recurrence.
- `--interval-debug-covariance-gradients`: validation-only closed-form scalar gradient print for `alpha_g/alpha_a` and raw variance parameters.
- `--interval-noise-init-model-path`: loads interval `raw_sigma_g2/raw_sigma_a2` from a prior interval-NLL checkpoint.
- `--nll-supervision-mode endpoint`: use only the final supervised state with propagated endpoint covariance.
- `--nll-covariance-strategy q_only`: freeze mean network and learn only `n_a/n_g`.
- `--nll-train-gyro`: default-off joint bg/ba endpoint-NLL fine-tune. It degraded S13/S15 in the tested run.
- `--nll-mean-anchor-weight` and `--nll-mean-anchor-mode`: default-off anchor to the initialized SE23 mean during NLL fine-tuning. It preserved the mean but did not improve it in the tested run.

Default behavior is unchanged unless these flags are explicitly set.

## Current Recommendation

- Use `corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_interval_nll_joint_lr1e5_from_covonly_continue` as the current best S13/S15 trajectory-mean model.
- Use interval-NLL learned noise for this model: `sigma_a=0.004065556`, `sigma_g=0.000355808`.
- Keep the old endpoint q-only covariance only as a historical comparison: `sigma_a=0.002793442`, `sigma_g=0.000241539`.
- Do not replace the mean with Acc-only SE23 extension, joint bg/ba endpoint NLL, or Acc-only anchored endpoint NLL from this session.
- For self-initialized OpenVINS, do not use full raw-minus-learned-bias injection. The current best OpenVINS learned-IMU strategy is duration-aware dark-only gyro residual injection relative to OpenVINS's current `b_g_ref`, plus the opt-in adaptive re-entry hook: `gamma_g=0.75` for light-off intervals shorter than `10 s`, otherwise `gamma_g=0.50`, gyro clip `0.05 rad/s`, no learned accelerometer correction, 1 s ramp, natural camera stream, clear stale tracks on confirmed dark-to-light, and covariance inflation at re-entry. If the OpenVINS source hook is unavailable, use duration-aware gyro-ref as the fallback; if a single gamma is required, use `gamma_g=0.50`.
- Do not use the tested `3 s` post-light-on correction tail, explicit dark-image-drop bags, clear-only re-entry, covariance-only re-entry, or global immediate/0.2 s light confirmation as the current policy.
- Next likelihood attempt should extend the interval-NLL joint cautiously from the current best, with frequent S13/S15 testing and early stop if full-test drift regresses.
- Short-window endpoint NLL alone is not sufficient; interval preintegration NLL is currently the promising path.

## Monitoring

Process/GPU:

```bash
pgrep -af "run_corldata_gt_supervision_experiment|run_liasuw_debias_ad_pipeline"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

Result checks:

```bash
cat results/debias_imu/<run>/evaluation_summary.txt
ls -lt results/debias_imu/<run>/Acc_weights
```

Note:

- In this environment, foreground tool sessions have been more reliable than `nohup` for long training visibility.
- If using background execution, immediately verify both `pgrep` and nonempty output/logs.
