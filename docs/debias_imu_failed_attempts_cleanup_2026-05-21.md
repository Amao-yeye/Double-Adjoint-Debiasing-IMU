# Debias IMU failed-attempt record and cleanup log (2026-05-21)

## Status

This file is a historical cleanup log from before the strict paper branch was
fixed. Its "current MSE mainline" refers to the older gated/hybrid search stage.

For the current paper-facing mainline, use:

```text
docs/README.md
docs/SE23_strict_definition_memory.md
docs/Euroc_double_adjoint_training_eval.md
docs/Euroc_strict_se23_acc_result.md
```

This file records failed or superseded EuRoC experiments before switching to the
new sparse NLL loss. All result directories listed under "Deleted" were either
unsuccessful, diagnostic-only, or superseded by the then-current MSE mainline.

## Historical Baseline And Target

- Baseline directory:
  `external/Debias_IMU/results/Euroc_baseline_w64`
- Baseline mean metrics:
  - AOE: `3.3927208662 deg`
  - velocity RMSE: `16.3727520466`
  - position RMSE / APE: `683.6750579834 m`
- Then-current SE23-MSE main result:
  `external/Debias_IMU/results/Euroc_PLATO_bg_damped145e3_ba_init_bagate22e2_oldV1P10_eval`
- Then-current SE23-MSE mean metrics:
  - AOE: `3.0437306881 deg` (`10.29%` better than baseline)
  - velocity RMSE: `15.5518524170`
  - position RMSE / APE: `610.5109115601 m` (`10.70%` better than baseline)

## Failed NLL attempts before sparse NLL

These attempts used the old dense/woodbury-style NLL path and did not improve
the current SE23-MSE main result by the requested `3%`.

| Attempt | Setting | Result | Decision |
| --- | --- | --- | --- |
| `Euroc_PLATO_nll_from_bg145_initgate22_w64_b64_lr1e4_freezeG` | fixed BG, train BA + covariance, `lr_NLL=1e-4`, no MSE anchor | eval APE `624.1433 m`; `2.23%` worse than MSE | rejected |
| `Euroc_PLATO_nll_from_bg145_initgate22_w64_b64_lr5e5_anchor001_freezeG` | fixed BG, train BA + covariance, `lr_NLL=5e-5`, MSE anchor `0.001` | eval APE `613.1861 m`; `0.44%` worse than MSE | rejected |
| `Euroc_PLATO_nll_from_bg145_initgate22_w64_b64_lr1e5_anchor001_freezeA` | fixed BA, train BG + covariance, `lr_NLL=1e-5`, MSE anchor `0.001` | training validation NLL decreased, but evaluation was stopped before completion per user request | rejected / incomplete |

Interpretation: the old NLL validation objective can decrease while
full-trajectory test APE does not improve. This motivates replacing it with the
new sparse NLL loss before continuing.

## Other failed or superseded attempts

These were already summarized in `docs/debias_imu_paper_code_notes.md` and are
safe to remove from `external/Debias_IMU/results` after recording:

- Multi-window BG training and fine-tuning attempts:
  - `Euroc_PLATO_bg_mwrot_w16_32_64_w64_b1000`
  - `Euroc_PLATO_bg_mwrot_w16_32_64_w64_b1000_eval_e300`
  - `Euroc_PLATO_bg_mwrot_ftbase_w16_32_64_lr1e4_b1000`
  - `Euroc_PLATO_bg_mwrot_ftbase_w16_32_64_lr1e4_b1000_eval_e100`
  - `Euroc_PLATO_bg_mwrot_ftbase_w16_32_64_lr1e4_b1000_eval_e400`
  - `Euroc_PLATO_bg_mwrot_ftbase_w16_32_64_lr5e4_b1000`
  - `Euroc_PLATO_bg_mwrot_ftbase_w16_32_64_lr5e4_b1000_eval_e120`
- Stronger BG damping or BA retraining attempts that improved AOE but did not
  improve position:
  - `Euroc_PLATO_bg_damped2e1_acc_oldV1P10_eval`
  - `Euroc_PLATO_bg_damped2e1_acc_V1P10_w64_b1000`
  - `Euroc_PLATO_bg_damped2e1_acc_V1P10_w64_b1000_eval_best`
  - `Euroc_PLATO_bg_damped2e1_acc_V1P10_w64_b1000_eval_e300`
  - `Euroc_PLATO_bg_damped2e1_acc_from_oldV1P10_lr1e4_w64_b1000`
  - `Euroc_PLATO_bg_damped2e1_acc_from_oldV1P10_lr1e4_w64_b1000_eval_best`
  - `Euroc_PLATO_bg_damped2e1_acc_from_oldV1P10_w400_b128_pos10`
  - `Euroc_PLATO_bg_damped2e1_acc_from_oldV1P10_w400_b128_pos10_eval_best`
  - `Euroc_PLATO_bg_damped_2e1_baseline_w64_eval`
- BG damping / BA weight scans that were useful diagnostics but not final:
  - `Euroc_PLATO_bg_damped1e1_acc_oldV1P10_eval`
  - `Euroc_PLATO_bg_damped15e2_acc_oldV1P10_eval`
  - `Euroc_PLATO_bg_damped15e2_acc_from_oldV1P10_w64_b1000_v01p10`
  - `Euroc_PLATO_bg_damped15e2_acc_from_oldV1P10_w64_b1000_v01p10_eval_best`
  - `Euroc_PLATO_bg_damped15e2_acc_from_oldV1P10_w64_b1000_v0p10`
  - `Euroc_PLATO_bg_damped15e2_acc_from_oldV1P10_w64_b1000_v0p10_eval_best`
  - `Euroc_PLATO_bg_damped15e2_ba_const_eval`
  - `Euroc_PLATO_bg_damped15e2_ba_damped1e1_oldV1P10_eval`
- BA structure attempts rejected before the current initial-gate result:
  - `Euroc_PLATO_bg_damped145e3_acc_biassup_w64_b1000_v0p10_ba1`
  - `Euroc_PLATO_bg_damped145e3_ba_bagate28e2_oldV1P10_eval`
  - `Euroc_PLATO_bg_damped145e3_ba_const_eval`
  - `Euroc_PLATO_bg_damped145e3_ba_rate_tanh1e2_oldV1P10_eval`
  - `Euroc_PLATO_bg_damped145e3_ba_scaled1e1_oldV1P10_eval`
  - `Euroc_PLATO_bg_damped14e2_acc_rate_tanh2e2_w64_b1000_v1p10_e1800`
  - `Euroc_PLATO_bg_damped14e2_acc_rate_tanh2e2_w64_b1000_v1p10_e1800_eval_best`
  - `Euroc_PLATO_bg_damped14e2_ba_bagate_hard22e2_oldV1P10_eval`
  - `Euroc_PLATO_bg_damped14e2_ba_const_eval`
- Older V1P10 NLL experiments superseded by the then-current MSE mainline:
  - `Euroc_PLATO_nll_from_V1P10_anchor005_w64_b64`
  - `Euroc_PLATO_nll_from_V1P10_anchor005_w64_b64_eval_best`

## Kept result directories

Keep these because they were baseline, the then-current main result, a
successful backup, or required checkpoints for reproducing the then-current
main result:

- `Euroc_baseline_w64`
- `Euroc_PLATO_acc_se23_hybrid_V1P10_w64_b1000`
- `Euroc_PLATO_acc_se23_hybrid_V1P10_w64_b1000_eval_best`
- `Euroc_PLATO_bg_damped145e3_ba_init_bagate22e2_oldV1P10_eval`
- `Euroc_PLATO_bg_damped14e2_ba_init_bagate22e2_oldV1P10_eval`
- `vo_quality`

## Cleanup status

Cleanup was executed after this record was written. Remaining directories under
`external/Debias_IMU/results` are:

- `Euroc_baseline_w64`
- `Euroc_PLATO_acc_se23_hybrid_V1P10_w64_b1000`
- `Euroc_PLATO_acc_se23_hybrid_V1P10_w64_b1000_eval_best`
- `Euroc_PLATO_bg_damped145e3_ba_init_bagate22e2_oldV1P10_eval`
- `Euroc_PLATO_bg_damped14e2_ba_init_bagate22e2_oldV1P10_eval`
- `vo_quality`

No `mainEurocPLATO.py`, `mainEuroc.py`, or `evaluate_gyro_aoe` process was left
running after cleanup.
