# Next Window Handoff - Network Improvement - 2026-05-19

## User Goal

Improve the bias-dynamics network under these constraints:

- One shared core architecture should cover EuRoC and corldata.
- Bias trend similarity to GT-differenced bias is not a target.
- Primary metric is held-out testing pure IMU long-horizon rollout position and rotation error; velocity is secondary.
- The structure must allow constant bias / `dot b = 0` as an optimal solution.
- Training strategy should be unified: SE23/MSE stabilizes the mean, then NLL fine-tunes uncertainty without destroying the mean.
- EuRoC does not need more dynamic bias; it needs dynamics that stay still when appropriate and make only small corrections when useful.

## Current Reliable Conclusions

EuRoC:

- The earlier apparently strong EuRoC result with `initial_bias_mode=median_window`, `initial_bias_window_s=5.0` was oracle-style because it used each test sequence's GT-differenced bias. Do not report it as a fair RSS comparison.
- Corrective first-bias test of the old MSE checkpoint was poor: about `23.218 deg / 141.311 vel / 5917.102 m`.
- Fair first-bias screens all failed to beat RSS:
  - conservative residual low LR: `15.956 / 91.609 / 3888.399`;
  - conservative residual high LR: `86.154 / 312.634 / 11157.326`;
  - large no-`u_dot`: `50.703 / 272.202 / 9533.365`;
  - large no-`u_dot` with held-out `V2_01_easy` validation: `66.882 / 380.405 / 13819.146`.
- Train-only global median constant bias is fairer but still not enough: `4.758 / 28.755 / 1129.517`.
- Retesting existing MSE checkpoint with train-global initialization did not improve it: `4.758 / 29.901 / 1176.710`.
- Re-training conservative residual with train-global y0 labels failed badly: `48.166 / 166.129 / 5473.824`.
- Short-window validation repeatedly improves while held-out full rollout gets worse. Do not trust short-window validation alone for EuRoC.

corldata:

- Current credible selected result is:
  `results/debias_imu/corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_interval_nll_joint_lr1e5_from_covonly_continue/`
- S13/S15 full-test against IMU-aligned GT:
  - S13 `7.166 deg / 15.995 vel / 181.865 m`;
  - S15 `6.534 / 18.101 / 290.986`.
- It improves over previous SE23 on S13/S15 and modestly on S20-S23, but the full six-sequence corldata test attempted later was stopped and is incomplete.

## Current Code Additions

Conservative shared network family:

- File: `plato/scripts/train_bias_dynamics/m2_bias_networks.py`
- Network names:
  - `bw_func_conservative_residual`
  - `ba_func_conservative_residual`
- Properties:
  - no `u_dot`;
  - zero-output initialization;
  - bounded residual bias rate;
  - near-zero decay, so constant bias is a valid initial solution.

EuRoC runner additions:

- File: `plato/scripts/train_bias_dynamics/run_euroc_debias_phys_delta_pipeline.py`
- Supports:
  - `--val-seqs`;
  - `--initial-bias-mode train_global_mean/train_global_median`;
  - `--skip-plots` while still writing numeric `evaluation_summary.txt`;
  - AD interval NLL through existing PLATO NLL trainer.

Diagnostics:

- `plato/scripts/train_bias_dynamics/evaluate_euroc_train_global_constant_bias.py`
- `plato/scripts/train_bias_dynamics/prepare_euroc_train_global_bias_cache.py`

Key records:

- `plato/docs/debias_imu_results/euroc_piecewise_bias_audit_2026_05_19.md`
- `plato/docs/debias_imu_results/euroc_fair_first_bias_screens_2026_05_19.md`
- `plato/docs/debias_imu_results/euroc_train_global_prior_screens_2026_05_19.md`
- `plato/docs/current_experiment_state.md`
- `plato/docs/project_memory.md`

## Important Pitfalls

- Do not compare any test-GT median/mean initialization result against RSS.
- Do not optimize for bias trend matching.
- Do not treat `Euroc_piecewise_gt_bias_1hz` old result as valid. It used the wrong bias index progression.
- Do not run NLL after a bad MSE mean checkpoint and expect NLL to fix full drift.
- Do not rely on same-sequence tail validation for EuRoC; it has repeatedly selected bad full-rollout models.
- EuRoC and corldata have different IMU/source rates. Compare physical time span, not raw step count.

## Recommended Next Technical Direction

The next useful design should be full-rollout-aware, not another short-window MSE screen.

Concrete plan:

1. Keep the same core architecture:
   - `bw_func_conservative_residual`;
   - `ba_func_conservative_residual`;
   - no `u_dot`;
   - zero-output / constant-bias solution preserved.

2. Add full-rollout-aware selection for EuRoC:
   - during MSE training, periodically run fast validation rollout on one or more held-out validation sequences;
   - select checkpoint by full-sequence AOE + position, not short-window SE23 only;
   - start with train/val split such as train `MH_01`, `MH_03`, `MH_05`, `V1_02`, val `V2_01`, test default.

3. Add a strong mean-preservation regularizer:
   - penalize integrated bias-rate norm over the rollout;
   - penalize deviation from constant-bias rollout unless full-rollout validation improves;
   - keep gyro especially conservative, because gyro overfitting destroys orientation.

4. Only after the selected MSE checkpoint beats or approaches RSS, run NLL:
   - small LR;
   - freeze or heavily regularize mean dynamics;
   - NLL is a fine-tune, not a rescue stage.

5. Then port the same core architecture and training strategy to corldata:
   - dataset-specific window duration and sampling rate are allowed;
   - keep objective and network core consistent.

## Suggested New-Window Prompt

Start the new window with:

```text
We are in /home/amao/workspace/PLATO. Read AGENTS.md, plato/docs/project_memory.md, plato/docs/current_experiment_state.md, and plato/docs/debias_imu_results/next_window_network_improvement_handoff_2026_05_19.md first.

Goal: improve PLATO/Debias_IMU bias-dynamics network for EuRoC and corldata. Same core architecture across datasets, no u_dot, constant bias must remain a valid optimum, optimize held-out pure IMU full-rollout pos/rot, not bias trend. Training strategy is MSE -> NLL, but only run NLL after MSE mean is good.

Current EuRoC fair attempts failed because short-window validation is misleading. Next task: implement full-rollout-aware EuRoC checkpoint selection/regularization for bw_func_conservative_residual / ba_func_conservative_residual, then run a small screen and compare against RSS baseline 3.494 deg / 16.295 vel / 678.010 m.
```
