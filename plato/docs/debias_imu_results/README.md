# Debias_IMU Result Records

This directory is the centralized record for `results/debias_imu/` cleanup and experiment status.

- `cleanup_2026_05_19.md`: what was kept, what was deleted, and why.
- `negative_attempts_2026_05_19.md`: negative or misleading attempts that should not be reported as final results.
- `euroc_piecewise_bias_audit_2026_05_19.md`: audit of the old EuRoC `piecewise_gt_bias_1hz` result and the corrected modeling rule for near-constant bias dynamics.
- `euroc_fair_first_bias_screens_2026_05_19.md`: fair first-bias EuRoC no-`u_dot` screens after the oracle median result was rejected.
- `euroc_train_global_prior_screens_2026_05_19.md`: train-only global bias prior diagnostics, checkpoint re-tests, and train-global-y0 conservative residual screen.
- `next_window_network_improvement_handoff_2026_05_19.md`: compact handoff for continuing network improvement in a fresh context window.

Current reporting rule:

- Select models by testing-set position and rotation in long-horizon pure IMU rollout.
- Bias trend is diagnostic only.
- EuRoC results using test-sequence GT-differenced bias initialization must be labeled as oracle-style initialization and must not be compared directly against RSS/original baselines.
- The old `Euroc_piecewise_gt_bias_1hz` diagnostic is invalid because its `bias_idx` advanced through a high-rate bias array one sample per second rather than selecting the bias by timestamp. Use the audit record above before citing any EuRoC piecewise-GT-bias number.
- Formal EuRoC/corldata comparison should use the same core no-`u_dot` network family and the same MSE-to-NLL strategy, with dataset-specific timing/calibration handled explicitly.
- The shared network family must include `dot b approximately 0` as a stable solution; EuRoC may correctly learn near-constant bias dynamics while corldata can still use the same residual dynamics structure for nonstationary behavior.
