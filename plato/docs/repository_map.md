# Repository Map

This map is based on files present after reorganization on 2026-05-08.

## Paper Submodule

Path: `paper/plato-overleaf`

Main purpose: Overleaf-managed PLATO manuscript source.

Evidence:
- `.gitmodules`
- `paper/README.md`
- `paper/plato-overleaf/main.tex`
- `paper/plato-overleaf/corl_2026.sty`
- `paper/plato-overleaf/corlref.bib`

Important directories and files:
- `paper/plato-overleaf/main.tex`: main manuscript source.
- `paper/plato-overleaf/Figures`: paper figure directory.
- `paper/plato-overleaf/corl_2026.sty`: CoRL style file.
- `paper/plato-overleaf/corlref.bib`: bibliography file.
- `paper/plato-overleaf/open_questions.tex`: open questions file.

Build system:
- No repository-specific build script was inspected. This appears to be a standard LaTeX / Overleaf project.

Connection to PLATO:
- The paper should provide scientific context and algorithmic claims for PLATO.
- Code and experiments should remain outside the paper submodule.
- If paper claims change the implementation plan, update `plato/docs/integration_plan.md` and `plato/docs/project_memory.md`.

Uncertain:
- The original local Overleaf checkout had uncommitted `main.tex` changes and untracked LaTeX build outputs when the submodule was added. The submodule clone reflects committed repository state unless those changes are committed or manually applied.

## Root Tools

Path: `tools`

Main purpose: small PLATO-owned preprocessing utilities requested for local experiments.

Important files:
- `tools/estimate_mocap_body_gt.py`: estimates the mocap marker-to-IMU/body extrinsic `T_MB` from mocap rigid body poses and VIO body poses, then exports body-frame ground truth in the mocap/world frame or first-frame-normalized frame. It can output GT on IMU timestamps, mocap timestamps, or a fixed rate such as 200 Hz, and can write Debias_IMU/TUM-VI-style mocap CSV files.

## PLATO Scripts

Path: `plato/scripts`

Main purpose: PLATO-owned glue scripts for dataset-specific ORB-SLAM3 export/evaluation, trajectory conversion, and future Debias_IMU/OpenVINS orchestration.

Workflow docs:
- `plato/docs/corldata_pseudogt_evaluation_workflow.md`: precise corldata pseudo-GT generation/evaluation procedure, including ORB-SLAM3 stereo VO generation from bag, time-offset search, fitted camera/body GT conversion, evo SE(3) no-scale alignment, metric formulas, outputs, visualization, and quality checks.

Important files:
- `plato/scripts/orbslam3_export/run_liasuw_orbslam3_trial.py`: LIASuw ORB-SLAM3 runner/evaluator. It supports `--mode stereo` and `--mode stereo-inertial`, writes raw estimates under `results/pseudo_gt/<dataset>/<sequence>/`, and computes SE(3) aligned position/orientation metrics plus plots and evo outputs.
- `plato/scripts/orbslam3_export/run_corldata_batch.py`: corldata S00-S05 ORB-SLAM3 stereo batch runner/evaluator. It exports DAVIS stereo bags, runs stereo visual-only ORB-SLAM3, automatically searches per-sequence time offsets by minimizing SE(3)-aligned position RMSE, and can regenerate historical batch diagnostics. Its old generated outputs were cleaned after the `evo_gt_aligned` workflow superseded them.
- `plato/scripts/orbslam3_export/diagnose_corldata_gt_quality.py`: corldata mocap `Body_0.tum` quality diagnostic. It checks GT timestamp, position, and orientation discontinuities, cross-checks them against aligned VO-GT residual spikes, and writes reports/plots under `results/pseudo_gt/corldata/gt_quality/`.
- `plato/scripts/orbslam3_export/evaluate_corldata_masked_segments.py`: supplementary corldata evaluator that masks selected suspicious intervals from `gt_quality/suspicious_segments.csv`, recomputes SE(3) no-scale alignment and fixed-frame-compensated orientation metrics, and can regenerate historical `masked_gt_only/` or `masked_gt_jumps/` diagnostics.
- `plato/scripts/orbslam3_export/detect_corldata_light_segments.py`: detects corldata testing-set lights-on/off intervals from exported DAVIS grayscale images, maps them onto the full converted GT trajectory using the saved time offset, and writes black/gray GT XY PNGs plus brightness diagnostics under `results/pseudo_gt/corldata/light_segments/`.
- `plato/scripts/orbslam3_export/make_corldata_3d_html.py`: writes self-contained draggable Canvas HTML views for corldata Ground truth vs SE(3)-aligned ORB-SLAM3 estimated VO, including XY/YZ/XZ projection canvases with meter ticks and scale bars; by default it discovers all sequences with `evo_gt_aligned` outputs.
- `plato/scripts/orbslam3_export/make_corldata_se3_fit_alignment.py`: diagnostic corldata evaluator that fits a per-sequence right-multiplied `T_camera_to_body_fit` and can regenerate historical `se3_fit/` aligned trajectories; this is not a calibrated global extrinsic.
- `plato/scripts/orbslam3_export/run_corldata_evo_pseudogt_workflow.py`: user-approved corldata pseudo-GT generation/evaluation workflow. It generates ORB-SLAM3 stereo VO from corldata bags when raw VO is missing, estimates time offset and a per-sequence diagnostic camera/body frame transform, converts mocap `Body_0` GT into fitted camera-frame GT, runs `evo_traj --align --save_as_tum` and `evo_ape --align`, stores evo's aligned trajectory under each sequence's `evo/`, and writes aggregate `<summary-prefix>_report.md/.csv/.json` outputs.
- `plato/scripts/orbslam3_export/evaluate_corldata_stereo_inertial_result.py`: evaluates an existing corldata ORB-SLAM3 stereo-inertial EuRoC trajectory. It treats ORB-SLAM3 `IMU_STEREO` output as IMU/body pose, fits a same-sequence diagnostic mocap Body_0-to-estimated-body transform, converts GT into the fitted body frame, and runs evo `--align` without scale correction under a separate output directory such as `results/pseudo_gt/corldata/S02/stereo_inertial_eval/`.
- `plato/scripts/export_liasuw_debias_body_gt.py`: exports Debias_IMU-ready LIASuw body GT for current pseudo-GT training candidates. It converts stereo-only VO left-camera poses `T_Wv_C` to IMU/body poses with camera-IMU extrinsics, estimates the mocap marker-to-IMU/body transform, writes per-sequence `debias_imu_body_gt_200hz` artifacts, and builds an aggregate TUM-VI-style dataset root at `results/debias_imu/LIASuw_training_candidates_200hz/`.
- `plato/scripts/export_corldata_debias_body_pseudogt.py`: exports corldata ORB-SLAM3 stereo visual-only teacher trajectories into IMU/body pseudo-GT. It reads `evo_gt_aligned/estimate_evo_aligned.tum`, applies the physical DAVIS-left-camera to IMU/body extrinsic, shifts IMU timestamps into the image/VO clock, gravity-aligns the world, and writes per-sequence `debias_imu_body_gt_200hz/vo_imu_traj.tum` plus an aggregate Debias_IMU-style tree under `results/debias_imu/corldata_training_pseudogt/`.
- `plato/scripts/train_bias_dynamics/run_liasuw_debias_ad_pipeline.py`: PLATO-owned automatic-differentiation Debias_IMU runner for LIASuw 200 Hz body GT, corldata IMU-aligned GT, and source-timing sparse endpoint supervision. It reuses external Debias_IMU training/test code, adds a sequence-list-based loader for TUM-VI-style exports, can train on non-interpolated `vo_imu_traj.tum` or sparse mocap endpoint poses transformed to the IMU/body frame, supports optional Route-A per-sequence gravity alignment via `--gravity-calibration-json`, supports stage-control flags such as `--stop-after-gyro` and `--skip-gyro-train`, supports default-off robust test initial-bias policies via `--initial-bias-mode`, supports explicit Acc checkpoint initialization via `--ba-init-model-path`, supports per-state or endpoint SE_2(3) NLL Acc refinement with learned `n_a/n_g` process covariance, supports interval-level preintegration-factor NLL via `--acc-loss-type interval_nll` with `mse_only`, `nll_cov_only`, and `nll_joint` modes, supports switchable interval covariance gradients via `--interval-covariance-gradient-mode {analytic_sensitivity,ad}` plus optional closed-form scalar gradient debugging, supports default-off joint bg/ba NLL fine-tuning via `--nll-train-gyro`, supports default-off initialized-mean anchoring via `--nll-mean-anchor-weight`, and writes outputs under `results/debias_imu/`.
- `plato/scripts/train_bias_dynamics/run_corldata_gt_supervision_experiment.py`: config-driven wrapper around the shared Debias_IMU runner for corldata experiments. It records resolved configs/splits and forwards VO-supervision fields such as `pseudo_gt_root`, `vo_supervision_world_frame`, `vo_gap_tolerance_s`, `gravity_accel_world`, robust initial-bias fields, trajectory-NLL refinement fields, and interval-preintegration NLL fields including the analytic-sensitivity/AD covariance-gradient switch.
- `plato/configs/debias_imu/corldata_imu_aligned_gt_baseline_se23_u_dot_u.yaml`: baseline Debias_IMU config for processed corldata `IMU-aligned-gt`, using original `bw_func_net` / `ba_func_net`, train `S00/S02/S03/S07`, and test `S13/S15/S20/S21/S22/S23`.
- `plato/configs/debias_imu/corldata_imu_aligned_gt_baseline_expanded_train_se23_u_dot_u.yaml`: formal expanded corldata `IMU-aligned-gt` baseline config using original `bw_func_net` / `ba_func_net`, train `S00/S01/S02/S03/S05/S07`, test `S13/S15/S20/S21/S22/S23`, `loss_window=64`, `dt=0.001`, and per-sequence quality-window filtering from the exported aggregate dataset.
- `plato/configs/debias_imu/corldata_imu_aligned_gt_baseline_expanded_train_w128_e1801.yaml`: strict original-style corldata `IMU-aligned-gt` baseline config using `loss_window=128`, `epoch=1801`, `save_freq=100`, last-checkpoint testing, and `valid_window_starts_128.txt` filtering.
- `plato/configs/debias_imu/corldata_sparse_endpoint_no_udot_bg64_ba128_e201_s13s15_best.yaml`: split-window sparse-endpoint no-`u_dot` experiment config. It records gyro `b_g` training with window `64`, accel `b_a` training with window `128`, train `S00/S01/S02/S03`, test `S13/S15`, separate caches, and best-validation checkpoint testing.
- `plato/configs/debias_imu/corldata_sparse_endpoint_no_udot_w128_sg101_initmedian5_e301_s13s15.yaml`: previous best SE23/MSE corldata GT-supervised sparse-endpoint no-`u_dot` config on S13/S15 and the starting mean for later interval-NLL refinement. It keeps the all-128 setup, increases gyro pseudo-bias SG smoothing to `101`, and uses `median_window=5s` for test initial bias.
- `plato/configs/debias_imu/corldata_sparse_endpoint_no_udot_w128_sg101_initmedian5_acc601_from_e301.yaml`: prepared Acc-only SE23 continuation config. It fixes the current best gyro, initializes Acc from the current best e301 Acc checkpoint, and adds another 301 lower-learning-rate Acc epochs under a separate output directory.
- `plato/configs/debias_imu/corldata_sparse_endpoint_no_udot_w128_sg101_initmedian5_e301_s13s15_nll_optq.yaml`: objective-only likelihood ablation from the current best corldata setup. It keeps the same no-`u_dot` network, train/test split, window, SG smoothing, and initial-bias policy, loads the current best gyro checkpoint, skips gyro retraining, and changes only the Acc objective to per-state SE_2(3) Gaussian NLL with alternating process-covariance adjustment.
- `plato/configs/debias_imu/corldata_sparse_endpoint_no_udot_w128_sg101_initmedian5_e301_s13s15_nll_fixed.yaml`: fixed-covariance version of the same objective-only NLL ablation, useful for separating likelihood weighting from covariance adaptation.
- `plato/configs/debias_imu/corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_nll_endpoint_qonly.yaml`: first robust paper-style likelihood stage. It keeps the current best SE23 gyro and Acc mean checkpoints fixed, uses a longer 256-pose sparse-GT window, and learns only endpoint-likelihood `n_a/n_g` covariance.
- `plato/configs/debias_imu/corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_nll_endpoint_finetune.yaml`: PLATO paper-style likelihood refinement config. It loads the current best SE23 gyro and Acc checkpoints, skips gyro retraining, uses a longer 256-pose sparse-GT window, and refines only the Acc bias dynamics plus `n_a/n_g` via endpoint SE_2(3) Gaussian NLL.
- `plato/configs/debias_imu/corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_nll_endpoint_joint_lr5e5_from_qonly.yaml`: constrained joint bg/ba endpoint-NLL follow-up from q-only covariance. It uses default-off runner support for `--nll-train-gyro`; validation NLL improved, but S13/S15 full-test regressed, so it is a negative ablation.
- `plato/configs/debias_imu/corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_nll_endpoint_acconly_anchor_lr1e5_from_qonly.yaml`: safer Acc-only endpoint-NLL follow-up using q-only covariance, fixed gyro, tiny Acc learning rate, and the runner's mean-anchor options. It preserved but did not improve the current SE23 mean, so it is not the current best.
- `plato/configs/debias_imu/corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_interval_nll_covonly_from_e301.yaml`: interval-level preintegration NLL covariance-only stage. It freezes the current SE23 e301 mean model and learns only scalar IMU measurement-noise variances `sigma_g2/sigma_a2` for reset-per-interval preintegration factors.
- `plato/configs/debias_imu/corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_interval_nll_joint_lr1e5_from_covonly.yaml`: interval-level preintegration NLL joint fine-tune stage. It initializes from the interval cov-only checkpoint, loads its raw variance parameters, and cautiously fine-tunes bg/ba dynamics plus scalar IMU variances with detached covariance-theta gradients.
- `plato/configs/debias_imu/corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_interval_nll_covonly_continue_from_covonly.yaml`: continued interval-NLL covariance-only calibration. It loads the first cov-only raw variances and extends scalar IMU measurement-noise learning before any mean update.
- `plato/configs/debias_imu/corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_interval_nll_joint_lr1e5_from_covonly_continue.yaml`: current best corldata S13/S15 interval-NLL model. It initializes mean and raw variances from the continued cov-only checkpoint, then low-LR jointly fine-tunes bg/ba dynamics plus scalar IMU variances.
- `plato/configs/debias_imu/corldata_vo_supervision_no_udot_w40_e301_s13s15_best.yaml`: corldata VO-supervision check matching the current best no-`u_dot` model family and train/test split. It uses `vo_imu_noninterp`, `vo_supervision_world_frame=vo`, `loss_window=40` VO poses to match about a 1.03 s physical window, train `S00/S01/S02/S03`, and test `S13/S15`.
- `plato/configs/debias_imu/corldata_vo_supervision_no_udot_w40_no_vel_e301_s13s15_best.yaml`: Acc no-velocity ablation for the same corldata VO-supervision setup. It keeps the prior gyro checkpoint/settings and changes only Acc loss weights to rotation `1`, velocity `0`, position `1`.
- `plato/docs/current_experiment_state.md`: compact handoff for the current corldata Debias_IMU/likelihood thread, including current best checkpoints, negative likelihood results, q-only calibration, completed Acc-extension/NLL follow-ups, interval-level preintegration NLL results, and the current recommendation to use the interval-NLL joint model for S13/S15.
- `plato/scripts/train_bias_dynamics/estimate_liasuw_gravity_bias.py`: estimates per-sequence LIASuw gravity acceleration vectors and constant accelerometer bias from raw IMU plus body GT for Route-A gravity-aligned Debias_IMU supervision.
- `plato/scripts/train_bias_dynamics/run_euroc_debias_phys_delta_pipeline.py`: PLATO-owned EuRoC Debias_IMU runner for testing PLATO bias-dynamics networks, including `bw_func_phys_delta` and `ba_func_phys_delta`, without modifying `external/Debias_IMU`. It reuses the external `EUROCDataset`, losses, and evaluation functions while keeping outputs under `results/debias_imu/`.
- `plato/scripts/train_bias_dynamics/verify_liasuw_supervision_frames.py`: verifies that aggregate Debias_IMU `mav0/mocap0/data.csv` files exactly match per-sequence `body_gt_world.csv`, confirming that LIASuw supervision uses world-frame IMU/body GT rather than camera, marker, or first-frame-normalized poses.
- `plato/scripts/train_bias_dynamics/compare_vo_gt_world_frames.py`: compares per-sequence `vo_imu_traj.tum` VO-world IMU poses with mocap/world-frame `body_gt_world.csv`, estimates the fixed `R_Wm_Wv` world rotation, and reports the gravity direction that Debias_IMU would need if training directly in the VO world frame.
- `plato/scripts/train_bias_dynamics/verify_liasuw_vo_supervision_frames.py`: verifies cached source-timing VO supervision poses are stored in mocap/GT world (`world_frame=gt`) and match `body_gt_world.csv` at VO timestamps to the expected calibration residual level.
- `plato/scripts/openvins_eval/export_liasuw_openvins_inputs.py`: exports LIASuw raw, constant-bias-corrected, and network-debiased IMU CSVs for the paper-style OpenVINS clean-IMU route. It reads Debias_IMU `FullTra_result.p`, preserves original high-rate LIASuw IMU timestamps as integers, interpolates learned bias to the raw IMU timestamps, and writes validation metadata under `results/openvins_eval/`.
- `plato/scripts/openvins_eval/make_liasuw_openvins_bags.py`: builds OpenVINS-readable ROS bags from the exported IMU CSVs and existing stereo image folders. It writes `/imu0` or dataset-specific IMU topics plus stereo image topics, crops the final image if `t_cam + timeshift_cam_imu` has no IMU coverage, records bag counts in `bag_validation.json`, and supports optional light-segment image dropping via `--light-segments`, `--sequence`, and `--drop-image-states`.
- `plato/scripts/openvins_eval/make_visual_dropout_bags.py`: creates artificial visual-failure bags by removing stereo image messages over a selected time interval while preserving the full IMU stream.
- `plato/scripts/openvins_eval/run_openvins_dropout_trial.sh`: runs one GT-initialized OpenVINS visual-dropout trial and records `/ros1_serial_msckf/odomimu` for propagation-focused evaluation.
- `plato/scripts/openvins_eval/summarize_liasuw_openvins_dropout.py`: compares dropout `odomimu` propagation output against LIASuw body GT and writes JSON/Markdown summaries.
- `plato/scripts/openvins_eval/check_liasuw_bias_consistency.py`: runs a standalone LIASuw bias consistency check by propagating raw IMU from the same GT initial pose/velocity with several bias sources: zero, OpenVINS dynamic-init bias, Debias_IMU constant/network bias, and direct GT-differenced pseudo-bias.
- `plato/scripts/openvins_eval/run_corldata_openvins_natural_trial.sh`: runs raw self-initialized OpenVINS on original corldata bags such as S13/S15, starts a ROS master if needed, and saves OpenVINS total-state estimate/std/timing/log files under `results/openvins_eval/corldata/natural/raw/<seq>/`.
- `plato/scripts/openvins_eval/summarize_corldata_openvins_natural.py`: evaluates corldata natural-light OpenVINS runs against `IMU-aligned-gt` with SE(3) no-scale alignment, splitting metrics into all/light-on/light-off, post-dark light-on, last-light-on, re-entry, and per detected natural light interval.
- `plato/scripts/openvins_eval/export_corldata_openvins_corrected_inputs.py`: exports corldata OpenVINS corrected-IMU CSVs from Debias_IMU `FullTra_result.p` outputs by subtracting predicted gyro/accel bias from the original high-rate `sequence/imu.txt`; supports raw, constant, full-network, gyro-only, and accel-only diagnostic streams without passing GT into OpenVINS.
- `plato/scripts/openvins_eval/export_corldata_openvins_gated_residual_inputs.py`: exports corldata light-off-only gated residual IMU correction streams for OpenVINS diagnostics. It can use absolute learned bias relative to raw OpenVINS bias references or learned bias delta from each light-off start, supports duration-aware gyro gain selection for short vs long dark intervals, and includes an optional correction-tail diagnostic without passing GT states into OpenVINS.
- `plato/scripts/openvins_eval/compare_corldata_openvins_natural_runs.py`: compares multiple `summary.json` files produced by `summarize_corldata_openvins_natural.py` and writes first-light-on/all-aligned Markdown and CSV tables for raw, corrected, and gated-residual OpenVINS runs.
- `plato/scripts/openvins_eval/compare_corldata_openvins_common_horizon.py`: compares corldata OpenVINS runs at the earliest shared final timestamp per sequence, avoiding misleading final-error comparisons when one run terminates earlier.
- `plato/scripts/openvins_eval/plot_corldata_openvins_reentry_trajectory.py`: plots raw, duration-aware learned-gyro, and re-entry-stabilized corldata OpenVINS trajectories against IMU-aligned GT, with light-off spans and position-error curves. Used for S13/S15 re-entry visualization under `results/openvins_eval/corldata/natural/gated_gyro_ref_duration_reentry_adaptive_gray_runs/plots/`.
- `plato/scripts/openvins_eval/evaluate_corldata_light_off_preintegration.py`: OpenVINS-free dark-interval preintegration evaluator. It initializes each natural corldata light-off interval from GT `R/v/p`, integrates raw IMU with learned/full/gyro-only/accel-only/constant/delta bias corrections from a Debias_IMU `FullTra_result.p`, and reports trajectory and endpoint errors against `IMU-aligned-gt`.
- `plato/configs/openvins/corldata_0511/estimator_config_dynamic_freeze_bias.yaml`: corldata dynamic-init OpenVINS config variant for corrected-IMU tests with internal OpenVINS IMU bias frozen at zero.
- `plato/configs/openvins/corldata_0511/estimator_config_dynamic_corrected_init.yaml`: corldata dynamic-init OpenVINS config variant for corrected-IMU streams that lowers `init_dyn_min_deg` to `1.0` while keeping normal online bias estimation.
- `plato/configs/openvins/corldata_0511/estimator_config_dynamic_corrected_init_freeze_bias.yaml`: combines the corrected-IMU dynamic-init threshold with frozen zero internal bias.
- `plato/configs/openvins/corldata_0511/estimator_config_dynamic_reentry.yaml`: corldata dynamic-init OpenVINS config with the opt-in PLATO dark-to-light re-entry hook enabled for natural-light S13/S15 tests. It keeps the natural camera stream, clears stale tracks on confirmed re-entry, inflates covariance, and uses adaptive light confirmation for long or gray-dark intervals.

Documentation:

- `plato/docs/vo_supervision_for_debias_imu.md`: records the correct VO supervision convention for Debias_IMU: use `vo_imu_traj.tum` as `T_Wv_I`, apply per-sequence `calibration_result.yaml` `T_WO`/`T_Wm_Wv`, and train with `--vo-supervision-world-frame gt`.
- `plato/docs/liasuw_routeA_gravity_alignment.md`: records the Route-A gravity calibration/alignment implementation, the corrected `phys_delta` training/test run, the invalid pseudo-bias-only intermediate, and the conclusion that Route-A improves constant-bias propagation but not the learned held-out model.
- `plato/docs/euroc_phys_delta_ablation.md`: records the EuRoC GT-supervised `phys_delta` feasibility ablation, including configuration, supervision source, training behavior, held-out metrics, and comparison with existing `Euroc_SE23_w16` and `Euroc_baseline_w16`.
- `plato/docs/liasuw_openvins_clean_imu_route.md`: records the LIASuw implementation of the Debiasing paper's OpenVINS route, including export commands, output paths, validation numbers, OpenVINS config location, bag counts, and the current moved-workspace OpenVINS runtime note.
- `plato/docs/corldata_openvins_learned_imu_lessons.md`: operational memory for corldata S13/S15 self-initialized OpenVINS learned-IMU experiments. It records the current best duration-aware gyro-ref plus adaptive re-entry policy, known-negative attempts, and the cleanup of deleted generated artifacts.
- `plato/scripts/train_bias_dynamics/plot_liasuw_bias_preintegration_window.py`: plots a self-contained draggable 3D HTML check for a short window propagated with raw IMU plus differenced pseudo-bias GT and GT initial pose/velocity.
- `plato/scripts/train_bias_dynamics/check_corldata_preintegration.py`: checks corldata IMU/body trajectory consistency by deriving pseudo-biases from exported body trajectories, propagating high-rate shifted IMU over selected local windows, and comparing raw, constant-bias, and pseudo-bias propagation. It supports `--pose-source vo_tum`, `gt_tum`, or `mocap_csv`; the direct GT/body CSV check writes summaries under `results/debias_imu/corldata_gt_body_1000hz_preintegration_checks/`.
- `plato/scripts/train_bias_dynamics/analyze_corldata_10hz_bias_distribution.py`: derives 10 Hz GT-differenced `b_g`/`b_a` diagnostics from corldata `IMU-aligned-gt`, filters bins using exported `valid_window_starts_<N>.txt`, and writes train/test bias CSVs, per-sequence six-axis plots, distribution overlays, and similarity summaries.
- `plato/scripts/train_bias_dynamics/analyze_corldata_motion_distribution.py`: derives 10 Hz GT-differenced motion diagnostics from corldata `IMU-aligned-gt`, including body-frame GT angular velocity, dynamic acceleration, raw IMU measurement summaries, train/test motion-distribution distances, and light-on/light-off motion similarity summaries.
- `plato/scripts/orbslam3_export/estimate_corldata_body0_imu_extrinsics.py`: estimates and records per-sequence corldata `Body_0` mocap marker-to-IMU/body SE(3) extrinsics from raw ORB-SLAM3 VO references plus the physical DAVIS camera-IMU calibration. Outputs are written under `results/pseudo_gt/corldata/body0_imu_extrinsics_vo_ref/` and mirrored per sequence at `results/pseudo_gt/corldata/<seq>/alignment_config/body0_imu_extrinsic_vo_ref.json`.
- `plato/scripts/export_corldata_imu_aligned_gt_from_extrinsics.py`: applies the recorded per-sequence `T_Body0_IMU` extrinsics to mocap `Body_0.tum`, gravity-aligns the result, and exports fixed-1000 Hz `IMU-aligned-gt` trajectories plus a Debias_IMU-style aggregate tree under `results/debias_imu/corldata_IMU_aligned_gt_vo_ref_1000hz/`.
- `plato/scripts/0203_pipeline_common.py`: shared LIASuw 0203/0205 scan/evaluation/conversion helpers. Its camera-IMU extrinsic parser is also reused by `export_liasuw_debias_body_gt.py`; for 0912 `cam-imu.yaml`, it now reads only the first matrix block after `T_ci`/`T_ic`.

## ORB-SLAM3

Path: `external/ORB_SLAM3`

Main purpose: real-time visual, visual-inertial, and multi-map SLAM for monocular, stereo, and RGB-D cameras. For PLATO, the intended use is stereo visual odometry trajectory generation on visually favorable data.

Evidence:
- `external/ORB_SLAM3/README.md`
- `external/ORB_SLAM3/CMakeLists.txt`

Important directories:
- `external/ORB_SLAM3/src`: core implementation including `System.cc`, `Tracking.cc`, `Optimizer.cc`, and `ImuTypes.cc`.
- `external/ORB_SLAM3/include`: public headers including `System.h`, `Tracking.h`, and `ImuTypes.h`.
- `external/ORB_SLAM3/Examples/Stereo`: stereo examples and configs, including `stereo_euroc.cc`, `stereo_tum_vi.cc`, `stereo_tum_vi_batch.cc`, `EuRoC.yaml`, `TUM-VI.yaml`, and local LIASuw configs.
- `external/ORB_SLAM3/Examples/Stereo-Inertial`: stereo-inertial examples and configs.
- `external/ORB_SLAM3/tools`: local Python utilities for LIASuw, EuRoC conversion, alignment, plotting, and batch stereo runs.
- `external/ORB_SLAM3/evaluation`: trajectory evaluation helpers.
- `external/ORB_SLAM3/Thirdparty`: DBoW2, g2o, Sophus, and local Pangolin install artifacts.

Build system:
- CMake project in `external/ORB_SLAM3/CMakeLists.txt`.
- `external/ORB_SLAM3/build.sh` builds third-party libraries and examples.
- `external/ORB_SLAM3/build_ros.sh` is present for optional ROS examples.
- CMake requires C++11, OpenCV 4.2 in this local CMake file, Eigen3, Pangolin, Boost serialization, and crypto.

Important executable or example entry points:
- `external/ORB_SLAM3/Examples/Stereo/stereo_euroc.cc`
- `external/ORB_SLAM3/Examples/Stereo/stereo_tum_vi.cc`
- `external/ORB_SLAM3/Examples/Stereo/stereo_tum_vi_batch.cc`
- `external/ORB_SLAM3/Examples/Stereo/stereo_kitti.cc`
- `external/ORB_SLAM3/Examples/Stereo-Inertial/stereo_inertial_euroc.cc`
- Built binaries already exist beside some example sources.

Configuration files:
- `external/ORB_SLAM3/Examples/Stereo/EuRoC.yaml`
- `external/ORB_SLAM3/Examples/Stereo/TUM-VI.yaml`
- `external/ORB_SLAM3/Examples/Stereo/LIASuw.yaml`
- `external/ORB_SLAM3/Examples/Stereo/LIASuw_image.yaml`
- `external/ORB_SLAM3/Examples/Stereo/LIASuw_image_robust.yaml`
- `plato/configs/orbslam3/LIASuw_0205_image_baseline.yaml`: PLATO-owned LIASuw 0203/0205 DAVIS APS stereo baseline converted from `LIASuw_Training/esio.yaml` and `event0/1_esio.yaml`.
- `plato/configs/orbslam3/LIASuw_0205_image_nf2000_far8_depth30.yaml`: 0203/0205 calibration with feature/depth settings mirroring the accepted 0912 stereo tuning variant.
- `plato/configs/orbslam3/corldata_davis_liasuw_0205_baseline.yaml`: PLATO corldata S00 stereo VO config that reuses the LIASuw 0203/0205 DAVIS APS stereo calibration and sets `Camera.fps: 38`.
- `plato/configs/orbslam3/corldata_davis_stereo_inertial.yaml`: PLATO corldata DAVIS stereo-inertial config using corrected corldata `cam0.yaml`/`cam1.yaml`, `uw_robot.yaml` left DAVIS-to-IMU extrinsic and IMU noise, `Camera.fps: 38`, and bold Pangolin viewer line widths for visual checks. Runners must shift IMU timestamps to the image clock before invoking ORB-SLAM3.
- `external/ORB_SLAM3/Examples/Stereo-Inertial/EuRoC.yaml`
- `external/ORB_SLAM3/Examples/Stereo-Inertial/LIASuw.yaml`
- `plato/configs/orbslam3/LIASuw_image_inertial_0912_camimu_nf2000_far8_depth30_imu2x.yaml` and `_imu4x.yaml`: PLATO S07 stereo-inertial diagnostic configs that increase IMU residual weight by scaling IMU noise/random-walk down by `1/sqrt(2)` and `0.5`.

Expected inputs:
- Vocabulary file: `external/ORB_SLAM3/Vocabulary/ORBvoc.txt`.
- Stereo image folders and timestamp files for examples such as `stereo_euroc.cc` and `stereo_tum_vi.cc`.
- Optional IMU files for stereo-inertial examples.

Expected outputs and trajectory formats:
- `System.h` declares `SaveTrajectoryTUM`, `SaveKeyFrameTrajectoryTUM`, `SaveTrajectoryEuRoC`, `SaveKeyFrameTrajectoryEuRoC`, and `SaveTrajectoryKITTI`.
- `System.cc` implements these trajectory writers.
- Stereo examples write `f_<trajectory_name>.txt` and `kf_<trajectory_name>.txt` when a trajectory name argument is provided; otherwise they write `CameraTrajectory.txt` and `KeyFrameTrajectory.txt`.
- Existing trajectory-like files are present: `external/ORB_SLAM3/f_liasuw_s01_300_stereo.txt`, `external/ORB_SLAM3/kf_liasuw_s01_300_stereo.txt`, `external/ORB_SLAM3/f_stereo_0203_s10_robust.txt`, and `external/ORB_SLAM3/kf_stereo_0203_s10_robust.txt`.
- Local tools mention ORB-SLAM3 `SaveTrajectoryEuRoC` text files and converters to EuRoC-like CSV or TUM formats.

Connection to PLATO:
- PLATO should use `external/ORB_SLAM3/Examples/Stereo` and existing trajectory writers to generate accurate stereo VO trajectories.
- PLATO-owned wrappers should live in `plato/scripts/orbslam3_export`, not in `external/ORB_SLAM3/tools`, unless a future task explicitly requires changing the external repo.

Uncertain:
- Exact frame convention for each local LIASuw trajectory must be verified before using it as bias supervision.
- Some local tools remove or move `f_*` and `kf_*` files during batch processing; future PLATO wrappers should avoid depending on side effects without inspection.

## Debias_IMU / rss_debiasing_restart

Path: `external/Debias_IMU/rss_debiasing_restart`

Main purpose: research code for learning continuous IMU bias dynamics with Neural ODEs. The nested `Debias_IMU` README describes debiasing gyroscope and accelerometer data, modeling bias dynamics with a neural network, and training via a Neural ODE on SO(3) using orientation, velocity, and position supervision without requiring ground-truth bias.

Evidence:
- `external/Debias_IMU/rss_debiasing_restart/README.md`
- `external/Debias_IMU/rss_debiasing_restart/Debias_IMU/README.md`
- `external/Debias_IMU/rss_debiasing_restart/Debias_IMU/plato_extensions/README.md`
- `external/Debias_IMU/rss_debiasing_restart/Debias_IMU/plato_extensions/docs/baseline_modification_record.md`
- `external/Debias_IMU/rss_debiasing_restart/Debias_IMU/plato_extensions/docs/project_memory/START_HERE.md`
- `external/Debias_IMU/rss_debiasing_restart/Debias_IMU/plato_extensions/docs/project_memory/code_review.md`

Important directories:
- `external/Debias_IMU/rss_debiasing_restart/Debias_IMU/BiasDy`: core training, model, loss, dataset, and analytic-gradient code.
- `external/Debias_IMU/rss_debiasing_restart/Debias_IMU/SO3diffeq`: SO(3) ODE solver code.
- `external/Debias_IMU/rss_debiasing_restart/Debias_IMU/Third_party`: vendored `torchdiffeq` and `controldiffeq`.
- `external/Debias_IMU/rss_debiasing_restart/Debias_IMU/data`: EuRoC-style dataset folders and cache locations.
- `external/Debias_IMU/rss_debiasing_restart/Debias_IMU/results`: many existing experiment outputs.
- `external/Debias_IMU/rss_debiasing_restart/Debias_IMU/plato_extensions`: PLATO-specific Debias_IMU diagnostics, LIASuw analysis tools, local shell helpers, and prior project notes separated from the baseline method tree.
- `external/Debias_IMU/rss_debiasing_restart/tools`: compatibility wrappers for LIASuw pseudo-bias and correlation analysis scripts now stored under `Debias_IMU/plato_extensions/tools/liasuw_bias`.

Important training or evaluation scripts:
- `external/Debias_IMU/rss_debiasing_restart/Debias_IMU/BiasDy/mainEuroc.py`: baseline EuRoC training and testing.
- `external/Debias_IMU/rss_debiasing_restart/Debias_IMU/BiasDy/mainEuroc_SE23.py`: SE_2(3) entry point with GT/VO supervision options, analytic loss options, and NLL options.
- `external/Debias_IMU/rss_debiasing_restart/Debias_IMU/BiasDy/mainTUM.py`
- `external/Debias_IMU/rss_debiasing_restart/Debias_IMU/BiasDy/mainFetch.py`
- `external/Debias_IMU/rss_debiasing_restart/Debias_IMU/plato_extensions/biasdy_diagnostics`: standalone diagnostics and diagnostic baselines moved out of `BiasDy`; old `BiasDy/*.py` diagnostic paths are wrappers.
- `external/Debias_IMU/rss_debiasing_restart/Debias_IMU/plato_extensions/tools/liasuw_bias`: LIASuw pseudo-bias plotting, gravity-correlation, and outlier-correlation tools.
- `external/Debias_IMU/rss_debiasing_restart/Debias_IMU/prepare_euroc_vo_pseudogt.py`
- `external/Debias_IMU/rss_debiasing_restart/Debias_IMU/export_openvins_imu.py`
- `external/Debias_IMU/rss_debiasing_restart/Debias_IMU/evo_evaluation/evo_results.py`

Model definitions and bias modules:
- `external/Debias_IMU/rss_debiasing_restart/Debias_IMU/BiasDy/network_for_BiadDy.py` defines `bw_func_net`, `bw_func_net_no_wdot`, `ba_func_net`, and `VFSE23vbiasBase`.
- `bw_func_net` consumes `[bw, w, w_dot]` in the current local version.
- `bw_func_net_no_wdot` is a gyro-bias ablation using `[bw, w]`.
- `ba_func_net` consumes `[ba, a, a_dot]`.
- `VFSE23vbiasBase` integrates a state containing rotation coordinates, time, gyro bias, velocity, position, and accel bias, and computes `bw_dot` and `ba_dot`.

Configuration and dependency files:
- `external/Debias_IMU/rss_debiasing_restart/Debias_IMU/Requirments.txt` is the dependency file referenced by the README; the filename is misspelled in the repository.
- No `setup.py` or top-level CMake build file was found for this Python research project.

Dataset assumptions:
- The README references EuRoC and TUM-VI datasets.
- `mainEuroc.py` uses `data/EUROC` and fixed train/test sequences.
- `mainEuroc_SE23.py` supports `--train_supervision gt` and `--train_supervision vo`, uses `dataset/Euroc_VO` for VO supervision, and uses cache directories such as `data/EUROC_cache_gt`, `data/EUROC_cache_vo_body`, and `data/EUROC_cache_test_gt`.
- Local LIASuw tools expect ROS bags and mocap-derived ground truth paths.

Bias-related outputs:
- Existing result folders contain `Acc_parameters.yaml`, `Gyro_parameters.yaml`, `Acc_weights`, `Gyro_weights`, and bias plots or CSVs in local LIASuw output folders.
- LIASuw pseudo-bias scripts write columns such as `bg_x_rad_s`, `bg_y_rad_s`, `bg_z_rad_s`, `ba_x_m_s2`, `ba_y_m_s2`, and `ba_z_m_s2`.

Connection to PLATO:
- PLATO should reuse the Neural ODE bias-dynamics implementation and training structure, but drive it from ORB-SLAM3 trajectory supervision prepared under `plato/scripts/train_bias_dynamics`.
- Existing `mainEuroc_SE23.py --train_supervision vo` is the closest local evidence of VO-supervised training, but PLATO still needs a clean conversion and orchestration layer.
- For fair Debias_IMU baseline reporting, use `mainEuroc.py`, `mainTUM.py`, or `mainFetch.py` with original-style model/loss/sampling choices. `mainEuroc_SE23.py`, analytic/NLL losses, VO supervision, and PLATO-owned runners are extensions or ablations, not the unmodified upstream method.

Uncertain:
- Some moved project-memory notes under `Debias_IMU/plato_extensions/docs/project_memory` still reference old absolute paths under `/home/amao/workspace/debiasingIMU/rss_debiasing_restart`.
- The exact compatibility of ORB-SLAM3 exported trajectories with `dataset/Euroc_VO` must be verified.
- Existing training/testing uses initial bias from pseudo-GT in some paths; this matters for PLATO claims and should be made explicit in experiments.

## OpenVINS

Path: `external/OpenVINS`

Main purpose: filter-based visual-inertial estimation. For PLATO, OpenVINS is the downstream VIO evaluation system for learned or corrected bias dynamics.

Evidence:
- `external/OpenVINS/src/open_vins/ReadMe.md`
- `external/OpenVINS/src/open_vins/ov_msckf/package.xml`
- `external/OpenVINS/src/open_vins/ov_msckf/CMakeLists.txt`

Important directories:
- `external/OpenVINS/src/open_vins/ov_core`: core vision, camera, feature, IMU type, simulation, and continuous preintegration code.
- `external/OpenVINS/src/open_vins/ov_msckf`: MSCKF estimator, propagation, updates, ROS launch files, simulation, and VIO manager.
- `external/OpenVINS/src/open_vins/ov_init`: static and dynamic initialization.
- `external/OpenVINS/src/open_vins/ov_eval`: trajectory recording, alignment, plotting, timing, and evaluation utilities.
- `external/OpenVINS/src/open_vins/config`: dataset configs for EuRoC, TUM-VI, KAIST, UZH-FPV, RealSense, and RPNG datasets.

Build system:
- This was moved as a catkin workspace: `external/OpenVINS` contains `src`, `build`, `devel`, `logs`, and `.catkin_tools`.
- Package build files exist under `external/OpenVINS/src/open_vins/ov_core/CMakeLists.txt`, `ov_init/CMakeLists.txt`, `ov_msckf/CMakeLists.txt`, `ov_eval/CMakeLists.txt`, and `ov_data/CMakeLists.txt`.
- `ov_msckf/package.xml` declares version `2.7.0`, GPLv3, ROS1/ROS2 dependencies, Eigen, OpenCV, Boost, and Ceres.
- `ov_msckf/CMakeLists.txt` uses C++14 and supports ROS1, ROS2, or ROS-free builds.

Important launch/config files:
- `external/OpenVINS/src/open_vins/ov_msckf/launch/subscribe.launch`: runs `run_subscribe_msckf`, optionally plays a bag, records `/ov_msckf/poseimu` via `ov_eval/pose_to_file`, and can run live trajectory alignment.
- `external/OpenVINS/src/open_vins/ov_msckf/launch/serial.launch`
- `external/OpenVINS/src/open_vins/ov_msckf/launch/simulation.launch`
- `external/OpenVINS/src/open_vins/config/euroc_mav/estimator_config.yaml`
- `external/OpenVINS/src/open_vins/config/euroc_mav/kalibr_imu_chain.yaml`
- `external/OpenVINS/src/open_vins/config/euroc_mav/kalibr_imucam_chain.yaml`
- `plato/configs/openvins/corldata_0511/kalibr_imucam_chain.yaml`: corldata 0511 DAVIS stereo camera-to-IMU calibration, reusing the user-confirmed LIASuw 0203/0205 DAVIS APS extrinsics/intrinsics with corldata topics `/davis_left/image_raw` and `/davis_right/image_raw`.
- `plato/configs/openvins/corldata_0511/kalibr_imu_chain.yaml`: corldata left-DAVIS IMU topic `/davis_left/imu`, with LIASuw-derived initial noise parameters.
- `plato/configs/openvins/corldata_0511/estimator_config.yaml` and `estimator_config_dynamic.yaml`: OpenVINS serial-bag configs for corldata; the dynamic version is the usable moving-start config verified on S23.
- `plato/configs/openvins/corldata_0511/estimator_config_dynamic_reentry.yaml`: opt-in corldata dynamic config for the PLATO S13/S15 re-entry stabilization hook. Defaults in OpenVINS keep the hook disabled unless this config or equivalent params enable it.

Estimator-related modules:
- `external/OpenVINS/src/open_vins/ov_msckf/src/core/VioManager.cpp`
- `external/OpenVINS/src/open_vins/ov_msckf/src/core/VioManagerOptions.h`
- `external/OpenVINS/src/open_vins/ov_msckf/src/state/State.cpp`
- `external/OpenVINS/src/open_vins/ov_msckf/src/state/StateHelper.cpp`
- `external/OpenVINS/src/open_vins/ov_msckf/src/update/UpdaterMSCKF.cpp`
- `external/OpenVINS/src/open_vins/ov_msckf/src/update/UpdaterSLAM.cpp`
- `external/OpenVINS/src/open_vins/ov_msckf/src/update/UpdaterZeroVelocity.cpp`
- PLATO re-entry modifications currently touch `ov_core/src/track/TrackBase.{h,cpp}`, `ov_core/src/track/TrackKLT.{h,cpp}`, `ov_msckf/src/core/VioManagerOptions.h`, `VioManager.h`, `VioManager.cpp`, and `ov_msckf/src/state/StateHelper.{h,cpp}`. These add resettable visual tracks and opt-in covariance inflation for confirmed dark-to-light transitions; original behavior is preserved when `plato_reentry_enable=false`.

IMU propagation and bias-related modules:
- `external/OpenVINS/src/open_vins/ov_msckf/src/state/Propagator.h` and `Propagator.cpp` perform mean and covariance propagation using IMU measurements.
- `external/OpenVINS/src/open_vins/ov_core/src/types/IMU.h` defines the IMU state with pose, velocity, gyroscope bias `bg`, and accelerometer bias `ba`.
- `external/OpenVINS/src/open_vins/ov_core/src/cpi/CpiBase.h`, `CpiV1.h`, and `CpiV2.h` implement continuous preintegration interfaces and Jacobians around gyro and accel bias linearization points.
- `external/OpenVINS/src/open_vins/config/euroc_mav/kalibr_imu_chain.yaml` contains IMU noise density and random-walk parameters.
- `external/OpenVINS/src/open_vins/config/euroc_mav/estimator_config.yaml` contains integration mode, initialization bias guesses `init_dyn_bias_g` and `init_dyn_bias_a`, and related bias covariance inflation settings.

Expected inputs:
- ROS image and IMU topics from bags or live streams; `subscribe.launch` defaults to EuRoC-style config and a bag path argument.
- Dataset-specific camera and IMU calibration files under `external/OpenVINS/src/open_vins/config`.

Expected outputs:
- `subscribe.launch` can record estimated trajectory to `path_est`, default `/tmp/traj_estimate.txt`, using `ov_eval` `pose_to_file`.
- `estimator_config.yaml` includes optional state and covariance output paths such as `/tmp/ov_estimate.txt`, `/tmp/ov_estimate_std.txt`, and `/tmp/ov_groundtruth.txt`.

Connection to PLATO:
- OpenVINS is the target evaluation environment for learned bias information.
- A conservative first PLATO step is to evaluate learned bias information offline or through config/preprocessed IMU streams before modifying `Propagator.cpp`.
- If source modification becomes necessary, likely touch points are `ov_msckf/src/state/Propagator.*`, `ov_core/src/types/IMU.h`, config parsing in `VioManagerOptions.h`, and ROS input/evaluation wrappers.

Uncertain:
- Moved `build`, `devel`, and `.catkin_tools` artifacts may contain old absolute paths and should be regenerated before serious OpenVINS runs.
- The best injection point for learned bias dynamics is not selected yet.
