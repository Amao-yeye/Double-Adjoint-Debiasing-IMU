# Integration Plan

Last updated: 2026-05-13

This document is the current high-level plan. Detailed experiment history remains in `plato/docs/project_memory.md`; pseudo-groundtruth acceptance decisions remain in `plato/docs/pseudo_groundtruth_registry.md`.

## Phase 0: Codebase Organization And Documentation

Goal: preserve the external projects under one PLATO workspace and keep the project memory accurate.

Status: implemented. ORB-SLAM3, Debias_IMU, OpenVINS, and the paper submodule are organized under the PLATO workspace. PLATO-owned scripts, configs, and docs live under `plato/`; generated outputs live under ignored `results/`.

Current cautions:

- External repositories are not pristine upstream checkouts and contain local modifications or generated artifacts.
- Do not modify external repositories unless explicitly required; if that happens, document the exact files in `AGENTS.md` and `plato/docs/project_memory.md`.
- Keep large datasets outside git under `/home/amao/datasets`.

## Phase 1: ORB-SLAM3 Trajectory Export

Goal: produce reliable stereo VO trajectory files from visually favorable LIASuw windows.

Status: implemented for the current LIASuw candidate search. The main runner is `plato/scripts/orbslam3_export/run_liasuw_orbslam3_trial.py`; PLATO-owned ORB-SLAM3 configs are under `plato/configs/orbslam3/`.

Current pseudo-groundtruth state:

- User-confirmed usable stereo-only windows: `0912s06_80`, `0912s07_30`, and `0912s10_40`.
- Usable-quality candidates pending formal admission to the training set: `0203s18_80`, `0203s19_80`, `0203s22_80`, and `0205s14_80`.
- `0912s15` and the rejected full-window or failed diagnostic runs listed in `plato/docs/pseudo_groundtruth_registry.md` must not be used for training.

Current cautions:

- `0203/0205` use different image calibration from `0912`; do not reuse the accepted `0912` image config for those sessions.
- Stereo-inertial diagnostic outputs were comparison runs, not selected pseudo-groundtruth.
- Future candidate selection should prioritize local quality filtering and sequence/window selection over repeated ORB extractor/depth-threshold sweeps.

## Phase 2: Trajectory Alignment And Supervisory Conversion

Goal: align ORB-SLAM3 trajectory output to dataset time/frame conventions and convert it into privileged IMU/body supervision.

Status: implemented for the current LIASuw 200 Hz export. `plato/scripts/export_liasuw_debias_body_gt.py` exports per-sequence artifacts under `results/pseudo_gt/LIASuw/<sequence>/debias_imu_body_gt_200hz/` and builds the aggregate Debias_IMU-style tree at `results/debias_imu/LIASuw_training_candidates_200hz/`.

Current convention:

```text
T_Wv_C    = ORB-SLAM3 stereo-only left-camera VO pose
T_Wv_I    = T_Wv_C * T_C_I
T_Wm_I_gt = T_Wm_M * T_M_I
```

The aggregate `mav0/imu0/data.csv` and `mav0/mocap0/data.csv` timestamps have been verified row-by-row for the selected/candidate sequences. The Debias_IMU supervision frame is mocap/world-frame IMU/body GT, not camera, marker, or first-frame-normalized poses.

Current cautions:

- Do not assume different datasets, sessions, or LIASuw sequences share the same camera, IMU/body, or mocap marker frame.
- VO supervision must use `--vo-supervision-world-frame gt`; raw VO-world supervision is invalid unless gravity is also transformed into that world.
- Local pseudo-groundtruth quality filtering is still not implemented.

## Phase 3: Neural ODE IMU Bias Dynamics Training

Goal: train bias-dynamics models for `b_g` and `b_a` using privileged trajectory supervision.

Status: implemented at research-prototype level. The main LIASuw runner is `plato/scripts/train_bias_dynamics/run_liasuw_debias_ad_pipeline.py`. It reuses external Debias_IMU training/evaluation code while keeping PLATO-specific loaders, model variants, and orchestration in `plato/`.

Current result summary:

- Original derivative-input LIASuw baseline is reproducible but has poor held-out velocity/position.
- `phys_delta` is the strongest current PLATO-owned model family.
- Corrected source-timing VO supervision with `--vo-supervision-world-frame gt` is currently the best LIASuw held-out velocity/position route: `results/debias_imu/LIASuw_phys_delta_vo_pseudogt_framefixed_e1800_w16pose_b64/`.
- The older raw-VO-world run `results/debias_imu/LIASuw_phys_delta_vo_pseudogt_e1800_w16pose_b64/` is an invalid diagnostic and should not be used for final conclusions.
- Route-A gravity alignment is a useful diagnostic but is not the current best learned model route.

Current cautions:

- The formal LIASuw train/validation/test split is not finalized. In particular, decide whether `0203s18_80`, `0203s19_80`, `0203s22_80`, and `0205s14_80` are promoted from usable-quality candidates.
- Training still needs local window sampling and RMSE-based pseudo-groundtruth filtering. Candidate window lengths include 32 and 64; maximum window length should not exceed 256.
- Initialization assumptions and pseudo-bias definitions must be explicit, especially before making deployment claims.

## Phase 4: Bias Export Interface

Goal: export learned bias trajectories or callable dynamics in a format usable by downstream evaluation scripts and OpenVINS.

Status: implemented for the current offline clean-IMU route and for an optional OpenVINS learned-dynamics hook.

Current outputs and scripts:

- Offline raw/constant/network IMU exports: `plato/scripts/openvins_eval/export_liasuw_openvins_inputs.py`.
- OpenVINS-readable bags: `plato/scripts/openvins_eval/make_liasuw_openvins_bags.py`.
- OpenVINS `phys_delta` YAML exporter: `plato/scripts/openvins_eval/export_phys_delta_to_openvins_yaml.py`.
- Current clean-IMU OpenVINS experiment root: `results/openvins_eval/LIASuw/phys_delta_vo_framefixed_clean_imu/0205s14_80/`.

Current cautions:

- The offline corrected-IMU route is the cleaner current evaluation path.
- The direct learned-bias OpenVINS hook runs, but no-GT/self-init tests show a deployment-distribution mismatch: OpenVINS-initialized bias is not the same quantity as the Debias_IMU pseudo-bias. Add a state/offset adaptation before treating that hook as a final deployment method.

## Phase 5: OpenVINS Integration Or Evaluation

Goal: evaluate learned or corrected bias information in OpenVINS.

Status: implemented for GT-initialized clean-IMU comparisons, artificial visual-dropout comparisons, and frozen-bias diagnostics. OpenVINS source also has opt-in learned-bias and frozen-bias modes after explicit user requests; defaults preserve original behavior.

Current result summary:

- Full GT-init clean-IMU comparison shows network-corrected IMU slightly improves velocity but is worse than constant bias on position and orientation.
- Artificial visual-dropout runs show the learned/network-corrected IMU is more useful when visual updates are removed.
- Frozen-bias dropout runs make the externally corrected IMU effect clearer and show strong network-vs-constant gains inside dropout intervals.

Current cautions:

- Self-initialized LIASuw OpenVINS VIO metrics are not yet fair or complete; static/dynamic initialization still needs LIASuw-specific tuning.
- GT-init and frozen-bias results are controlled diagnostics, not by themselves final deployment evidence.
- The known OpenVINS `class_loader::LibraryUnloadException` appears at shutdown after outputs are written and should be separated from data/config validity.

## Phase 6: Experiments On Visually Degraded Datasets

Goal: test whether learned bias dynamics improve VIO accuracy when visual data are poor.

Status: partially implemented through artificial visual dropout on `0205s14_80`. Natural visually degraded sequence evaluation is not yet formalized.

Current cautions:

- Define train/validation/test splits before headline claims to avoid leakage from visually favorable training windows into evaluation windows.
- Report dropout-window metrics separately from post-dropout recovery metrics, because long pure-IMU drift can dominate final errors.

## Near-Term Priorities

1. Finalize the LIASuw split and explicitly promote or exclude each usable-quality candidate.
2. Implement local pseudo-groundtruth quality filtering and run 32/64-window training comparisons, including the baseline and PLATO `phys_delta` model.
3. Treat corrected VO-supervised `phys_delta` plus offline clean-IMU export as the main current route for OpenVINS evidence.
4. Add state/offset adaptation before using the direct OpenVINS learned-bias hook for self-initialized deployment tests.
5. Keep `README.md`, `AGENTS.md`, `plato/docs/project_memory.md`, and this plan synchronized after meaningful experiment or pipeline changes.
