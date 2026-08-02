# PLATO Deletion And Keep Policy

Last updated: 2026-05-20

This note is the first stop before removing generated files from the PLATO
workspace. It separates reproducible build products, scratch experiments,
current results, and paper-critical artifacts.

## Rule Of Thumb

Do not delete a path only because it is large. Delete only when at least one of
these is true:

- the path is a build/cache product that can be regenerated from source;
- the path is explicitly recorded as superseded, negative, or partial;
- the path is listed as disposable in `plato/experiments/registry.yaml`;
- the important conclusion has already been copied into `plato/docs/`.

When in doubt, move a candidate into an archive location first or add it to the
registry with `keep: review`.

## Always Keep

These are source or high-value state:

- `README.md`
- `plato/`
- `tools/`
- `paper/`
- `.git/` and nested external `.git/` directories
- `external/ORB_SLAM3/`, except reproducible build products
- `external/Debias_IMU/rss_debiasing_restart/`, except caches and generated
  diagnostics explicitly marked disposable
- `external/OpenVINS/src/open_vins/`
- `baseline_code_original/`
- `plato/patches/external/`

## Keep Current Results

The following result classes are protected unless a newer registry entry
supersedes them:

- current corldata interval-NLL model:
  `results/debias_imu/corldata_sparse_endpoint_no_udot_w256_sg101_initmedian5_interval_nll_joint_lr1e5_from_covonly_continue/`
- previous corldata SE23/e301 baseline:
  `results/debias_imu/corldata_sparse_endpoint_no_udot_w128_sg101_initmedian5_e301_s13s15/`
- S20-S23 corldata only-test comparisons:
  `results/debias_imu/corldata_interval_nll_joint_best_onlytest_s20s21s22s23/`
  and `results/debias_imu/corldata_se23_e301_best_onlytest_s20s21s22s23/`
- final corldata OpenVINS full-trajectory table:
  `results/openvins_eval/corldata/natural/final_s13s23_raw_debiasing_ours_full_trajectory.md`
  plus the matching `.csv` and `.json`
- accepted LIASuw pseudo-GT windows under `results/pseudo_gt/LIASuw/`
- corldata testing pseudo-GT/evaluation artifacts for `S13`, `S15`, `S20`,
  `S21`, `S22`, and `S23`
- reusable exported datasets under `results/debias_imu/`, especially
  `LIASuw_training_candidates_200hz/`,
  `corldata_IMU_aligned_gt_vo_ref_1000hz/`,
  `corldata_IMU_aligned_gt_vo_ref_sparse_gt/`, and
  `Euroc_GT_at_VO_timestamps_dedup/`

Use `plato/experiments/registry.yaml` as the canonical short list for current
keep decisions.

## Safe To Delete First

These have the lowest risk:

- `__pycache__/`
- `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`
- LaTeX build artifacts such as `*.aux`, `*.bbl`, `*.blg`, `*.fdb_latexmk`,
  `*.fls`, `*.log`, `*.out`, `*.synctex.gz`
- build directories that are not the active build tree

Already deleted on 2026-05-20:

- `external/OpenVINS/build_old_moved_path/`
- `external/OpenVINS/devel_old_moved_path/`
- `external/OpenVINS/logs_old_moved_path/`

Reason: no repository references, ignored by git, stale paths pointed to
`/home/amao/workspace/openvins`, and contents were catkin/CMake products.

## Usually Regenerable But Check First

These save substantial space but may cost time to rebuild:

- `external/OpenVINS/build/`
- `external/OpenVINS/devel/`
- `external/OpenVINS/logs/`
- `external/ORB_SLAM3/build/`
- `external/ORB_SLAM3/build_plato/`
- generated ROS bags under `results/openvins_eval/`
- Debias_IMU caches under `results/debias_imu/cache/`

Before deleting active build products, confirm no running process depends on
them and that the rebuild command/environment is still known.

## High Risk

Do not delete these without a fresh registry update:

- `results/debias_imu/*best*`
- `results/debias_imu/checkpoints/`
- `results/openvins_eval/corldata/natural/final_*`
- `results/openvins_eval/corldata/natural/*summary*`
- `results/pseudo_gt/LIASuw/0912s06_80/`
- `results/pseudo_gt/LIASuw/0912s07_30/`
- `results/pseudo_gt/LIASuw/0912s10_40/`
- `results/pseudo_gt/corldata/light_segments/`
- paper figures under `pictures_in_paper/`

## Known Disposable Classes

These are candidates for deletion after checking the registry and current docs:

- smoke runs;
- probe/profile directories;
- interrupted partial tests;
- result directories documented as `negative`, `superseded`, or `oracle`;
- old summaries whose conclusions are already preserved in `plato/docs/`;
- OpenVINS sweeps that are not part of final, fair, or diagnostic tables.

Important negative-result records:

- `plato/docs/debias_imu_results/negative_attempts_2026_05_19.md`
- `plato/docs/debias_imu_results/cleanup_2026_05_19.md`
- `plato/docs/current_experiment_state.md`
- `plato/docs/corldata_openvins_learned_imu_lessons.md`

## Minimal Deletion Checklist

1. Search for references:

   ```bash
   rg -n "path_or_directory_name" .
   ```

2. Check whether git tracks it:

   ```bash
   git ls-files path_or_directory_name
   git check-ignore -v path_or_directory_name
   ```

3. Check size and file types:

   ```bash
   du -sh path_or_directory_name
   find path_or_directory_name -maxdepth 3 -type f | sed -n '1,80p'
   ```

4. Check for running processes if it is a build/runtime directory:

   ```bash
   ps -ef | rg "OpenVINS|ORB_SLAM3|Debias|roslaunch|rosbag|catkin"
   ```

5. Update `plato/experiments/registry.yaml` if the deletion changes what is
   considered current, protected, or disposable.
