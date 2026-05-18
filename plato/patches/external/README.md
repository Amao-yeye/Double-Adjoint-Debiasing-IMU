# External Repository Patch Snapshots

These patch files preserve tracked local source changes in nested external repositories so the main PLATO repository can be pushed without vendoring large upstream worktrees.

Created on 2026-05-18:

- `openvins_local_changes_2026-05-18.patch`: tracked diff from `external/OpenVINS/src/open_vins`.
- `orbslam3_local_changes_2026-05-18.patch`: tracked diff from `external/ORB_SLAM3`.
- `debias_imu_local_changes_2026-05-18.patch`: tracked diff from `external/Debias_IMU/rss_debiasing_restart`.

These snapshots do not include untracked generated outputs or nested build artifacts. For exact external repository restoration, also keep the external worktrees or push their changes to appropriate forks.
