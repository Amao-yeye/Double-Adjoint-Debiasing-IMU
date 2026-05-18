# PLATO

**PLATO: Privileged Learning from Accurate Trajectory Observations for IMU Bias Dynamics**

PLATO is a research workspace for studying whether accurate visual trajectory estimates can supervise continuous IMU bias dynamics. The long-term goal is to learn or estimate gyroscope bias `b_g` and accelerometer bias `b_a` dynamics, then evaluate whether those learned dynamics improve IMU propagation and VIO performance when visual conditions degrade.

The workspace has moved beyond initial organization. Current PLATO-owned scripts can generate LIASuw ORB-SLAM3 pseudo-groundtruth candidates, export Debias_IMU-ready IMU/body supervision, train/evaluate bias-dynamics variants, and build OpenVINS clean-IMU and visual-dropout evaluations. The current results are still research-stage: corrected VO-supervised `phys_delta` is the strongest LIASuw training route so far, while self-initialized OpenVINS evaluation and local pseudo-groundtruth quality filtering remain open work.

## Motivation

Visual-inertial odometry can fail or drift when images are blurred, texture-poor, dark, or otherwise visually degraded. In visually favorable segments or datasets, stereo visual odometry can provide high-quality pose trajectories. PLATO treats those accurate trajectories as privileged supervisory observations for training IMU bias dynamics, with the expectation that better bias dynamics can support longer and more accurate IMU preintegration when visual updates become unreliable.

## External Projects

- `external/ORB_SLAM3`: generates high-quality stereo visual odometry trajectories on visually favorable datasets.
- `external/Debias_IMU/rss_debiasing_restart`: provides Neural ODE based IMU bias dynamics learning for `b_g` and `b_a`.
- `external/OpenVINS`: provides downstream VIO evaluation where learned or corrected IMU bias dynamics can be injected or compared.

## Intended Pipeline

1. Run ORB-SLAM3 stereo VO to obtain accurate trajectories.
2. Convert accepted VO windows into IMU/body supervision with explicit time, frame, and gravity conventions.
3. Train Neural ODE or related bias-dynamics models.
4. Export learned bias estimates or bias dynamics.
5. Evaluate with OpenVINS using offline corrected IMU streams and optional learned-bias hooks, especially under visual dropout or degraded visual updates.

PLATO-specific glue code, configs, and documentation should live under `plato/`. External repositories should remain as close to their original state as possible.

For the current state of experiments and accepted pseudo-groundtruth decisions, start with `AGENTS.md`, `plato/docs/project_memory.md`, and `plato/docs/pseudo_groundtruth_registry.md`.
