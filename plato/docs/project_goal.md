# PLATO Project Goal

PLATO studies privileged learning from accurate trajectory observations for IMU bias dynamics. The central idea is to obtain accurate pose trajectories from visually favorable data, then use those trajectories as supervisory measurements for learning continuous-time IMU bias dynamics.

## Technical Objective

The target biases are:

- gyroscope bias `b_g`
- accelerometer bias `b_a`

The intended model class is a Neural ODE for continuous bias dynamics. Rather than assuming that dense ground-truth bias labels are always available, PLATO will use high-quality pose, velocity, or trajectory-derived observations as supervision. ORB-SLAM3 stereo VO is the initial source of these privileged trajectory observations.

## Supervision Source

ORB-SLAM3 stereo visual odometry should be run on visually favorable datasets or favorable segments. Its trajectory estimates will be aligned and converted into a supervision format suitable for IMU bias-dynamics training. These trajectories act as privileged observations during training, not necessarily as inputs available at deployment time.

## Learning Target

The Debias_IMU / `rss_debiasing_restart` codebase already contains Neural ODE style bias dynamics for gyro and accelerometer signals. PLATO will adapt that direction so the learned dynamics can be trained from ORB-SLAM3 trajectory supervision and later exported in a format usable for OpenVINS experiments.

## Evaluation Target

OpenVINS will be used as the downstream VIO system. The planned evaluation is to compare standard OpenVINS behavior against variants that use learned or corrected bias information, especially on visually degraded datasets or datasets with visually poor segments. The key question is whether learned IMU bias dynamics allow more accurate and longer IMU preintegration when visual updates are weak.
