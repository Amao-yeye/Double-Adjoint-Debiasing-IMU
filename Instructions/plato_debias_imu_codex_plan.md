# PLATO / Debias_IMU: Codex Implementation Plan for No-\(\dot u\) Neural ODE + Trajectory-State NLL

## 0. Purpose of This Document

This document defines the implementation and experiment plan for the next PLATO / Debias_IMU training stage.

The main goal is to train an IMU bias dynamics Neural ODE using pose-level trajectory supervision, while keeping the implementation clean, reusable, and compatible with both **EuRoC** and **corldata**.

The expected final model should:

1. use no \(\dot u\) input by default;
2. use the derived trajectory-state NLL loss;
3. support sparse but effective pose supervision;
4. minimize testing-set position and rotation error in long-horizon pure IMU rollout;
5. use one unified architecture for both EuRoC and corldata;
6. use the same two-stage training strategy on both EuRoC and corldata: MSE pretraining followed by NLL fine-tuning;
7. reuse existing code whenever possible instead of duplicating functionality.

Current priority update:

- The primary model-selection criterion is testing-set position error and rotation error. Velocity is useful to report, but secondary.
- Bias-trend similarity to a GT-differenced or pseudo-reference bias is a diagnostic only. Do not optimize for it directly, and do not reject a model only because the predicted bias trend has low correlation if testing position/rotation improve.
- EuRoC does not need to beat every internal ablation; it must beat the original RSS/Debias_IMU baseline under the same testing protocol.
- Formal cross-dataset claims must use the same core network structure and the same MSE-to-NLL training pipeline on EuRoC and corldata. Dataset-specific differences are allowed only for sampling rates, physical window duration, calibration, frame alignment, supervision source, train/test split, and noise initialization.

Current EuRoC status as of 2026-05-18:

- A unified no-`u_dot` AD run using `bw_func_net_no_udot` / `ba_func_net_no_udot`, GT poses sampled at VO timestamps, `loss_window=32` source poses, fixed zero-output gyro, and Acc-only SE23 MSE pretraining achieved mean held-out EuRoC `2.0154 deg / 13.9936 vel / 624.5582 m`.
- Its AD interval-NLL fine-tune, using `interval_nll_mode=nll_joint` and `interval_covariance_gradient_mode=ad`, achieved `2.0154 deg / 14.1680 vel / 630.8582 m`.
- Both stages beat the RSS/original references (`Euroc_baseline_w64` `3.4944 / 16.2950 / 678.0098`, `Euroc_SE23_w64` `3.4944 / 16.0696 / 672.2834`), but NLL slightly regresses from MSE. Treat the NLL result as the formal MSE-to-NLL EuRoC checkpoint for the current RSS-beating target, while recording that MSE is the stronger mean-trajectory checkpoint.

---

## 1. Mandatory First Step: Inspect Existing Code and Reuse What Already Exists

Before implementing new modules, first inspect the current repository and identify existing components that can be reused.

Do **not** blindly create new implementations for logic that already exists.

### 1.1 Required Repository Audit

Please first check whether the repository already contains reusable implementations for:

- IMU preintegration;
- SE(3), SE\(_2(3)\), SO(3), or Lie group utilities;
- pose residual computation;
- covariance propagation;
- Cholesky / matrix solve utilities;
- existing MSE loss;
- existing NLL or Gaussian likelihood loss;
- existing bias model or Neural ODE model;
- dataset loaders for EuRoC and corldata;
- timestamp synchronization;
- coordinate-frame alignment;
- normalization utilities;
- logging and visualization scripts;
- evaluation metrics such as ATE, AOE, endpoint error, and trajectory drift.

Use existing modules whenever they are correct and compatible with the new objective.

### 1.2 Suggested Code Search Keywords

Search the repository using keywords such as:

```text
preintegrat
covariance
cholesky
logdet
nll
negative_log
gaussian
SE3
SE23
SO3
lie
expmap
logmap
pose_error
residual
bias
gyro_bias
acc_bias
NeuralODE
ode
EuRoC
corldata
ATE
AOE
trajectory
alignment
```

### 1.3 Required Reuse Map

Before making major code changes, create a short internal reuse map.

The reuse map should answer:

| Component | Existing file/module | Reuse directly? | Need modification? | Notes |
|---|---|---:|---:|---|
| IMU preintegration | TBD | Yes/No | Yes/No | TBD |
| Lie group ops | TBD | Yes/No | Yes/No | TBD |
| Pose residual | TBD | Yes/No | Yes/No | TBD |
| Covariance propagation | TBD | Yes/No | Yes/No | TBD |
| MSE loss | TBD | Yes/No | Yes/No | TBD |
| NLL loss | TBD | Yes/No | Yes/No | TBD |
| Dataset loader | TBD | Yes/No | Yes/No | TBD |
| Evaluation | TBD | Yes/No | Yes/No | TBD |
| Visualization | TBD | Yes/No | Yes/No | TBD |

If an existing component is only partially compatible, extend it with minimal changes instead of replacing it.

### 1.4 Implementation Principle

Prefer this order:

1. reuse existing implementation directly;
2. wrap existing implementation with a new interface;
3. minimally extend existing implementation;
4. create a new module only when no reliable reusable implementation exists.

Avoid large invasive rewrites.

Keep all new behavior behind explicit config flags.

---

## 2. Overall Modeling Goal

The target bias dynamics model is:

\[
\dot b_t = f_\theta(b_t, u_t),
\]

where:

- \(b_t\) is the IMU bias state;
- \(u_t\) is the IMU measurement input;
- \(\theta\) denotes the Neural ODE parameters.

The model should improve long-horizon pure IMU preintegration and eventually be usable inside a filter such as OpenVINS as a better bias prediction model.

The main supervision signal is a high-quality pose trajectory, which may come from:

- ground truth;
- motion capture;
- high-quality visual odometry;
- another reliable pose source.

---

## 3. Core Requirement 1: Network Input Must Not Use \(\dot u\)

The default model must not use \(\dot u\).

Use:

\[
\dot b_t = f_\theta(b_t, u_t).
\]

Do **not** use:

\[
\dot b_t = f_\theta(b_t, u_t, \dot u_t).
\]

### 3.1 Reason

\(\dot u\) requires numerical differentiation or spline fitting. This can introduce:

- additional noise;
- extra hyperparameters;
- spline-related implementation complexity;
- dataset-specific tuning;
- weaker generalization.

Therefore, the no-\(\dot u\) model should be the default and primary experimental setting.

### 3.2 Config Requirement

Add or confirm an explicit config flag:

```yaml
model:
  use_udot: false
```

The default should be:

```yaml
use_udot: false
```

A future ablation may support \(\dot u\), but this stage should focus on the clean no-\(\dot u\) setting.

---

## 4. Core Requirement 2: Use the Derived Trajectory-State NLL Loss

The training objective should use the derived trajectory-state negative log-likelihood loss.

Do not only use independent per-state NLL.

The supervised states are temporally correlated because they are generated by IMU preintegration. Therefore, the covariance over supervised timestamps is generally not independent.

However, under invariant preintegration and log-linearized error dynamics, the precision structure can often be exploited efficiently.

---

## 5. Trajectory-State Residual Definition

For a training window with \(M\) supervised states, define the stacked residual:

\[
r^s =
\begin{bmatrix}
r^s_1 \\
r^s_2 \\
\cdots \\
r^s_M
\end{bmatrix}.
\]

Each residual should be computed on the Lie group.

A typical pose residual is:

\[
r^s_i =
\log\left(
Y^s_i
\left(\bar X^s_i\right)^{-1}
\right),
\]

where:

- \(Y^s_i\) is the supervised pose at the \(i\)-th supervised timestamp;
- \(\bar X^s_i\) is the predicted preintegrated pose;
- \(M\) is the number of supervised timestamps in the current window.

The exact left-invariant or right-invariant convention must be consistent with the existing preintegration and covariance propagation code.

If the repository already has a pose residual function, reuse it after verifying its convention.

---

## 6. Trajectory-State NLL Loss

The loss should follow:

\[
\mathcal L_{\mathrm{NLL}}
=
\left(r^s\right)^\top
\left(\Sigma^s\right)^{-1}
r^s
+
\log\left|\Sigma^s\right|.
\]

Here:

- \(r^s\) is the stacked trajectory residual;
- \(\Sigma^s\) is the covariance of the supervised trajectory states;
- \((\Sigma^s)^{-1}\) is the precision matrix.

\(\Sigma^s\) should come from the propagated covariance of the invariant preintegration system projected to the supervised timestamps.

Do not replace \(\Sigma^s\) with arbitrary independent per-state covariance unless the experiment is explicitly marked as a baseline.

---

## 7. Numerical Implementation Requirements for NLL

Avoid explicitly computing:

```python
torch.inverse(Sigma)
```

unless the supervised dimension is very small and this path is only used for debugging.

Preferred approaches:

1. Cholesky factorization;
2. triangular solve;
3. sparse precision formulation;
4. block-tridiagonal precision structure;
5. stable log-determinant from Cholesky diagonal;
6. adding small jitter only when necessary.

Recommended safe pattern:

```python
# Pseudocode only
L = cholesky(Sigma + jitter * I)
alpha = cholesky_solve(r, L)
quad = r.T @ alpha
logdet = 2.0 * sum(log(diag(L)))
loss = quad + logdet
```

If an existing Cholesky, matrix-solve, or Gaussian NLL utility already exists, prefer reusing it.

---

## 8. Core Requirement 3: Sparse Supervision Is Allowed, But Must Be Tuned Carefully

The supervision frequency can be lower than the IMU frequency.

It is not necessary to supervise every IMU timestamp.

Possible supervision strategies:

- endpoint-only supervision;
- sparse intermediate supervision;
- supervision every \(K\) IMU steps;
- fixed number of supervised states per window;
- adaptive supervision based on trajectory duration.

Important hyperparameters:

- supervision stride;
- window length;
- number of supervised states per window;
- covariance scaling;
- process noise initialization;
- learning rate;
- batch size;
- gradient clipping threshold;
- MSE pretraining epochs;
- NLL fine-tuning epochs;
- robust loss threshold;
- outlier rejection threshold.

The goal is not to maximize supervision frequency.

The goal is to find a stable, efficient, and interpretable sparse trajectory supervision setting.

---

## 9. Recommended Training Strategy

Use the same two-stage strategy on both EuRoC and corldata.

The formal training pipeline is:

1. SE23 MSE pretraining for the mean bias-dynamics model.
2. NLL fine-tuning from the MSE checkpoint.

Do not report a cross-dataset final model that uses MSE-only on one dataset and MSE-to-NLL on the other. MSE-only checkpoints may still be reported as ablations or initialization checkpoints.

### 9.1 Stage 1: SE23 MSE Pretraining

First train the Neural ODE mean model using the existing SE23 MSE-style loss.

Reuse the existing MSE loss if available.

This stage should produce:

- stable mean trajectory prediction;
- non-divergent long-horizon preintegration;
- reasonable initial bias dynamics.

### 9.2 Stage 2: Trajectory-State NLL Fine-Tuning

After MSE pretraining, switch to the trajectory-state NLL loss.

Start conservatively:

- lower learning rate than MSE pretraining;
- gradient clipping enabled;
- stable covariance initialization;
- sparse supervision;
- optional mean-model freezing for debugging.

If NLL training is unstable, run the following debugging sequence:

1. endpoint-only NLL;
2. sparse trajectory NLL with fixed covariance;
3. q-only NLL with frozen mean model;
4. trajectory-state NLL with frozen mean model;
5. full joint fine-tuning of mean model and covariance parameters.

---

## 10. Core Requirement 4: Bias Trend Is Diagnostic, Not the Optimization Target

The model should be selected by testing-set position and rotation accuracy.

Bias trajectories should still be visualized because they are useful diagnostics, but bias-trend correlation is not the primary objective and is not a required success criterion.

After training, visualize and compare:

1. predicted gyroscope bias trajectory;
2. predicted accelerometer bias trajectory;
3. reference bias trajectory, if available;
4. pseudo-reference bias trend inferred from high-quality pose trajectory, if true bias is unavailable;
5. before-training and after-training bias trajectories;
6. bias difference trend;
7. long-horizon pure IMU preintegration trajectory.

Expected behavior:

- testing-set position error should decrease relative to the RSS/original Debias_IMU baseline;
- testing-set rotation error should decrease relative to the RSS/original Debias_IMU baseline;
- long-horizon pure IMU drift should decrease;
- NLL should not be judged by scalar likelihood alone; if NLL decreases but testing position/rotation worsens, record it as a negative or calibration-only result;
- bias plots should be used to diagnose failure modes, not to select the final model.

If no ground-truth bias is available, clearly label any inferred bias as pseudo-reference rather than true bias.

---

## 11. Core Requirement 5: Unified Architecture for EuRoC and corldata

Use the same core architecture for both EuRoC and corldata.

Do not design separate network structures for different datasets.

For the formal result, do not use one network family for EuRoC and a different network family for corldata. For example, a EuRoC-only tiny slow-residual model and a corldata-only full no-\(\dot u\) MLP can be kept as ablations, but they do not satisfy the unified-architecture requirement.

Dataset-specific logic is allowed only in:

- data loading;
- sensor calibration;
- time synchronization;
- coordinate-frame alignment;
- normalization;
- train-test split;
- noise initialization;
- supervision source construction.

The following should stay unified:

- Neural ODE architecture;
- bias dynamics formulation;
- loss definition;
- training pipeline, specifically MSE pretraining followed by NLL fine-tuning;
- evaluation metrics;
- visualization scripts.

Final evaluation should report results on both datasets using the same architecture.

---

## 12. Suggested Experiment Order

### Step 1: Repository Audit

Inspect existing modules and complete the reuse map.

Do this before adding new code.

### Step 2: Select One Shared No-\(\dot u\) Architecture

Choose one core Neural ODE architecture that will be used on both EuRoC and corldata.

Record any dataset-specific parameters separately from the model structure.

### Step 3: MSE Pretraining on Both Datasets

Run the selected no-\(\dot u\) model on both datasets using the existing SE23 MSE loss.

Confirm stable training and testing-set position/rotation metrics on each dataset.

### Step 4: NLL Fine-Tuning on Both Datasets

Starting from the MSE checkpoints, run NLL fine-tuning on both datasets with the same NLL family and the same staged strategy.

Start with sparse supervision and conservative covariance settings.

Compare NLL checkpoints against the MSE checkpoints and the RSS/original Debias_IMU baseline using testing-set position and rotation.

### Step 5: Supervision Frequency and Physical Window Sweep

Sweep, using physical time rather than raw sample count:

- supervision stride;
- window length;
- number of supervised states;
- covariance scaling.

Find a setting that is stable and improves long-horizon preintegration.

### Step 6: Bias Trend Diagnostics

Plot learned bias trajectories.

Use bias trend only to diagnose why a model succeeds or fails. Do not use bias-trend correlation as the primary selection metric.

### Step 7: Unified Architecture Evaluation

Report final results on both EuRoC and corldata using the same model structure and MSE-to-NLL training pipeline.

EuRoC is considered successful when it beats the RSS/original Debias_IMU baseline on testing position and rotation; it does not need to beat every internal PLATO ablation.

---

## 13. Required Baselines

Keep the following baselines runnable:

1. original Debias_IMU baseline;
2. SE23 MSE training;
3. endpoint-only NLL;
4. independent per-state NLL;
5. q-only NLL with frozen mean model;
6. trajectory-state NLL;
7. trajectory-state NLL with no-\(\dot u\) Neural ODE.

The final comparison should explain whether the gain comes from:

- better bias dynamics learning;
- better covariance calibration;
- better temporal correlation modeling;
- better long-horizon preintegration;
- better use of pose-level supervision.

---

## 14. Recommended Code Organization

Prefer modular additions.

Suggested structure:

```text
losses/
  se23_mse_loss.py
  endpoint_nll_loss.py
  trajectory_state_nll_loss.py
  covariance_regularization.py

models/
  bias_dynamics_node.py

utils/
  lie_group.py
  covariance.py
  trajectory_residual.py
  visualization.py
```

However, do not create these files blindly.

If the repository already has equivalent modules, reuse or extend them.

For example:

- if `lie_group.py` already exists, do not create another Lie group implementation;
- if a preintegration covariance function already exists, reuse it;
- if an MSE loss already exists, call it instead of rewriting it;
- if plotting utilities already exist, add bias plots there instead of creating duplicate scripts.

---

## 15. Suggested Config Interface

Add or verify explicit config fields.

Example:

```yaml
model:
  type: bias_dynamics_node
  use_udot: false

loss:
  type: trajectory_state_nll
  use_sparse_precision: true
  use_logdet: true
  covariance_scaling: 1.0
  jitter: 1.0e-6
  robust_threshold: null

supervision:
  source: gt_or_vo
  stride: 20
  num_supervised_states: null
  endpoint_only: false
  align_to_imu_frame: true

training:
  mse_pretrain_epochs: 50
  nll_finetune_epochs: 50
  learning_rate_mse: 1.0e-3
  learning_rate_nll: 1.0e-4
  gradient_clip_norm: 1.0
  freeze_mean_for_q_only_debug: false

dataset:
  name: corldata_or_euroc
  use_dataset_specific_calibration: true
  normalize_imu: true
```

The default config should correspond to:

```yaml
model:
  use_udot: false

loss:
  type: trajectory_state_nll
```

---

## 16. Logging and Debugging Requirements

Log the following for every experiment:

- total loss;
- MSE loss if hybrid training is used;
- NLL quadratic term;
- NLL log-determinant term;
- covariance scaling;
- learned \(\sigma_a\);
- learned \(\sigma_g\);
- gradient norm;
- Cholesky success or failure;
- minimum covariance eigenvalue if available;
- failed window index;
- supervision timestamps;
- training and validation metrics.

If NLL is unstable, check:

### 16.1 Residual and Frame Consistency

- Is the residual left-invariant or right-invariant?
- Is this convention consistent with covariance propagation?
- Are ground-truth or VO poses aligned to the IMU body frame?
- Are timestamps synchronized?
- Are gravity direction and world frame conventions consistent?

### 16.2 Covariance Stability

- Is the covariance positive definite?
- Is Cholesky factorization stable?
- Is the log-determinant finite?
- Is covariance scaling too small?
- Is process noise initialized reasonably?
- Is jitter too large or too small?

### 16.3 Loss Balance

- Does the quadratic term dominate?
- Does the log-determinant term dominate?
- Does covariance collapse?
- Do gradients explode?
- Does trajectory improve only at supervised timestamps but degrade between them?

---

## 17. Required Outputs

For each experiment, save:

1. full config file;
2. random seed;
3. training loss curves;
4. validation loss curves;
5. ATE and AOE metrics;
6. endpoint error;
7. long-horizon pure IMU preintegration trajectory;
8. predicted gyroscope bias trajectory;
9. predicted accelerometer bias trajectory;
10. reference or pseudo-reference bias trend;
11. before-training and after-training bias comparison;
12. covariance parameters;
13. NLL quadratic term and log-det term;
14. model checkpoint;
15. evaluation summary table.

---

## 18. Success Criteria

This stage is successful if:

1. the no-\(\dot u\) Neural ODE trains stably;
2. the derived trajectory-state NLL loss runs without numerical failure;
3. sparse supervision achieves stable and efficient training;
4. the formal pipeline uses MSE pretraining followed by NLL fine-tuning on both EuRoC and corldata;
5. testing-set position and rotation error improve over the RSS/original Debias_IMU baseline;
6. long-horizon pure IMU preintegration drift is reduced;
7. the same core architecture works on both EuRoC and corldata;
8. the implementation reuses existing code where appropriate;
9. all baselines remain runnable;
10. the results support the claim that pose-level supervisory measurements can effectively guide IMU bias dynamics learning.

---

## 19. Important Principle

The purpose of the NLL loss is not only to reduce a scalar loss value.

The real goal is to improve testing-set position and rotation in long-horizon pure IMU rollout while keeping the likelihood formulation statistically meaningful.

Therefore, always evaluate:

1. trajectory accuracy;
2. long-horizon pure IMU drift;
3. NLL quadratic/log-det behavior;
4. bias evolution behavior as a diagnostic.

A model should not be selected because its NLL scalar improves or its bias trend looks better. Select by testing-set position and rotation first; use the other diagnostics to explain the result and catch obvious implementation failures.

---

## 20. Expected Codex Behavior

When implementing this plan, Codex should:

1. inspect the existing codebase first;
2. identify reusable components;
3. avoid duplicate implementations;
4. keep changes modular and config-controlled;
5. preserve existing baselines;
6. add trajectory-state NLL incrementally;
7. add sufficient logging and visualization;
8. validate testing-set position/rotation before drawing conclusions;
9. keep bias-trend plots as diagnostics;
10. keep EuRoC and corldata under one unified architecture and one MSE-to-NLL training strategy.
