# PLATO Double-Adjoint Gradient Notes

Date: 2026-05-21

## Source

Local paper:

```text
papers/mypaper/PLATO.pdf
```

Relevant sections:

```text
3.4 Continuous Adjoint Method for NODEs
6 Gradient Estimation
6.1 Outer Adjoint for Bias Sensitivity
6.2 Inner Adjoint Method for NODE Training
```

## Main Idea

The gradient can be estimated without backpropagating through the full SE_2(3)
trajectory graph.

The paper separates the computation into two adjoint levels:

```text
1. Outer analytic adjoint:
   loss -> invariant SE_2(3) error -> bias sequence b_k

2. Inner NODE adjoint:
   bias sequence b_k -> neural bias dynamics parameters theta
```

This is the double-adjoint strategy.

## Outer Analytic Adjoint

Use the right-invariant SE_2(3) error:

```text
xi_k = Log(X_gt,k * X_pred,k^{-1})
```

Under first-order linearization, the invariant error dynamics have a log-linear
transition:

```text
xi_{k+1} = F_k xi_k + G_k n_k
```

For bias sensitivity, replace the noise perturbation by the bias perturbation:

```text
xi_{k+1} = F_k xi_k + M_k delta b_k
M_k = G_k
```

Here `G_k` means the signed matrix returned by the current implementation
`plato.covariance.se23_Gd`. That helper already includes the minus sign from
the IMU residual convention.

The backward adjoint recursion is:

```text
lambda_N = d ell_N / d xi_N
lambda_k = d ell_k / d xi_k + F_k^T lambda_{k+1}
```

Then the loss gradient with respect to each bias state is:

```text
dL / d(delta b_k) = M_k^T lambda_{k+1}
                  = G_k^T lambda_{k+1}
```

The useful property is that the transition `F_k` has an analytic form for the
right-invariant SE_2(3) error:

```text
F = [ I              0     0
      g^ dt          I     0
      0.5 g^ dt^2    dt I  I ]
```

This matches the implemented covariance transition:

```text
external/Debias_IMU/BiasDy/plato/covariance.py::se23_phi
```

The injection matrix is implemented as:

```text
external/Debias_IMU/BiasDy/plato/covariance.py::se23_Gd
```

## Inner NODE Adjoint

After the outer adjoint gives gradients with respect to the bias sequence:

```text
dL / db_k
```

the bias dynamics are treated as the NODE:

```text
dot b = f_theta(b, u)
```

For fixed-step Euler, a discrete inner adjoint can be written as:

```text
b_{k+1} = b_k + dt * f_theta(b_k, u_k)

mu_k = dL/db_k + (I + dt * df/db_k)^T mu_{k+1}
dL/dtheta += dt * (df/dtheta)^T mu_{k+1}
```

This avoids storing the full differentiable rollout graph. The local VJP
through `f_theta` can still use PyTorch autograd on each step, while the full
trajectory propagation uses analytic adjoint recursions.

## Current Project Status

The default training path still uses normal PyTorch backpropagation through
`odeint_SO3`, preserving existing behavior:

```text
external/Debias_IMU/BiasDy/plato/training.py
external/Debias_IMU/SO3diffeq
```

The baseline RSS paper explicitly states that its SO(3) NODE gradient uses
numerical backpropagation and that the adjoint method is future work.

The current PLATO paper addresses this limitation by using the double-adjoint
strategy above.

## FormalPLATO Implementation

Clean implementation paths in this repository:

```text
external/Debias_IMU/BiasDy/plato/analytic_gradient.py
external/Debias_IMU/BiasDy/plato/bias_adjoint.py
external/Debias_IMU/BiasDy/plato/so3_analytic_gradient.py
external/Debias_IMU/BiasDy/plato/gyro_adjoint.py
external/Debias_IMU/BiasDy/plato/training.py
external/Debias_IMU/BiasDy/mainEurocPLATO.py
```

Design choices:

```text
1. Do not import from the old rss_debiasing_restart workspace.
2. Keep autograd as the default gradient mode.
3. Expose double-adjoint as an explicit option:
   --gradient_mode double_adjoint
4. Keep the open-source-safe analytical path limited to derived/checked cases:
   gt_se23 + mse_residual_mode=right + gyro/acc MSE + sparse GT NLL + euler
5. Use current-project loss/block ordering [phi, rho_v, rho_p].
6. Default analytic residual Jacobian is strict BCH:
   H = -dexp^{-1}_{-r}, r = Log(X_gt X_pred^{-1}).
   In code this is `-SEn3rightJaco_inv(r)` for SE23 and
   `-SO3rightJacoInv(r)` for SO3. `identity/left/right` remain available
   only as explicit ablation options.
```

Implemented pieces:

```text
analytic_gradient.weighted_se23_mse_state_gradient
    H^T dL/dr for 0.5 * mean(||W r||^2), with strict BCH residual
    Jacobian by default.

analytic_gradient.invariant_transition_matrices
    analytic F and G from current plato.covariance.se23_phi / se23_Gd.

analytic_gradient.adjoint_bias_gradient
    outer backward recursion, producing dL/db_k.

bias_adjoint.accumulate_ba_euler_adjoint_gradients
    inner fixed-step Euler adjoint with local PyTorch VJP through f_ba.

so3_analytic_gradient.so3_gyro_bias_gradient
    SO(3) right-invariant outer adjoint for gyro/bg supervision, using
    strict BCH residual Jacobian by default.

gyro_adjoint.accumulate_bg_euler_adjoint_gradients
    inner fixed-step Euler adjoint with local PyTorch VJP through f_bg.
```

Ordering convention:

```text
ODE state order:        [xi, t, bg, v, p, ba]
SE23 residual order:    [phi, rho_v, rho_p]
se23_Gd columns:        [n_a, n_g]
adjoint bias gradient:  [ba, bg]
```

This means acc training must use `analytic.bias_gradient[..., 0:3]` for `ba`.
The `bg` slots are `analytic.bias_gradient[..., 3:6]`; the gyro-only SO3 path
uses its dedicated three-dimensional `so3_gyro_bias_gradient`.

Training entry:

```text
/home/amao/anaconda3/envs/debiasimu/bin/python BiasDy/mainEurocPLATO.py \
  --plato_stage gyro|acc \
  --plato_supervision gt_se23 \
  --mse_residual_mode right \
  --gradient_mode double_adjoint
```

Current support and limitations:

```text
double_adjoint is enabled for:
- GT gyro/bg MSE with strict SO3 right residual.
- GT acc/ba SE23-MSE with mse_residual_mode=right.
- Sparse GT NLL with nll_gt_residual_mode=right, using local perturbation AD
  for the sparse NLL residual/noise blocks and analytical adjoint for the
  trajectory/bias dynamics.

double_adjoint is intentionally not enabled for VO supervision, hybrid/global
residuals, joint training, or non-Euler integrators yet. Those need separate
derivations or gradient checks before being open-source safe.

For the current local networks, `f_bg` and `f_ba` do not use the `R0` argument.
If a future network uses per-step chart rotation in `R0`, the inner adjoint
should store or reconstruct the per-step chart before enabling double-adjoint
for that architecture.
```

Validation added:

```text
BiasDy/plato/loss_smoke_test.py
    - weighted SE23 MSE state gradient matches autograd on residual tensors
    - strict SE23 BCH Jacobian matches autograd for Log(Exp(r) Exp(-delta))
    - outer adjoint returns expected F/G/lambda/bias-gradient shapes
    - weighted SO3 MSE state gradient matches autograd on residual tensors
    - strict SO3 BCH Jacobian matches autograd for Log(Exp(r) Exp(-delta))
    - SO3 gyro outer adjoint returns expected lambda/bias-gradient shapes
    - SE23 bias-gradient order is verified as [ba, bg] against se23_Gd
```

Validation commands completed:

```text
/home/amao/anaconda3/envs/debiasimu/bin/python -m py_compile \
  external/Debias_IMU/BiasDy/plato/analytic_gradient.py \
  external/Debias_IMU/BiasDy/plato/bias_adjoint.py \
  external/Debias_IMU/BiasDy/plato/so3_analytic_gradient.py \
  external/Debias_IMU/BiasDy/plato/gyro_adjoint.py \
  external/Debias_IMU/BiasDy/plato/training.py \
  external/Debias_IMU/BiasDy/mainEurocPLATO.py \
  external/Debias_IMU/BiasDy/plato/loss_smoke_test.py

/home/amao/anaconda3/envs/debiasimu/bin/python BiasDy/plato/loss_smoke_test.py
/home/amao/anaconda3/envs/debiasimu/bin/python BiasDy/mainEurocPLATO.py --debug_smoke
```

Acc dry-run command completed:

```text
/home/amao/anaconda3/envs/debiasimu/bin/python BiasDy/mainEurocPLATO.py \
  --plato_supervision gt_se23 \
  --plato_stage acc \
  --outputdir results/Euroc_PLATO_double_adjoint_acc_smoke \
  --network_type_Gyro bw_func_net_damped_145e3 \
  --network_type_Acc ba_func_net_rate_tanh_1e2 \
  --resume_gyro results/Euroc_baseline_w64/Gyro_weights/model_epoch_1800.pt \
  --resume_acc results/Euroc_PLATO_acc_se23_hybrid_V1P10_w64_b1000/Acc_weights/best_val_model.pt \
  --plato_window_steps 16 \
  --loss_window 16 \
  --batch_size 8 \
  --epoch_mse 1 \
  --val_freq 1 \
  --save_freq 1 \
  --train_sequence_strategy cycle \
  --mse_residual_mode right \
  --mse_acc_rot_weight 0 \
  --mse_acc_vel_weight 1 \
  --mse_acc_pos_weight 10 \
  --gradient_mode double_adjoint \
  --analytic_grad_scale 1 \
  --analytic_grad_clip_norm 1 \
  --device cuda
```

Dry-run result:

```text
epoch: 0
train_loss: 1747.004150390625
val_loss: 11502.673950195312
```

The temporary smoke output directory was removed after validation to keep the
result tree clean.

Gyro dry-run command completed:

```text
/home/amao/anaconda3/envs/debiasimu/bin/python BiasDy/mainEurocPLATO.py \
  --plato_supervision gt_se23 \
  --plato_stage gyro \
  --outputdir results/Euroc_PLATO_double_adjoint_gyro_smoke \
  --network_type_Gyro bw_func_net_damped_145e3 \
  --plato_window_steps 16 \
  --loss_window 16 \
  --batch_size 8 \
  --epoch_mse 1 \
  --val_freq 1 \
  --save_freq 1 \
  --train_sequence_strategy cycle \
  --mse_residual_mode right \
  --gradient_mode double_adjoint \
  --analytic_grad_scale 1 \
  --analytic_grad_clip_norm 1 \
  --device cuda
```

Gyro dry-run result:

```text
epoch: 0
train_loss: 1.0235698223114014
val_loss: 22.789779663085938
```

The temporary gyro smoke output directory was removed after validation.

## AD Consistency Check

Small-window end-to-end gradient checks were run against normal PyTorch AD
through `odeint_SO3` with:

```text
gt_se23 supervision
w16
batch_size 4
mse_residual_mode right
analytic_residual_jacobian bch
analytic_grad_clip_norm 0
```

Results:

| Stage | Loss AD | Loss double-adjoint | Grad cosine | Relative grad error |
| --- | ---: | ---: | ---: | ---: |
| gyro/bg random init | 0.9565474391 | 0.9565474391 | 1.000000119 | 1.753476972e-4 |
| acc/ba random init | 1130.153442 | 1130.153442 | 0.9999997616 | 2.324083397e-7 |

The loaded BA checkpoint used in one check had zero BA parameter gradient for
both AD and double-adjoint on that batch, so the random-init BA case is the
meaningful sign/direction check.

## Existing Memory Benchmark

Existing benchmark record:

```text
Results/double_adjoint_memory_benchmark/summary.md
```

For `w64`, batch size `1000`:

| Stage | AD max MB | Adj max MB | Memory ratio |
| --- | ---: | ---: | ---: |
| gyro | 494.0 | 234.7 | 0.475 |
| acc | 1069.3 | 279.8 | 0.262 |

This supports the paper claim: double-adjoint mainly reduces memory, especially
for acc-stage training and longer windows. Runtime is comparable or slightly
faster in the recorded benchmark.

## Implementation Implication

This should be treated as an efficiency improvement, not a change to the loss
definition or evaluation metric.

Recommended implementation path:

```text
1. Keep the existing AD training path as default.
2. Add an optional gradient mode, e.g. --gradient_mode autograd|double_adjoint.
3. Use right-invariant SE23 residuals and analytic F/G for the outer adjoint.
4. Use local VJP / inner adjoint for the neural bias dynamics.
5. Validate gradients against normal AD on small windows before using it for
   long-window training.
```

Gradient validation should compare:

```text
loss value
parameter gradient cosine similarity
relative gradient error
max GPU memory
per-step runtime
```

Do not use this as an accuracy claim unless the final model is trained and
evaluated with the same loss, checkpoint selection, and EuRoC test metric.
