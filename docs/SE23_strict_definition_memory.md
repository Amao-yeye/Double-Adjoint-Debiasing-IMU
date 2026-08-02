# SE23 Strict Definition Memory

Date: 2026-05-21

This document fixes the SE23 loss definition used for the paper experiments.

## Paper Definition

The paper uses the strict right-invariant SE_2(3) residual:

```text
r_i = Log( X_i^gt * (X_i_hat)^(-1) )^vee
```

with block order:

```text
r_i = [phi_i, rho_v_i, rho_p_i]
```

Equivalently:

```text
phi_i = Log( R_i^gt * R_i_hat^T )^vee

rho_v_i = J_l^{-1}(phi_i) *
          ( v_i^gt - R_i^gt * R_i_hat^T * v_i_hat )

rho_p_i = J_l^{-1}(phi_i) *
          ( p_i^gt - R_i^gt * R_i_hat^T * p_i_hat )
```

This is the implementation path:

```text
external/Debias_IMU/BiasDy/plato/losses.py::se23_gt_right_residual
```

Training should use:

```text
--mse_residual_mode right
```

## MSE Loss

For SE23-MSE, the paper loss is:

```text
L = 1e6 * 0.5 * mean( || W r_i ||^2 )
```

For the accepted strict acc-stage setting:

```text
rotation weight = 0
velocity weight = 1
position weight = 10
```

Even when rotation weight is zero in the BA/acc stage, `rho_v` and `rho_p` are
still computed from the strict SE23 group residual, so the translational blocks
retain the rotation coupling from `R_i^gt * R_i_hat^T`.

## Analytical Gradient

For analytical/double-adjoint training, the residual Jacobian follows BCH:

```text
H_i = -dexp^{-1}_{-r_i}
```

In code:

```text
SE23: H_i = -SEn3rightJaco_inv(r_i)
SO3:  H_i = -SO3rightJacoInv(r_i)
```

The invariant propagation order is:

```text
SE23 residual/state:   [phi, rho_v, rho_p]
se23_Gd columns:       [n_a, n_g]
adjoint bias gradient: [ba, bg]
```

Therefore acc-stage double-adjoint training uses:

```text
analytic.bias_gradient[..., 0:3]
```

for BA gradients.

## What Is Not The Paper Definition

The following residual modes are not the paper's main SE23-MSE definition:

```text
log
global
hybrid
```

They may appear in older experiment records as engineering attempts or ablation
branches, but they should not be described as the paper's SE23 loss design.

In particular, older no-gate `hybrid` results are historical references. The
paper-facing strict SE23-MSE result should use `mse_residual_mode=right`.

## Current Strict SE23 Result

Reference strict SE23-MSE AD result:

```text
external/Debias_IMU/results/Euroc_PLATO_strict_right_ba_rate_tanh1e2_fromV1P10_lr1e6_w64_b1000_e101_eval_best
```

Analytical/double-adjoint strict SE23-MSE result:

```text
external/Debias_IMU/results/Euroc_PLATO_strict_right_double_adjoint_ba_rate_tanh1e2_fromV1P10_lr1e6_w64_b1000_e101_eval_best
```

Mean metrics:

| Run | AOE deg | Velocity RMSE | Position RMSE / APE m |
| --- | ---: | ---: | ---: |
| Baseline w64 | 3.392720866 | 16.372752047 | 683.675057983 |
| Strict SE23-MSE AD | 3.043793869 | 16.384823990 | 644.772181702 |
| Strict SE23-MSE double-adjoint | 3.043793869 | 16.389042807 | 644.928808594 |

The strict SE23-MSE AD result improves over baseline by:

```text
AOE improvement: 10.284578%
APE improvement: 5.690258%
```

The strict SE23-MSE double-adjoint result improves over baseline by:

```text
AOE improvement: 10.284577%
APE improvement: 5.667349%
```

## NLL Note

For the paper-facing strict branch, sparse GT NLL should also be initialized
from the strict SE23-MSE checkpoint and should use the strict right SE23
residual:

```text
--nll_gt_residual_mode right
--mse_residual_mode right
```

Strict-right NLL analytical run:

```text
external/Debias_IMU/results/Euroc_PLATO_sparse_gt_nll_rightres_double_adjoint_from_strict_right_mse_w64_b1000_lr1e6_anchor001_freezeA_e51
```

Recommended strict-right NLL checkpoint by APE:

```text
external/Debias_IMU/results/Euroc_PLATO_sparse_gt_nll_rightres_double_adjoint_from_strict_right_mse_w64_b1000_lr1e6_anchor001_freezeA_e51/NLL_weights/model_epoch_40.pt
```

Evaluation:

```text
external/Debias_IMU/results/Euroc_PLATO_sparse_gt_nll_rightres_double_adjoint_from_strict_right_mse_w64_b1000_lr1e6_anchor001_freezeA_eval_e40
```

Mean metrics:

| Run | AOE deg | Velocity RMSE | Position RMSE / APE m |
| --- | ---: | ---: | ---: |
| Baseline w64 | 3.392720866 | 16.372752047 | 683.675057983 |
| Strict SE23-MSE AD | 3.043793869 | 16.384823990 | 644.772181702 |
| Strict-right NLL double-adjoint e40 | 3.075663114 | 15.752899933 | 631.278599548 |

Relative to baseline:

```text
AOE improvement: 9.345235%
APE improvement: 7.663942%
```

Relative to strict SE23-MSE AD:

```text
AOE: 1.047024% worse
Velocity RMSE: 3.856764% better
APE: 2.092767% better
```
