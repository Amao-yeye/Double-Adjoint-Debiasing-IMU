# Implement sparse full-state ground-truth NLL on `SE_2(3)`

## Goal

Implement the sparse negative log-likelihood (NLL) for **noise-free full-state ground-truth supervision**.

This is the ground-truth counterpart of the noisy SE(3) pose NLL. It should be used only when the supervision is available in the **same state space** as the propagated trajectory, i.e.

\[
    X_i^{\mathrm{gt}} \in SE_2(3),
    \qquad
    \bar X_i^s \in SE_2(3).
\]

Do **not** use this loss for pose-only SE(3) supervision. If the supervision only gives poses, use the noisy pose NLL with a small positive definite measurement covariance \(W_i\), not the zero-noise full-state formula.

The purpose is to replace a full-state MSE-style residual loss with an exact sparse Gaussian NLL induced by IMU preintegration. The dense trajectory covariance must never be formed.

The state perturbation order is assumed to be

\[
    \xi_i^s =
    \begin{bmatrix}
        \delta\theta_i \\
        \delta v_i \\
        \delta p_i
    \end{bmatrix}
    \in \mathbb R^9 .
\]

If the code uses a different tangent order, adjust all matrices consistently.

---

## 1. Learnable diagonal IMU process-noise covariance

The learnable quantity is the diagonal covariance of the IMU noise

\[
    n_{\mathrm{imu}}
    =
    \begin{bmatrix}
        n_a \\
        n_g
    \end{bmatrix},
\]

where both accelerometer noise \(n_a\) and gyroscope noise \(n_g\) have diagonal covariance.

Do **not** optimize each preintegrated covariance \(Q_i^s\) independently. Instead, optimize one diagonal IMU covariance and use it to build all preintegrated covariances.

Use log-variance parameters

\[
    \alpha
    =
    \begin{bmatrix}
        \alpha_{a,x} &
        \alpha_{a,y} &
        \alpha_{a,z} &
        \alpha_{g,x} &
        \alpha_{g,y} &
        \alpha_{g,z}
    \end{bmatrix}^\top .
\]

Convert them to positive variances by

\[
    q = \exp(\alpha) + \epsilon_q .
\]

Then

\[
    Q_{\mathrm{imu}}
    =
    \operatorname{diag}
    \left(
        q_{a,x},q_{a,y},q_{a,z},
        q_{g,x},q_{g,y},q_{g,z}
    \right).
\]

This assumes the columns of the IMU noise Jacobian \(G_k\) are ordered as

\[
    [n_a^\top,n_g^\top]^\top .
\]

If the implementation uses \([n_g^\top,n_a^\top]^\top\), reorder the diagonal entries of \(Q_{\mathrm{imu}}\).

---

## 2. Full-state ground-truth residual on `SE_2(3)`

For each supervised timestamp \(t_i^s\), compute the nominal full-state residual

\[
    r_i^{\mathrm{gt}}
    =
    \Log_{SE_2(3)}
    \left(
        X_i^{\mathrm{gt}}(\bar X_i^s)^{-1}
    \right)^\vee
    \in \mathbb R^9 .
\]

The propagated state uses the right-invariant perturbation

\[
    X_i^s(\xi_i^s)
    =
    \Exp_{SE_2(3)}((\xi_i^s)^\wedge)\bar X_i^s .
\]

The perturbed residual is

\[
\begin{aligned}
    e_i^{\mathrm{gt}}(\xi_i^s)
    &=
    \Log_{SE_2(3)}
    \left(
        X_i^{\mathrm{gt}}
        (X_i^s(\xi_i^s))^{-1}
    \right)^\vee \\
    &=
    \Log_{SE_2(3)}
    \left(
        \Exp((r_i^{\mathrm{gt}})^\wedge)
        \Exp(- (\xi_i^s)^\wedge)
    \right)^\vee .
\end{aligned}
\]

Using the BCH expansion,

\[
    e_i^{\mathrm{gt}}(\xi_i^s)
    \approx
    r_i^{\mathrm{gt}}
    +
    H_i^{\mathrm{gt}}\xi_i^s,
    \qquad
    H_i^{\mathrm{gt}}
    =
    -J_{r,SE_2(3)}^{-1}(r_i^{\mathrm{gt}}),
\]

where \(J_{r,SE_2(3)}(\cdot)\) is the right Jacobian on \(SE_2(3)\).

Equivalently, if the code uses `dexp`, then

\[
    J_{r}^{-1}(r)
    =
    (\mathrm{dexp}^{R}_{r})^{-1}.
\]

If the available `dexp` implementation is the left Jacobian \(J_l(r)=\mathrm{dexp}_r^L\), use

\[
    J_r(r)=J_l(-r),
    \qquad
    J_r^{-1}(r)=J_l^{-1}(-r).
\]

The transformed residual used by the zero-noise NLL is

\[
    \tilde r_i^{\mathrm{gt}}
    =
    (H_i^{\mathrm{gt}})^{-1}r_i^{\mathrm{gt}} .
\]

For this specific right-invariant residual,

\[
    (H_i^{\mathrm{gt}})^{-1}
    =
    -J_r(r_i^{\mathrm{gt}}),
    \qquad
    J_r(r_i^{\mathrm{gt}})r_i^{\mathrm{gt}}
    =
    r_i^{\mathrm{gt}},
\]

so

\[
    \tilde r_i^{\mathrm{gt}}=-r_i^{\mathrm{gt}}.
\]

For the quadratic term, using \(\tilde r_i^{\mathrm{gt}}\) or \(r_i^{\mathrm{gt}}\) gives the same value because all residual blocks are multiplied by the same sign. However, the exact NLL also contains the Jacobian log-determinant term

\[
    \sum_{i=0}^{M}\log\left|\det H_i^{\mathrm{gt}}\right|.
\]

When optimizing **only** the IMU noise covariance, this term is constant and can be dropped.

---

## 3. Preintegrated transition and covariance

Between supervised timestamps, the downsampled error process is

\[
    \xi_0^s\sim\mathcal N(0,P_0^s),
    \qquad
    \xi_{i+1}^s
    =
    \Phi_i^s\xi_i^s+w_i^s,
    \qquad
    w_i^s\sim\mathcal N(0,Q_i^s).
\]

Here

\[
    \Phi_i^s\in\mathbb R^{9\times 9},
    \qquad
    Q_i^s\in\mathbb R^{9\times 9}.
\]

The preintegrated covariance \(Q_i^s\) must be built from the diagonal IMU covariance \(Q_{\mathrm{imu}}\).

For one IMU step,

\[
    \xi_{k+1}
    =
    \Phi_k\xi_k+G_kn_k,
    \qquad
    n_k\sim\mathcal N(0,Q_{\mathrm{imu}}).
\]

The per-step covariance contribution is

\[
    Q_k = G_k Q_{\mathrm{imu}}G_k^\top .
\]

For each supervised interval \([t_i^s,t_{i+1}^s]\), compute \(\Phi_i^s\) and \(Q_i^s\) by forward propagation:

```python
Phi_acc = I_9
Q_acc = zeros(9, 9)

for k in imu_indices_between_supervised_times_i:
    Q_step = G_k @ Q_imu @ G_k.T

    Q_acc = Phi_k @ Q_acc @ Phi_k.T + Q_step
    Phi_acc = Phi_k @ Phi_acc

Phi_s[i] = Phi_acc
Q_s[i] = sym(Q_acc) + jitter_Q * I_9
```

Use

```python
sym(A) = 0.5 * (A + A.T)
```

for numerical symmetry.

Important: if the existing implementation treats \(Q_{\mathrm{imu}}\) as a continuous-time noise density, follow the existing discretization convention and do not multiply by \(\Delta t\) twice. The formula above assumes \(G_k\) already contains the correct time discretization.

---

## 4. Preferred sparse GT NLL: transition-factor form

The sparse ground-truth NLL can be evaluated without constructing the dense covariance \(\bar P^s\).

Use the transformed residuals \(\tilde r_i^{\mathrm{gt}}\). Define the transition residual

\[
    d_i
    =
    \tilde r_{i+1}^{\mathrm{gt}}
    -
    \Phi_i^s\tilde r_i^{\mathrm{gt}},
    \qquad
    i=0,\ldots,M-1 .
\]

Then the exact sparse NLL is

\[
\boxed{
\begin{aligned}
    \mathcal L_{\mathrm{gt}}
    =
    &
    \frac12
    (\tilde r_0^{\mathrm{gt}})^\top
    (P_0^s)^{-1}
    \tilde r_0^{\mathrm{gt}}
    +
    \frac12
    \sum_{i=0}^{M-1}
    d_i^\top
    (Q_i^s)^{-1}
    d_i
    \\
    &+
    \frac12\log\det P_0^s
    +
    \frac12
    \sum_{i=0}^{M-1}
    \log\det Q_i^s
    +
    \sum_{i=0}^{M}
    \log\left|
        \det H_i^{\mathrm{gt}}
    \right| .
\end{aligned}
}
\]

If optimizing only the diagonal IMU process-noise covariance and \(P_0^s\) is fixed, the Q-dependent objective is

\[
\boxed{
    \mathcal L_{Q\text{-only}}
    =
    \frac12
    \sum_{i=0}^{M-1}
    d_i^\top
    (Q_i^s)^{-1}
    d_i
    +
    \frac12
    \sum_{i=0}^{M-1}
    \log\det Q_i^s .
}
\]

The terms

\[
    \frac12
    (\tilde r_0^{\mathrm{gt}})^\top(P_0^s)^{-1}\tilde r_0^{\mathrm{gt}},
    \qquad
    \frac12\log\det P_0^s,
    \qquad
    \sum_i\log|\det H_i^{\mathrm{gt}}|
\]

are constant with respect to \(Q_{\mathrm{imu}}\), as long as the propagated mean trajectory and residuals are fixed.

---

## 5. Equivalent block-tridiagonal information form

The joint covariance of the stacked perturbation

\[
    \bm\xi^s=[(\xi_0^s)^\top,\ldots,(\xi_M^s)^\top]^\top
\]

is dense, but the joint information matrix

\[
    \bar\Lambda^s=(\bar P^s)^{-1}
\]

is block tridiagonal.

Its nonzero blocks are

\[
\begin{aligned}
    \bar\Lambda^s_{0,0}
    &=
    (P_0^s)^{-1}
    +
    (\Phi_0^s)^\top(Q_0^s)^{-1}\Phi_0^s, \\
    \bar\Lambda^s_{i,i}
    &=
    (Q_{i-1}^s)^{-1}
    +
    (\Phi_i^s)^\top(Q_i^s)^{-1}\Phi_i^s,
    \qquad i=1,\ldots,M-1, \\
    \bar\Lambda^s_{M,M}
    &=
    (Q_{M-1}^s)^{-1}, \\
    \bar\Lambda^s_{i,i+1}
    &=
    -(\Phi_i^s)^\top(Q_i^s)^{-1},
    \qquad
    \bar\Lambda^s_{i+1,i}
    =
    -(Q_i^s)^{-1}\Phi_i^s .
\end{aligned}
\]

The equivalent block-sum NLL is

\[
\begin{aligned}
    \mathcal L_{\mathrm{gt}}
    =
    &
    \frac12
    \sum_{i=0}^{M}
    (\tilde r_i^{\mathrm{gt}})^\top
    \bar\Lambda^s_{i,i}
    \tilde r_i^{\mathrm{gt}}
    +
    \sum_{i=0}^{M-1}
    (\tilde r_i^{\mathrm{gt}})^\top
    \bar\Lambda^s_{i,i+1}
    \tilde r_{i+1}^{\mathrm{gt}}
    \\
    &+
    \frac12\log\det P_0^s
    +
    \frac12
    \sum_{i=0}^{M-1}
    \log\det Q_i^s
    +
    \sum_{i=0}^{M}
    \log\left|
        \det H_i^{\mathrm{gt}}
    \right| .
\end{aligned}
\]

For implementation, the transition-factor form in Section 4 is simpler and less error-prone. It is mathematically equivalent to the block-tridiagonal information form.

---

## 6. Implementation pseudocode

### 6.1 Build preintegrated blocks from diagonal IMU noise

```python
def build_preintegrated_blocks(
    imu_intervals,      # list of IMU index ranges for each supervised interval
    Phi_steps,          # list/dict of Phi_k, each shape [9, 9]
    G_steps,            # list/dict of G_k, each shape [9, 6]
    logvar_imu,         # shape [6], learnable
    eps_var=1e-8,
    jitter_Q=1e-8,
):
    # Order: [n_a_x, n_a_y, n_a_z, n_g_x, n_g_y, n_g_z]
    var_imu = torch.exp(logvar_imu) + eps_var
    Q_imu = torch.diag(var_imu)

    I9 = torch.eye(9, device=logvar_imu.device, dtype=logvar_imu.dtype)

    Phi_blocks = []
    Q_blocks = []

    for interval in imu_intervals:
        Phi_acc = I9
        Q_acc = torch.zeros_like(I9)

        for k in interval:
            Phi_k = Phi_steps[k]
            G_k = G_steps[k]

            Q_step = G_k @ Q_imu @ G_k.T

            Q_acc = Phi_k @ Q_acc @ Phi_k.T + Q_step
            Phi_acc = Phi_k @ Phi_acc

        Q_acc = 0.5 * (Q_acc + Q_acc.T) + jitter_Q * I9

        Phi_blocks.append(Phi_acc)
        Q_blocks.append(Q_acc)

    return Phi_blocks, Q_blocks
```

### 6.2 Sparse GT NLL

```python
def sym(A):
    return 0.5 * (A + A.T)


def logdet_from_cholesky(L):
    return 2.0 * torch.sum(torch.log(torch.diagonal(L)))


def sparse_gt_se23_nll(
    residuals_gt,       # list of r_i^gt, each shape [9]
    Phi_blocks,         # list of Phi_i^s, i=0..M-1, each shape [9, 9]
    Q_blocks,           # list of Q_i^s, i=0..M-1, each shape [9, 9]
    P0,                 # P_0^s, shape [9, 9]
    H_blocks=None,      # optional list of H_i^gt, each shape [9, 9]
    include_constants=True,
    q_only=False,
    jitter=1e-8,
):
    """
    Sparse full-state GT NLL.

    If q_only=True, returns only the terms depending on Q_blocks:
        0.5 * sum_i d_i^T Q_i^{-1} d_i
        + 0.5 * sum_i logdet(Q_i)

    Use q_only=True when optimizing only logvar_imu and P0/residuals/H are fixed.
    """
    M = len(residuals_gt) - 1
    device = residuals_gt[0].device
    dtype = residuals_gt[0].dtype
    I9 = torch.eye(9, device=device, dtype=dtype)

    # ------------------------------------------------------------
    # 1. Transformed residuals: tilde_r_i = H_i^{-1} r_i
    # ------------------------------------------------------------
    tilde_r = []
    logdet_H = torch.zeros((), device=device, dtype=dtype)

    if H_blocks is None:
        # For the right-invariant residual on SE_2(3), H_i = -J_r^{-1}(r_i),
        # hence H_i^{-1} r_i = -r_i. The global sign does not affect
        # the quadratic term, but use -r_i for consistency.
        for i in range(M + 1):
            tilde_r.append(-residuals_gt[i])
    else:
        for i in range(M + 1):
            H_i = H_blocks[i]
            r_i = residuals_gt[i]

            # H_i is not necessarily SPD, so use a general linear solve.
            tilde_i = torch.linalg.solve(H_i, r_i)
            tilde_r.append(tilde_i)

            # Exact NLL includes log |det H_i|.
            sign_i, logabsdet_i = torch.linalg.slogdet(H_i)
            logdet_H = logdet_H + logabsdet_i

    # ------------------------------------------------------------
    # 2. Optional prior term from P0
    # ------------------------------------------------------------
    loss = torch.zeros((), device=device, dtype=dtype)

    if not q_only:
        P0 = sym(P0) + jitter * I9
        chol_P0 = torch.linalg.cholesky(P0)
        P0_inv_r0 = torch.cholesky_solve(tilde_r[0][:, None], chol_P0)[:, 0]

        prior_quad = tilde_r[0] @ P0_inv_r0
        logdet_P0 = logdet_from_cholesky(chol_P0)

        loss = loss + 0.5 * prior_quad

        if include_constants:
            loss = loss + 0.5 * logdet_P0

    # ------------------------------------------------------------
    # 3. Transition terms from Q_i^s
    # ------------------------------------------------------------
    logdet_Q_total = torch.zeros((), device=device, dtype=dtype)
    transition_quad_total = torch.zeros((), device=device, dtype=dtype)

    for i in range(M):
        Phi_i = Phi_blocks[i]
        Q_i = sym(Q_blocks[i]) + jitter * I9

        d_i = tilde_r[i + 1] - Phi_i @ tilde_r[i]

        chol_Q = torch.linalg.cholesky(Q_i)
        Q_inv_d = torch.cholesky_solve(d_i[:, None], chol_Q)[:, 0]

        transition_quad_total = transition_quad_total + d_i @ Q_inv_d
        logdet_Q_total = logdet_Q_total + logdet_from_cholesky(chol_Q)

    loss = loss + 0.5 * transition_quad_total
    loss = loss + 0.5 * logdet_Q_total

    # ------------------------------------------------------------
    # 4. Exact measurement Jacobian log determinant
    # ------------------------------------------------------------
    if include_constants and (not q_only) and (H_blocks is not None):
        loss = loss + logdet_H

    return loss
```

---

## 7. What Codex should optimize

If the task is to optimize only the diagonal IMU process noise, the learnable parameter should be

```python
logvar_imu = torch.nn.Parameter(torch.zeros(6))
```

where the order is

```text
[n_a_x, n_a_y, n_a_z, n_g_x, n_g_y, n_g_z]
```

The training loop should be:

```python
Phi_blocks, Q_blocks = build_preintegrated_blocks(
    imu_intervals=imu_intervals,
    Phi_steps=Phi_steps,
    G_steps=G_steps,
    logvar_imu=logvar_imu,
)

loss = sparse_gt_se23_nll(
    residuals_gt=residuals_gt,
    Phi_blocks=Phi_blocks,
    Q_blocks=Q_blocks,
    P0=P0,
    H_blocks=None,          # OK for Q-only optimization
    include_constants=False,
    q_only=True,
)

loss.backward()
optimizer.step()
```

Use `H_blocks` only if the code needs the exact full NLL including the \(\log|\det H_i|\) term. For Q-only optimization, `H_blocks=None` is sufficient because \(H_i\) does not depend on \(Q_{\mathrm{imu}}\).

---

## 8. Numerical requirements

1. Never explicitly invert covariance matrices. Use Cholesky solves.

2. Symmetrize every covariance before Cholesky:

```python
A = 0.5 * (A + A.T)
```

3. Add a small jitter before Cholesky:

```python
A = A + jitter * I
```

4. Keep the IMU covariance diagonal:

```python
Q_imu = torch.diag(torch.exp(logvar_imu) + eps_var)
```

Do not introduce off-diagonal entries for \(n_a\) or \(n_g\).

5. The exact NLL omits the constant Gaussian normalization term

\[
    \frac12 d\log(2\pi),
\]

because it does not affect optimization.

6. For multiple trajectories, compute one sparse GT NLL per trajectory and average or sum over the batch.

7. If the optimizer drives variances to extreme values, clamp or regularize `logvar_imu`, for example:

```python
logvar_imu_clamped = torch.clamp(logvar_imu, min=-12.0, max=6.0)
```

or add a weak prior on `logvar_imu`.

---

## 9. Sanity checks

1. With fixed residuals and fixed \(\Phi_i^s\), increasing \(Q_i^s\) should reduce the quadratic term but increase the log-determinant term.

2. If all transition residuals \(d_i\) are close to zero, the optimum tends to smaller process noise unless regularized or lower-bounded by `eps_var`.

3. If transition residuals are large, the learned process noise should increase along the affected axes.

4. Compare the transition-factor form and the block-tridiagonal precision form on a short trajectory. They should match numerically up to floating-point error.

5. The loss should be differentiable with respect to `logvar_imu` through all `Q_blocks`.
