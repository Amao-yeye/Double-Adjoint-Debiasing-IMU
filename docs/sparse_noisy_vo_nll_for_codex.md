# Sparse noisy SE(3) pose NLL for optimizing diagonal IMU process noise

This note specifies how to replace an SE(3) pose MSE loss with a sparse Gaussian negative log-likelihood (NLL). The target implementation should optimize the diagonal IMU process-noise covariance for accelerometer and gyroscope noise.

Use this document as an implementation prompt for Codex.

---

## 0. Objective

The current pose loss is an SE(3) residual MSE:

$$
\mathcal L_{\mathrm{MSE}}
=
\frac12\sum_{i=0}^{M}\|r_i\|_2^2 .
$$

Replace it with the noisy pose NLL

$$
\mathcal L_{\mathrm{pose}}
=
\frac12
\bm r^\top
\left(H\bar P^sH^\top+W\right)^{-1}
\bm r
+
\frac12
\log\det\left(H\bar P^sH^\top+W\right),
$$

but **do not form the dense covariance** $\bar P^s$. Use the sparse block-tridiagonal form below.

The state perturbation is assumed to be

$$
\xi_i^s
=
\begin{bmatrix}
\delta\theta_i \\
\delta v_i \\
\delta p_i
\end{bmatrix}
\in\mathbb R^9,
$$

and the SE(3) pose residual is assumed to be

$$
r_i\in\mathbb R^6,
\qquad
r_i=[\delta\theta_i^\top,\delta p_i^\top]^\top .
$$

If the code uses the SE(3) tangent order $[\delta p^\top,\delta\theta^\top]^\top$, consistently reorder $r_i$, $H_i$, $W_i$, and the SE(3) Jacobian.

---

## 1. Learnable diagonal IMU process noise

Do **not** optimize each preintegrated covariance $Q_i^s$ independently. Optimize one diagonal IMU noise covariance and use it to build all preintegrated interval covariances.

The IMU noise vector is

$$
n_{\mathrm{imu}}
=
\begin{bmatrix}
n_a \\
n_g
\end{bmatrix}
\in\mathbb R^6,
$$

where both $n_a$ and $n_g$ have diagonal covariance.

Use learnable log-variance parameters:

$$
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
$$

Convert them to positive variances by

$$
q=\exp(\alpha)+\epsilon_q .
$$

Then

$$
Q_{\mathrm{imu}}
=
\operatorname{diag}
\left(
q_{a,x},q_{a,y},q_{a,z},q_{g,x},q_{g,y},q_{g,z}
\right).
$$

Important implementation detail: this assumes the IMU noise Jacobian $G_k$ uses column order

$$
[n_a^\top,n_g^\top]^\top .
$$

If the implementation uses $[n_g^\top,n_a^\top]^\top$, reorder the diagonal entries of $Q_{\mathrm{imu}}$ accordingly.

---

## 2. SE(3) pose residual and linearization

For each supervised timestamp $t_i^s$, compute the SE(3) pose residual

$$
r_i
=
\Log_{SE(3)}
\left(
Y_i^p\,\pi_p(\bar X_i^s)^{-1}
\right)^\vee
\in\mathbb R^6,
$$

where:

- $Y_i^p\in SE(3)$ is the supervised pose,
- $\bar X_i^s\in SE_2(3)$ is the propagated mean state,
- $\pi_p(\bar X_i^s)\in SE(3)$ extracts pose from the $SE_2(3)$ state.

Use the right-invariant state perturbation

$$
X_i^s(\xi_i^s)
=
\Exp_{SE_2(3)}((\xi_i^s)^\wedge)\bar X_i^s .
$$

The residual linearization is

$$
e_i(\xi_i^s)
\approx
r_i+H_i\xi_i^s .
$$

For the residual definition above, use

$$
H_i
=
-J_{r,SE(3)}^{-1}(r_i)\Pi_p,
$$

where $J_{r,SE(3)}^{-1}(r_i)$ is the inverse right Jacobian of SE(3), and

$$
\Pi_p
=
\begin{bmatrix}
I_3 & 0 & 0 \\
0 & 0 & I_3
\end{bmatrix}
\in\mathbb R^{6\times 9}.
$$

This $\Pi_p$ maps the $SE_2(3)$ tangent order $[\delta\theta^\top,\delta v^\top,\delta p^\top]^\top$ to the SE(3) pose tangent order $[\delta\theta^\top,\delta p^\top]^\top$.

Implementation options for $H_i$:

1. Preferred: use an existing Lie library function for $J_{r,SE(3)}^{-1}(r_i)$.
2. Safer fallback: compute $H_i$ as the Jacobian of the residual function with respect to the local perturbation $\xi_i^s$ at zero perturbation.
3. Add a finite-difference test to verify the sign of $H_i$.

---

## 3. Measurement covariance

For each pose residual, use a positive definite covariance

$$
W_i\in\mathbb R^{6\times 6},
\qquad
W_i\succ0 .
$$

Do **not** set $W_i=0$. The noisy sparse NLL requires $W_i^{-1}$.

If the current code only has an unweighted SE(3) loss, start with a fixed diagonal covariance:

$$
W_i
=
\operatorname{diag}
\left(
\sigma_\theta^2,
\sigma_\theta^2,
\sigma_\theta^2,
\sigma_p^2,
\sigma_p^2,
\sigma_p^2
\right).
$$

Here $\sigma_\theta^2$ weights the rotational residual and $\sigma_p^2$ weights the translational residual.

---

## 4. Preintegrated transition and covariance

Between supervised timestamps, use the Gaussian Markov model

$$
\xi_{i+1}^s
=
\Phi_i^s\xi_i^s+w_i^s,
\qquad
w_i^s\sim\mathcal N(0,Q_i^s),
$$

where

$$
\Phi_i^s\in\mathbb R^{9\times 9},
\qquad
Q_i^s\in\mathbb R^{9\times 9}.
$$

The covariance $Q_i^s$ must be computed from the diagonal IMU noise covariance $Q_{\mathrm{imu}}$.

For one IMU step $k$,

$$
\xi_{k+1}
=
\Phi_k\xi_k+G_kn_k,
\qquad
n_k\sim\mathcal N(0,Q_{\mathrm{imu}}).
$$

The per-step covariance contribution is

$$
Q_k=G_kQ_{\mathrm{imu}}G_k^\top .
$$

If the existing implementation treats $Q_{\mathrm{imu}}$ as a continuous-time noise density, preserve the existing discretization convention and do not multiply by $\Delta t$ twice. The formula above assumes $G_k$ already contains the required time discretization.

For each supervised interval, compute $\Phi_i^s$ and $Q_i^s$ by forward covariance propagation:

```python
Phi_acc = I_9
Q_acc = zeros(9, 9)

for k in imu_indices_between_supervised_times_i:
    # xi_{k+1} = Phi_k xi_k + G_k n_k
    Q_step = G_k @ Q_imu @ G_k.T

    Q_acc = Phi_k @ Q_acc @ Phi_k.T + Q_step
    Phi_acc = Phi_k @ Phi_acc

Phi_s[i] = Phi_acc
Q_s[i] = sym(Q_acc) + jitter_Q * I_9
```

Use

```python
def sym(A):
    return 0.5 * (A + A.T)
```

for numerical symmetry.

---

## 5. Final sparse noisy pose NLL

The sparse NLL to implement is

$$
\boxed{
\begin{aligned}
\mathcal L_{\mathrm{pose}}
=
&
\frac12
\sum_{i=0}^{M}
r_i^\top W_i^{-1}r_i
-
\frac12
\sum_{i=0}^{M}
\gamma_i^\top z_i
+
\frac12
\sum_{i=0}^{M}
\log\det W_i
\\
&
+
\frac12
\sum_{i=0}^{M}
\log\det D_i
+
\frac12
\log\det P_0^s
+
\frac12
\sum_{i=0}^{M-1}
\log\det Q_i^s .
\end{aligned}
}
$$

where

$$
\gamma_i
=
H_i^\top W_i^{-1}r_i
\in\mathbb R^9,
$$

and

$$
\Omega_i
=
H_i^\top W_i^{-1}H_i
\in\mathbb R^{9\times 9}.
$$

The vector $z=[z_0^\top,\ldots,z_M^\top]^\top$ is obtained by solving

$$
Kz=\gamma,
\qquad
\gamma=[\gamma_0^\top,\ldots,\gamma_M^\top]^\top .
$$

Do not form $K^{-1}$.

---

## 6. Block-tridiagonal matrix K

The matrix $K$ is block tridiagonal with $9\times9$ blocks.

Define diagonal blocks $A_i$ and off-diagonal blocks $B_i$.

For $i=0$,

$$
A_0
=
(P_0^s)^{-1}
+(
\Phi_0^s
)^\top
(Q_0^s)^{-1}
\Phi_0^s
+
\Omega_0 .
$$

For $i=1,\ldots,M-1$,

$$
A_i
=
(Q_{i-1}^s)^{-1}
+(
\Phi_i^s
)^\top
(Q_i^s)^{-1}
\Phi_i^s
+
\Omega_i .
$$

For $i=M$,

$$
A_M
=
(Q_{M-1}^s)^{-1}
+
\Omega_M .
$$

For $i=0,\ldots,M-1$,

$$
B_i
=
-(\Phi_i^s)^\top(Q_i^s)^{-1} .
$$

Then

$$
K_{i,i}=A_i,
\qquad
K_{i,i+1}=B_i,
\qquad
K_{i+1,i}=B_i^\top .
$$

---

## 7. Block LDL recursion for log det K

Compute the block Schur complements

$$
D_0=A_0,
$$

$$
D_i
=
A_i-B_{i-1}^\top D_{i-1}^{-1}B_{i-1},
\qquad
 i=1,\ldots,M .
$$

Then

$$
\log\det K
=
\sum_{i=0}^{M}\log\det D_i .
$$

Implementation rules:

- Do not explicitly invert $D_i$.
- Use Cholesky solves.
- Use Cholesky log-determinants.

For an SPD matrix $S$:

```python
L = torch.linalg.cholesky(S)
logdet_S = 2.0 * torch.sum(torch.log(torch.diagonal(L)))
solve_S_B = torch.cholesky_solve(B, L)
```

---

## 8. Solve K z = gamma with the same block LDL factors

After computing all $D_i$, solve $Kz=\gamma$ by block forward/back substitution.

Forward solve:

$$
y_0=\gamma_0 .
$$

For $i=1,\ldots,M$,

$$
y_i
=
\gamma_i
-
B_{i-1}^\top D_{i-1}^{-1}y_{i-1} .
$$

Diagonal solve:

$$
u_i=D_i^{-1}y_i,
\qquad
 i=0,\ldots,M .
$$

Backward solve:

$$
z_M=u_M .
$$

For $i=M-1,\ldots,0$,

$$
z_i
=
u_i-D_i^{-1}B_i z_{i+1} .
$$

Then

$$
\gamma^\top K^{-1}\gamma
=
\gamma^\top z
=
\sum_{i=0}^{M}\gamma_i^\top z_i .
$$

---

## 9. PyTorch-style implementation pseudocode

```python
import torch


def sym(A):
    return 0.5 * (A + A.transpose(-1, -2))


def logdet_from_cholesky(L):
    return 2.0 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)))


def sparse_noisy_se3_pose_nll(
    residuals,      # list length M+1, each r_i shape [6]
    jacobians,      # list length M+1, each H_i shape [6, 9]
    W_blocks,       # list length M+1, each W_i shape [6, 6]
    Phi_blocks,     # list length M, each Phi_i^s shape [9, 9]
    Q_blocks,       # list length M, each Q_i^s shape [9, 9]
    P0,             # P_0^s, shape [9, 9]
    jitter=1e-8,
):
    M = len(residuals) - 1
    dtype = residuals[0].dtype
    device = residuals[0].device
    I9 = torch.eye(9, dtype=dtype, device=device)
    I6 = torch.eye(6, dtype=dtype, device=device)

    # ------------------------------------------------------------
    # 1. Compute gamma_i, Omega_i, residual term, and logdet W_i
    # ------------------------------------------------------------
    gamma = []
    Omega = []
    residual_term = residuals[0].new_tensor(0.0)
    logdet_W_term = residuals[0].new_tensor(0.0)

    for i in range(M + 1):
        r_i = residuals[i]      # [6]
        H_i = jacobians[i]      # [6, 9]
        W_i = sym(W_blocks[i]) + jitter * I6

        chol_W = torch.linalg.cholesky(W_i)
        W_inv_r = torch.cholesky_solve(r_i[:, None], chol_W)[:, 0]
        W_inv_H = torch.cholesky_solve(H_i, chol_W)

        residual_term = residual_term + r_i @ W_inv_r
        logdet_W_term = logdet_W_term + logdet_from_cholesky(chol_W)

        gamma_i = H_i.T @ W_inv_r
        Omega_i = H_i.T @ W_inv_H

        gamma.append(gamma_i)
        Omega.append(sym(Omega_i))

    # ------------------------------------------------------------
    # 2. Compute P0^{-1}, Q_i^{-1}, logdet P0, and logdet Q_i
    # ------------------------------------------------------------
    P0 = sym(P0) + jitter * I9
    chol_P0 = torch.linalg.cholesky(P0)
    P0_inv = torch.cholesky_solve(I9, chol_P0)
    logdet_P0 = logdet_from_cholesky(chol_P0)

    Q_inv = []
    logdet_Q_term = residuals[0].new_tensor(0.0)

    for i in range(M):
        Q_i = sym(Q_blocks[i]) + jitter * I9
        chol_Q = torch.linalg.cholesky(Q_i)
        Q_i_inv = torch.cholesky_solve(I9, chol_Q)

        Q_inv.append(sym(Q_i_inv))
        logdet_Q_term = logdet_Q_term + logdet_from_cholesky(chol_Q)

    # ------------------------------------------------------------
    # 3. Build block-tridiagonal K through A_i and B_i
    # ------------------------------------------------------------
    A = []
    B = []

    # A_0
    Phi_0 = Phi_blocks[0]
    A0 = P0_inv + Phi_0.T @ Q_inv[0] @ Phi_0 + Omega[0]
    A.append(sym(A0))

    # A_i, i = 1, ..., M-1
    for i in range(1, M):
        Phi_i = Phi_blocks[i]
        Ai = Q_inv[i - 1] + Phi_i.T @ Q_inv[i] @ Phi_i + Omega[i]
        A.append(sym(Ai))

    # A_M
    AM = Q_inv[M - 1] + Omega[M]
    A.append(sym(AM))

    # B_i, i = 0, ..., M-1
    for i in range(M):
        Phi_i = Phi_blocks[i]
        Bi = -Phi_i.T @ Q_inv[i]
        B.append(Bi)

    # ------------------------------------------------------------
    # 4. Block LDL recursion: D_i and logdet K
    # ------------------------------------------------------------
    D_chol = []

    D0 = sym(A[0]) + jitter * I9
    chol_D0 = torch.linalg.cholesky(D0)
    D_chol.append(chol_D0)

    logdet_K = logdet_from_cholesky(chol_D0)

    for i in range(1, M + 1):
        # X = D_{i-1}^{-1} B_{i-1}
        X = torch.cholesky_solve(B[i - 1], D_chol[i - 1])

        Di = A[i] - B[i - 1].T @ X
        Di = sym(Di) + jitter * I9

        chol_Di = torch.linalg.cholesky(Di)
        D_chol.append(chol_Di)
        logdet_K = logdet_K + logdet_from_cholesky(chol_Di)

    # ------------------------------------------------------------
    # 5. Solve K z = gamma using the LDL factors
    # ------------------------------------------------------------

    # Forward solve
    y = [None] * (M + 1)
    y[0] = gamma[0]

    for i in range(1, M + 1):
        tmp = torch.cholesky_solve(y[i - 1][:, None], D_chol[i - 1])[:, 0]
        y[i] = gamma[i] - B[i - 1].T @ tmp

    # Diagonal solve
    u = [None] * (M + 1)
    for i in range(M + 1):
        u[i] = torch.cholesky_solve(y[i][:, None], D_chol[i])[:, 0]

    # Backward solve
    z = [None] * (M + 1)
    z[M] = u[M]

    for i in reversed(range(M)):
        rhs = B[i] @ z[i + 1]
        tmp = torch.cholesky_solve(rhs[:, None], D_chol[i])[:, 0]
        z[i] = u[i] - tmp

    gamma_z_term = residuals[0].new_tensor(0.0)
    for i in range(M + 1):
        gamma_z_term = gamma_z_term + gamma[i] @ z[i]

    # ------------------------------------------------------------
    # 6. Sparse noisy NLL
    # ------------------------------------------------------------
    loss = (
        0.5 * residual_term
        - 0.5 * gamma_z_term
        + 0.5 * logdet_W_term
        + 0.5 * logdet_K
        + 0.5 * logdet_P0
        + 0.5 * logdet_Q_term
    )

    return loss
```

---

## 10. Build Q_i^s from diagonal n_a and n_g covariance

Use the following routine to produce `Phi_blocks` and `Q_blocks` for the NLL above.

```python
def build_preintegrated_blocks(
    imu_intervals,      # list of IMU index ranges, one range per supervised interval
    Phi_steps,          # list/dict of Phi_k, each shape [9, 9]
    G_steps,            # list/dict of G_k, each shape [9, 6]
    logvar_imu,         # shape [6], learnable
    eps_var=1e-8,
    jitter_Q=1e-8,
):
    """
    logvar_imu order:
        [n_a_x, n_a_y, n_a_z, n_g_x, n_g_y, n_g_z]

    If G_k uses noise order [n_g, n_a], reorder logvar_imu before forming Q_imu.
    """
    dtype = logvar_imu.dtype
    device = logvar_imu.device
    I9 = torch.eye(9, dtype=dtype, device=device)

    var_imu = torch.exp(logvar_imu) + eps_var
    Q_imu = torch.diag(var_imu)

    Phi_blocks = []
    Q_blocks = []

    for interval in imu_intervals:
        Phi_acc = I9
        Q_acc = torch.zeros((9, 9), dtype=dtype, device=device)

        for k in interval:
            Phi_k = Phi_steps[k]
            G_k = G_steps[k]

            Q_step = G_k @ Q_imu @ G_k.T

            Q_acc = Phi_k @ Q_acc @ Phi_k.T + Q_step
            Phi_acc = Phi_k @ Phi_acc

        Q_acc = sym(Q_acc) + jitter_Q * I9

        Phi_blocks.append(Phi_acc)
        Q_blocks.append(Q_acc)

    return Phi_blocks, Q_blocks
```

---

## 11. Q-only optimization objective

The exact sparse NLL is

$$
\begin{aligned}
\mathcal L_{\mathrm{pose}}
=
&
\frac12
\sum_i
r_i^\top W_i^{-1}r_i
-
\frac12
\sum_i
\gamma_i^\top z_i
+
\frac12
\sum_i
\log\det W_i
\\
&
+
\frac12
\sum_i
\log\det D_i
+
\frac12
\log\det P_0^s
+
\frac12
\sum_i
\log\det Q_i^s .
\end{aligned}
$$

When optimizing only the IMU process-noise parameters `logvar_imu`, the following terms are constant with respect to `logvar_imu`, assuming the residuals, Jacobians, measurement covariances, and $P_0^s$ are fixed:

$$
\frac12\sum_i r_i^\top W_i^{-1}r_i,
\qquad
\frac12\sum_i\log\det W_i,
\qquad
\frac12\log\det P_0^s .
$$

Therefore, the effective Q-only objective is

$$
\mathcal L_{Q\text{-only}}
=
-
\frac12
\sum_{i=0}^{M}
\gamma_i^\top z_i
+
\frac12
\sum_{i=0}^{M}
\log\det D_i
+
\frac12
\sum_{i=0}^{M-1}
\log\det Q_i^s .
$$

However, implement the full NLL first and only drop constants after verifying numerical correctness.

---

## 12. Numerical and implementation requirements

1. Never explicitly compute matrix inverses. Use Cholesky solves.

2. Symmetrize every covariance and Schur complement:

```python
A = 0.5 * (A + A.T)
```

3. Add small jitter before Cholesky:

```python
A = A + jitter * I
```

4. Keep all covariance matrices positive definite:

$$
W_i\succ0,
\qquad
P_0^s\succ0,
\qquad
Q_i^s\succ0 .
$$

5. The learnable IMU covariance must remain diagonal:

```python
Q_imu = torch.diag(torch.exp(logvar_imu) + eps_var)
```

Do not introduce off-diagonal entries for $n_a$ or $n_g$.

6. The NLL omits the constant Gaussian normalization term

$$
\frac12 d\log(2\pi),
$$

because it does not affect optimization.

7. For batching multiple trajectories, compute one NLL per trajectory and average or sum over the batch.

8. Validate the implementation by comparing the sparse NLL to the dense NLL on a short trajectory. For small $M$, build dense $\bar P^s$ or dense precision/covariance explicitly and verify that both losses match up to numerical tolerance.

9. Validate the sign and tangent convention of $H_i$ by finite differences. For a small random perturbation $\delta\xi$, verify

$$
e_i(\delta\xi)-e_i(0)
\approx
H_i\delta\xi .
$$

---

## 13. Expected learnable parameters

The final learnable parameter vector should be exactly six scalars:

```python
logvar_imu = torch.nn.Parameter(torch.zeros(6))
```

with order

```text
[n_a_x, n_a_y, n_a_z, n_g_x, n_g_y, n_g_z]
```

unless the existing IMU noise Jacobian uses a different noise order.

The corresponding covariance is

```python
var_imu = torch.exp(logvar_imu) + eps_var
Q_imu = torch.diag(var_imu)
```

This is the only process-noise covariance that should be optimized. All interval covariances $Q_i^s$ must be produced by preintegration from this diagonal $Q_{\mathrm{imu}}$.
