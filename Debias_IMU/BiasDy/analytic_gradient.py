from dataclasses import dataclass

import torch

import lie_algebra as Lie
from covariance import paper_invariant_error_matrices


@dataclass
class BiasAdjointGradient:
    """Analytic adjoint quantities for the invariant-error chain.

    Shapes use a leading time dimension T and optional batch dimensions.
    Bias/noise order is [ba, bg], matching target_CoRL26.pdf Section 6.
    """

    lambda_state: torch.Tensor
    bias_gradient: torch.Tensor
    F: torch.Tensor
    G: torch.Tensor
    state_gradient: torch.Tensor


def se23_invariant_residual(
    X_pred: torch.Tensor,
    X_ref: torch.Tensor,
    invariant_error: str = "right",
) -> torch.Tensor:
    """Return SE_2(3) residual in the [theta, v, p] tangent order.

    Args:
        X_pred: Predicted trajectory, shape (T, ..., 5, 5).
        X_ref: Supervisory/reference trajectory, shape (T, ..., 5, 5).
        invariant_error:
            "right": residual = Log(X_ref @ inv(X_pred)).
            "left": residual = Log(inv(X_ref) @ X_pred), matching SE23_loss.

    The "right" convention is the default because covariance.py and
    target_CoRL26.pdf use right-invariant SE_2(3) error dynamics.
    """
    _validate_time_major("X_pred", X_pred, trailing_shape=(5, 5))
    _validate_time_major("X_ref", X_ref, trailing_shape=(5, 5))
    if X_pred.shape != X_ref.shape:
        raise ValueError(f"X_pred and X_ref should have the same shape, got {tuple(X_pred.shape)} and {tuple(X_ref.shape)}")
    if invariant_error == "right":
        return Lie.SEn3log(X_ref @ Lie.SEn3inverse(X_pred, numerial_invese=True))
    if invariant_error == "left":
        return Lie.SEn3log(Lie.SEn3inverse(X_ref) @ X_pred)
    raise ValueError("invariant_error should be one of: right, left")


def mse_state_gradient(
    residual: torch.Tensor,
    weight: torch.Tensor | None = None,
    reduction: str = "mean",
    skip_initial: int = 0,
    residual_jacobian: str = "identity",
) -> torch.Tensor:
    """Analytic gradient dL/dxi for a quadratic residual loss.

    Computes the tangent-space gradient for:
        L = sum_k residual_k^T W residual_k
    or the corresponding mean reduction. With W omitted, W = I.

    Args:
        residual: Tangent residuals, shape (T, ..., 9).
        weight: Optional weighting matrix, shape (..., 9, 9) or (T, ..., 9, 9).
        reduction: "sum", "mean", or "none". "none" is equivalent to "sum"
            for the returned per-state gradient.
        skip_initial: Set the first `skip_initial` state gradients to zero.
    """
    _validate_time_major("residual", residual, trailing_shape=(9,))
    _validate_reduction(reduction)

    if weight is None:
        grad_r = 2.0 * residual
    else:
        weight = weight.to(device=residual.device, dtype=residual.dtype)
        if weight.shape[-2:] != (9, 9):
            raise ValueError(f"weight should have trailing shape (9, 9), got {tuple(weight.shape)}")
        grad_r = 2.0 * (weight @ residual.unsqueeze(-1)).squeeze(-1)

    if reduction == "mean":
        grad_r = grad_r / residual.numel()

    grad = apply_log_residual_jacobian_transpose(residual, grad_r, residual_jacobian)

    if skip_initial:
        _validate_skip_initial(skip_initial, residual.shape[0])
        grad = grad.clone()
        grad[:skip_initial] = 0.0
    return grad


def nll_state_gradient(
    residual: torch.Tensor,
    covariance: torch.Tensor,
    measurement_cov: torch.Tensor | None = None,
    H: torch.Tensor | None = None,
    jitter: float = 1e-6,
    reduction: str = "mean",
    skip_initial: int = 0,
    residual_jacobian: str = "identity",
) -> torch.Tensor:
    """Analytic dL/dxi for per-state Gaussian NLL in invariant coordinates.

    For each state, this returns the gradient of:
        0.5 * (r^T S^-1 r + logdet S)

    with respect to the full invariant state error xi. The logdet term has no
    direct xi gradient here because S is treated as fixed for this outer
    sensitivity helper. If H is provided, r = H xi and S = H P H^T + W.

    This is the per-state approximation currently used by Acc_train_SE23_NLL,
    not the full chain-structured likelihood from target_CoRL26.pdf.
    """
    _validate_time_major("residual", residual, trailing_shape=(9,))
    _validate_time_major("covariance", covariance, trailing_shape=(9, 9))
    _validate_reduction(reduction)
    if covariance.shape[0] != residual.shape[0]:
        raise ValueError("residual and covariance should have the same time length")

    xi = residual
    P = covariance.to(device=xi.device, dtype=xi.dtype)

    if H is None:
        r = xi
        S = P
    else:
        H = H.to(device=xi.device, dtype=xi.dtype)
        if H.shape[-1] != 9:
            raise ValueError(f"H should have trailing shape (..., residual_dim, 9), got {tuple(H.shape)}")
        r = (H @ xi.unsqueeze(-1)).squeeze(-1)
        S = H @ P @ H.transpose(-1, -2)

    if measurement_cov is not None:
        measurement_cov = measurement_cov.to(device=xi.device, dtype=xi.dtype)
        if measurement_cov.shape[-2:] != S.shape[-2:]:
            raise ValueError(
                f"measurement_cov should have trailing shape {tuple(S.shape[-2:])}, "
                f"got {tuple(measurement_cov.shape[-2:])}"
            )
        S = S + measurement_cov

    eye = torch.eye(S.shape[-1], device=xi.device, dtype=xi.dtype)
    S = 0.5 * (S + S.transpose(-1, -2)) + jitter * eye
    chol = torch.linalg.cholesky(S)
    solved = torch.cholesky_solve(r.unsqueeze(-1), chol).squeeze(-1)

    if H is None:
        grad_r = solved
    else:
        grad_r = (H.transpose(-1, -2) @ solved.unsqueeze(-1)).squeeze(-1)

    if reduction == "mean":
        grad_r = grad_r / residual.shape[0]

    grad = apply_log_residual_jacobian_transpose(residual, grad_r, residual_jacobian)

    if skip_initial:
        _validate_skip_initial(skip_initial, residual.shape[0])
        grad = grad.clone()
        grad[:skip_initial] = 0.0
    return grad


def apply_log_residual_jacobian_transpose(
    residual: torch.Tensor,
    grad_residual: torch.Tensor,
    residual_jacobian: str = "identity",
) -> torch.Tensor:
    """Map dL/dr to dL/dxi through the log residual Jacobian.

    For a residual r = Log(E), perturbing the group error gives first-order
    BCH forms:
        Log(Exp(delta) Exp(r)) ~= r + J_l^{-1}(r) delta
        Log(Exp(r) Exp(delta)) ~= r + J_r^{-1}(r) delta

    The adjoint recursion needs dL/dxi, so this applies H^T dL/dr with
    H = J_l^{-1}, J_r^{-1}, or I.
    """
    if residual_jacobian == "identity":
        return grad_residual
    if residual_jacobian == "left":
        H = Lie.SEn3leftJaco_inv(residual)
    elif residual_jacobian == "right":
        H = Lie.SEn3rightJaco_inv(residual)
    else:
        raise ValueError("residual_jacobian should be one of: identity, left, right")
    return (H.transpose(-1, -2) @ grad_residual.unsqueeze(-1)).squeeze(-1)


def invariant_transition_matrices(
    R: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    t: torch.Tensor | None = None,
    dt: float | torch.Tensor | None = None,
    gravity: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build discrete F_k, G_k for the right-invariant SE_2(3) error chain.

    Args:
        R: Nominal rotations, shape (T, ..., 3, 3).
        v: Nominal velocities, shape (T, ..., 3).
        p: Nominal positions, shape (T, ..., 3).
        t: Optional time vector with shape (T,). Used to compute dt.
        dt: Optional scalar or length T-1 transition duration.
        gravity: Optional gravity vector.

    Returns:
        F, G with shapes (T-1, ..., 9, 9) and (T-1, ..., 9, 6).
    """
    _validate_time_major("R", R, trailing_shape=(3, 3), min_length=2)
    _validate_time_major("v", v, trailing_shape=(3,), min_length=2)
    _validate_time_major("p", p, trailing_shape=(3,), min_length=2)
    if R.shape[0] != v.shape[0] or R.shape[0] != p.shape[0]:
        raise ValueError("R, v, and p should have the same time length")

    dt_transitions = _transition_dt(R.shape[0], R.device, R.dtype, t=t, dt=dt)
    if dt_transitions.ndim == 1 and R.ndim > 3:
        dt_transitions = dt_transitions.reshape(dt_transitions.shape[0], *([1] * (R.ndim - 3)))
    F, G = paper_invariant_error_matrices(
        R[:-1],
        v[:-1],
        p[:-1],
        dt_transitions,
        gravity=gravity,
    )
    return F, G


def adjoint_bias_gradient(
    state_gradient: torch.Tensor,
    F: torch.Tensor,
    G: torch.Tensor,
    detach: bool = True,
) -> BiasAdjointGradient:
    """Run the analytic backward recursion from target_CoRL26.pdf Section 6.

    The linearized error model is:
        xi_{k+1} = F_k xi_k + G_k n_k

    With bias order b_k = [ba_k, bg_k] and n_k = measured - true - b_k,
    dn_k/db_k = -I. Therefore:
        dL/db_k = -G_k^T lambda_{k+1}

    Args:
        state_gradient: d ell_k / d xi_k, shape (T, ..., 9).
        F: Transition matrices, shape (T-1, ..., 9, 9).
        G: Noise Jacobians, shape (T-1, ..., 9, 6).
        detach: Detach returned tensors from any existing autograd graph.
    """
    _validate_time_major("state_gradient", state_gradient, trailing_shape=(9,), min_length=2)
    _validate_time_major("F", F, trailing_shape=(9, 9))
    _validate_time_major("G", G, trailing_shape=(9, 6))
    T = state_gradient.shape[0]
    if F.shape[0] != T - 1 or G.shape[0] != T - 1:
        raise ValueError("F and G should have time length T-1")

    lambda_state = torch.empty_like(state_gradient)
    lambda_state[-1] = state_gradient[-1]
    for k in range(T - 2, -1, -1):
        lambda_state[k] = state_gradient[k] + (F[k].transpose(-1, -2) @ lambda_state[k + 1].unsqueeze(-1)).squeeze(-1)

    bias_gradient = torch.empty(*G.shape[:-2], 6, device=G.device, dtype=G.dtype)
    for k in range(T - 1):
        bias_gradient[k] = -(G[k].transpose(-1, -2) @ lambda_state[k + 1].unsqueeze(-1)).squeeze(-1)

    if detach:
        lambda_state = lambda_state.detach()
        bias_gradient = bias_gradient.detach()
        F = F.detach()
        G = G.detach()
        state_gradient = state_gradient.detach()

    return BiasAdjointGradient(
        lambda_state=lambda_state,
        bias_gradient=bias_gradient,
        F=F,
        G=G,
        state_gradient=state_gradient,
    )


def se23_mse_bias_gradient(
    X_pred: torch.Tensor,
    X_ref: torch.Tensor,
    R: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    t: torch.Tensor | None = None,
    dt: float | torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    gravity: torch.Tensor | None = None,
    reduction: str = "mean",
    skip_initial: int = 0,
    detach: bool = True,
    residual_jacobian: str = "identity",
    invariant_error: str = "right",
) -> BiasAdjointGradient:
    """Convenience helper for SE_2(3) MSE residual -> analytic bias gradient."""
    residual = se23_invariant_residual(X_pred, X_ref, invariant_error=invariant_error)
    state_grad = mse_state_gradient(
        residual,
        weight=weight,
        reduction=reduction,
        skip_initial=skip_initial,
        residual_jacobian=residual_jacobian,
    )
    F, G = invariant_transition_matrices(R, v, p, t=t, dt=dt, gravity=gravity)
    return adjoint_bias_gradient(state_grad, F, G, detach=detach)


def se23_nll_bias_gradient(
    X_pred: torch.Tensor,
    X_ref: torch.Tensor,
    R: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    covariance: torch.Tensor,
    t: torch.Tensor | None = None,
    dt: float | torch.Tensor | None = None,
    measurement_cov: torch.Tensor | None = None,
    H: torch.Tensor | None = None,
    gravity: torch.Tensor | None = None,
    jitter: float = 1e-6,
    reduction: str = "mean",
    skip_initial: int = 0,
    detach: bool = True,
    residual_jacobian: str = "identity",
    invariant_error: str = "right",
) -> BiasAdjointGradient:
    """Convenience helper for per-state SE_2(3) NLL -> analytic bias gradient."""
    residual = se23_invariant_residual(X_pred, X_ref, invariant_error=invariant_error)
    state_grad = nll_state_gradient(
        residual,
        covariance,
        measurement_cov=measurement_cov,
        H=H,
        jitter=jitter,
        reduction=reduction,
        skip_initial=skip_initial,
        residual_jacobian=residual_jacobian,
    )
    F, G = invariant_transition_matrices(R, v, p, t=t, dt=dt, gravity=gravity)
    return adjoint_bias_gradient(state_grad, F, G, detach=detach)


def _transition_dt(
    time_length: int,
    device: torch.device,
    dtype: torch.dtype,
    t: torch.Tensor | None = None,
    dt: float | torch.Tensor | None = None,
) -> torch.Tensor:
    if (t is None) == (dt is None):
        raise ValueError("Provide exactly one of t or dt")
    if t is not None:
        t = torch.as_tensor(t, device=device, dtype=dtype)
        if t.shape[0] != time_length:
            raise ValueError(f"t should have length {time_length}, got {t.shape[0]}")
        return t[1:] - t[:-1]

    dt_tensor = torch.as_tensor(dt, device=device, dtype=dtype)
    if dt_tensor.ndim == 0:
        return dt_tensor
    if dt_tensor.shape[0] != time_length - 1:
        raise ValueError(f"dt should be scalar or have length {time_length - 1}, got {tuple(dt_tensor.shape)}")
    return dt_tensor


def _validate_time_major(
    name: str,
    tensor: torch.Tensor,
    trailing_shape: tuple[int, ...],
    min_length: int = 1,
) -> None:
    if tensor.ndim < len(trailing_shape) + 1:
        raise ValueError(f"{name} should have a leading time dimension and trailing shape {trailing_shape}")
    if tuple(tensor.shape[-len(trailing_shape):]) != trailing_shape:
        raise ValueError(f"{name} should have trailing shape {trailing_shape}, got {tuple(tensor.shape)}")
    if tensor.shape[0] < min_length:
        raise ValueError(f"{name} should have at least {min_length} time steps, got {tensor.shape[0]}")


def _validate_reduction(reduction: str) -> None:
    if reduction not in {"sum", "mean", "none"}:
        raise ValueError("reduction should be one of: sum, mean, none")


def _validate_skip_initial(skip_initial: int, time_length: int) -> None:
    if skip_initial < 0:
        raise ValueError("skip_initial should be non-negative")
    if skip_initial >= time_length:
        raise ValueError("skip_initial should be smaller than the trajectory length")
