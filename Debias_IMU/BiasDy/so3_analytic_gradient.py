from dataclasses import dataclass

import torch

import lie_algebra as Lie


@dataclass
class GyroBiasAdjointGradient:
    lambda_state: torch.Tensor
    bias_gradient: torch.Tensor
    state_gradient: torch.Tensor
    residual: torch.Tensor


def so3_invariant_residual(
    R_pred: torch.Tensor,
    R_ref: torch.Tensor,
    invariant_error: str = "right",
) -> torch.Tensor:
    """Return SO(3) residual in tangent coordinates.

    Args:
        R_pred: Predicted rotations, shape (T, ..., 3, 3).
        R_ref: Reference rotations, shape (T, ..., 3, 3).
        invariant_error:
            "right": residual = Log(R_ref @ R_pred^T), matching
                the right-invariant error used in target_CoRL26.pdf.
            "left": residual = Log(R_pred^T @ R_ref), matching the
                original SO3log_loss implementation.
    """
    _validate_time_major("R_pred", R_pred, trailing_shape=(3, 3))
    _validate_time_major("R_ref", R_ref, trailing_shape=(3, 3))
    if R_pred.shape != R_ref.shape:
        raise ValueError(f"R_pred and R_ref should have the same shape, got {tuple(R_pred.shape)} and {tuple(R_ref.shape)}")
    if invariant_error == "right":
        return Lie.SO3log(R_ref @ R_pred.transpose(-1, -2))
    if invariant_error == "left":
        return Lie.SO3log(R_pred.transpose(-1, -2) @ R_ref)
    raise ValueError("invariant_error should be one of: right, left")


def apply_so3_log_residual_jacobian_transpose(
    residual: torch.Tensor,
    grad_residual: torch.Tensor,
    residual_jacobian: str = "left",
) -> torch.Tensor:
    """Map dL/dr to the chosen perturbation coordinate through d Log."""
    if residual_jacobian == "identity":
        return grad_residual
    if residual_jacobian == "left":
        H = Lie.SO3leftJacoInv(residual)
    elif residual_jacobian == "right":
        H = Lie.SO3rightJacoInv(residual)
    else:
        raise ValueError("residual_jacobian should be one of: identity, left, right")
    return (H.transpose(-1, -2) @ grad_residual.unsqueeze(-1)).squeeze(-1)


def so3_mse_state_gradient(
    residual: torch.Tensor,
    reduction: str = "mean",
    skip_initial: int = 0,
    residual_jacobian: str = "left",
) -> torch.Tensor:
    """Return dL/dxi for L = MSE(Log(rotation error), 0)."""
    _validate_time_major("residual", residual, trailing_shape=(3,))
    _validate_reduction(reduction)

    grad_r = 2.0 * residual
    if reduction == "mean":
        grad_r = grad_r / residual.numel()

    grad = apply_so3_log_residual_jacobian_transpose(residual, grad_r, residual_jacobian)
    if skip_initial:
        _validate_skip_initial(skip_initial, residual.shape[0])
        grad = grad.clone()
        grad[:skip_initial] = 0.0
    return grad


def so3_gyro_bias_gradient(
    R_pred: torch.Tensor,
    R_ref: torch.Tensor,
    t: torch.Tensor | None = None,
    dt: float | torch.Tensor | None = None,
    reduction: str = "mean",
    skip_initial: int = 0,
    detach: bool = True,
    residual_jacobian: str = "left",
    invariant_error: str = "right",
) -> GyroBiasAdjointGradient:
    """Compute analytic dL/db_g,k for right-invariant SO(3) error.

    The right-invariant log-linear rotation error has F_k = I. With noise
    order n_g = b_true - b_pred, the first-order discrete input matrix is
    G_k = -dt_k R_pred,k. Because dn_g/db_g = -I:

        dL/db_g,k = -G_k^T lambda_{k+1}
                  =  dt_k R_pred,k^T lambda_{k+1}
    """
    _validate_time_major("R_pred", R_pred, trailing_shape=(3, 3), min_length=2)
    _validate_time_major("R_ref", R_ref, trailing_shape=(3, 3), min_length=2)
    residual = so3_invariant_residual(R_pred, R_ref, invariant_error=invariant_error)
    state_grad = so3_mse_state_gradient(
        residual,
        reduction=reduction,
        skip_initial=skip_initial,
        residual_jacobian=residual_jacobian,
    )
    dt_transitions = _transition_dt(R_pred.shape[0], R_pred.device, R_pred.dtype, t=t, dt=dt)
    if dt_transitions.ndim == 1 and R_pred.ndim > 3:
        dt_transitions = dt_transitions.reshape(dt_transitions.shape[0], *([1] * (R_pred.ndim - 3)))

    lambda_state = torch.empty_like(state_grad)
    lambda_state[-1] = state_grad[-1]
    for k in range(R_pred.shape[0] - 2, -1, -1):
        lambda_state[k] = state_grad[k] + lambda_state[k + 1]

    bias_gradient = torch.empty(R_pred.shape[0] - 1, *R_pred.shape[1:-2], 3, device=R_pred.device, dtype=R_pred.dtype)
    for k in range(R_pred.shape[0] - 1):
        dt_k = dt_transitions[k]
        while dt_k.ndim < lambda_state[k + 1].ndim:
            dt_k = dt_k.unsqueeze(-1)
        bias_gradient[k] = dt_k * (R_pred[k].transpose(-1, -2) @ lambda_state[k + 1].unsqueeze(-1)).squeeze(-1)

    if detach:
        lambda_state = lambda_state.detach()
        bias_gradient = bias_gradient.detach()
        state_grad = state_grad.detach()
        residual = residual.detach()

    return GyroBiasAdjointGradient(
        lambda_state=lambda_state,
        bias_gradient=bias_gradient,
        state_gradient=state_grad,
        residual=residual,
    )


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
        raise ValueError(f"{name} should have at least {min_length} time steps")


def _validate_reduction(reduction: str) -> None:
    if reduction not in {"sum", "mean", "none"}:
        raise ValueError("reduction should be one of: sum, mean, none")


def _validate_skip_initial(skip_initial: int, time_length: int) -> None:
    if skip_initial < 0 or skip_initial > time_length:
        raise ValueError(f"skip_initial should be in [0, {time_length}], got {skip_initial}")
