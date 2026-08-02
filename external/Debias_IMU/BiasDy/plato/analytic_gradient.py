from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import lie_algebra as Lie
except ImportError:  # Support `python -m BiasDy.plato...` from Debias_IMU root.
    from BiasDy import lie_algebra as Lie

try:
    from plato.covariance import se23_Gd, se23_phi
except ImportError:
    from .covariance import se23_Gd, se23_phi


@dataclass
class BiasAdjointGradient:
    lambda_state: torch.Tensor
    bias_gradient: torch.Tensor
    F: torch.Tensor
    G: torch.Tensor
    state_gradient: torch.Tensor


def weighted_se23_mse_state_gradient(
    residual: torch.Tensor,
    rot_weight: float = 1.0,
    vel_weight: float = 1.0,
    pos_weight: float = 1.0,
    blocks: tuple[str, ...] = ("rot", "vel", "pos"),
    residual_jacobian: str = "bch",
) -> torch.Tensor:
    """Analytic dL/dxi for 0.5 * mean(||W r||^2).

    The residual order is the project SE_2(3) order [phi, rho_v, rho_p].
    This matches `plato.losses.se23_mse_loss`, excluding the outer 1e6 scale.
    """
    _validate_time_major("residual", residual, trailing_shape=(9,), min_length=1)
    block_set = set(blocks)
    unknown = block_set.difference({"rot", "vel", "pos"})
    if unknown:
        raise ValueError(f"Unknown SE23 MSE blocks: {sorted(unknown)}")

    weights = torch.zeros(9, device=residual.device, dtype=residual.dtype)
    if "rot" in block_set:
        weights[0:3] = float(rot_weight)
    if "vel" in block_set:
        weights[3:6] = float(vel_weight)
    if "pos" in block_set:
        weights[6:9] = float(pos_weight)

    state_count = 1
    for size in residual.shape[:-1]:
        state_count *= int(size)
    grad_residual = weights.square() * residual / max(state_count, 1)
    return apply_log_residual_jacobian_transpose(
        residual,
        grad_residual,
        residual_jacobian=residual_jacobian,
    )


def apply_log_residual_jacobian_transpose(
    residual: torch.Tensor,
    grad_residual: torch.Tensor,
    residual_jacobian: str = "bch",
) -> torch.Tensor:
    """Map dL/dr to dL/d(delta) through the SE_2(3) residual Jacobian.

    For r = Log(X_gt X_pred^{-1}) and a left perturbation
    X_pred(delta) = Exp(delta) X_pred, BCH gives:

        r(delta) = Log(Exp(r) Exp(-delta))
        dr / ddelta = -dexp^{-1}_{-r}

    In this codebase, dexp^{-1}_{-r} is SEn3leftJaco_inv(-r), equivalently
    SEn3rightJaco_inv(r).
    """
    if residual_jacobian == "identity":
        return grad_residual
    if residual_jacobian == "bch":
        H = -Lie.SEn3rightJaco_inv(residual)
        return (H.transpose(-1, -2) @ grad_residual.unsqueeze(-1)).squeeze(-1)
    if residual_jacobian == "left":
        H = Lie.SEn3leftJaco_inv(residual)
    elif residual_jacobian == "right":
        H = Lie.SEn3rightJaco_inv(residual)
    else:
        raise ValueError("residual_jacobian must be one of: bch, identity, left, right")
    return (H.transpose(-1, -2) @ grad_residual.unsqueeze(-1)).squeeze(-1)


def invariant_transition_matrices(
    R: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    t: torch.Tensor | None = None,
    dt: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build discrete F_k, G_k for right-invariant SE_2(3) error dynamics.

    `F` is the analytic log-linear transition. `G` maps IMU noise ordered as
    [n_a, n_g] into state error ordered as [phi, rho_v, rho_p].
    """
    _validate_time_major("R", R, trailing_shape=(3, 3), min_length=2)
    _validate_time_major("v", v, trailing_shape=(3,), min_length=2)
    _validate_time_major("p", p, trailing_shape=(3,), min_length=2)
    if R.shape[0] != v.shape[0] or R.shape[0] != p.shape[0]:
        raise ValueError("R, v, and p must have the same leading time length")

    dt_transitions = _transition_dt(R.shape[0], R.device, R.dtype, t=t, dt=dt)
    if dt_transitions.ndim == 1 and R.ndim > 3:
        dt_transitions = dt_transitions.reshape(dt_transitions.shape[0], *([1] * (R.ndim - 3)))

    F = se23_phi(dt_transitions, device=R.device, dtype=R.dtype)
    G = se23_Gd(R[:-1], v[:-1], p[:-1], dt_transitions)
    return F, G


def adjoint_bias_gradient(
    state_gradient: torch.Tensor,
    F: torch.Tensor,
    G: torch.Tensor,
    detach: bool = True,
) -> BiasAdjointGradient:
    """Run the analytic outer-adjoint recursion for bias sensitivity.

    Error model:
        xi_{k+1} = F_k xi_k + G_k n_k

    `se23_Gd` already carries the sign of the IMU-error injection used by the
    current right-invariant dynamics. Its columns are ordered [n_a, n_g],
    which are used here as the bias-gradient order [ba_k, bg_k]:
        dL/db_k = G_k^T lambda_{k+1}
    """
    _validate_time_major("state_gradient", state_gradient, trailing_shape=(9,), min_length=2)
    _validate_time_major("F", F, trailing_shape=(9, 9))
    _validate_time_major("G", G, trailing_shape=(9, 6))
    T = state_gradient.shape[0]
    if F.shape[0] != T - 1 or G.shape[0] != T - 1:
        raise ValueError("F and G must have time length T-1")

    lambda_state = torch.empty_like(state_gradient)
    lambda_state[-1] = state_gradient[-1]
    for k in range(T - 2, -1, -1):
        lambda_state[k] = state_gradient[k] + (
            F[k].transpose(-1, -2) @ lambda_state[k + 1].unsqueeze(-1)
        ).squeeze(-1)

    bias_gradient = torch.empty(*G.shape[:-2], 6, device=G.device, dtype=G.dtype)
    for k in range(T - 1):
        bias_gradient[k] = (
            G[k].transpose(-1, -2) @ lambda_state[k + 1].unsqueeze(-1)
        ).squeeze(-1)

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
            raise ValueError(f"t must have length {time_length}, got {t.shape[0]}")
        return t[1:] - t[:-1]

    dt_tensor = torch.as_tensor(dt, device=device, dtype=dtype)
    if dt_tensor.ndim == 0:
        return dt_tensor
    if dt_tensor.shape[0] != time_length - 1:
        raise ValueError(f"dt must be scalar or have length {time_length - 1}, got {tuple(dt_tensor.shape)}")
    return dt_tensor


def _validate_time_major(
    name: str,
    tensor: torch.Tensor,
    trailing_shape: tuple[int, ...],
    min_length: int = 1,
) -> None:
    if tensor.ndim < len(trailing_shape) + 1:
        raise ValueError(f"{name} must have a leading time dimension and trailing shape {trailing_shape}")
    if tuple(tensor.shape[-len(trailing_shape):]) != trailing_shape:
        raise ValueError(f"{name} must have trailing shape {trailing_shape}, got {tuple(tensor.shape)}")
    if tensor.shape[0] < min_length:
        raise ValueError(f"{name} must have at least {min_length} time steps")
