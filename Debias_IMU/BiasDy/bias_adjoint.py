import torch


def accumulate_ba_euler_adjoint_gradients(
    model,
    y_solution: torch.Tensor,
    t_odeint: torch.Tensor,
    jump_grad_ba: torch.Tensor,
    jump_indices: torch.Tensor | None = None,
    parameters: list[torch.nn.Parameter] | None = None,
    vjp_mode: str = "local_ad",
) -> torch.Tensor:
    """Accumulate acc-bias NODE parameter gradients with fixed-step adjoint.

    This implements the discrete adjoint of the explicit Euler recurrence used
    by the fixed-grid forward solver:

        ba_{k+1} = ba_k + dt_k f_a(ba_k, u_k, u_dot_k; theta)

    Given jump gradients dL/dba_k at selected forward states, it walks backward
    on the same time grid and accumulates dL/dtheta into parameter.grad.

    Args:
        model: VFSE23vbiasBase with u_func/u_dot_func already set.
        y_solution: Forward states on the full Euler grid, shape (T, B, 16).
            The tensor should normally be detached/no-grad.
        t_odeint: Forward time grid, shape (T,).
        jump_grad_ba: Jump gradients, shape (J, B, 3), matching ba order.
        jump_indices: Optional integer indices into the forward grid for each
            jump gradient. Defaults to arange(J).
        parameters: Optional explicit parameter list. Defaults to trainable
            parameters of model.biasfunc_a.
        vjp_mode: "local_ad" computes single-step vector-Jacobian products
            with PyTorch autograd while keeping the global ODE graph detached.

    Returns:
        lambda_ba at the initial state, useful for diagnostics.
    """
    if vjp_mode != "local_ad":
        raise NotImplementedError("Only vjp_mode='local_ad' is implemented for acc inner adjoint.")
    if y_solution.ndim < 3 or y_solution.shape[-1] != 16:
        raise ValueError(f"y_solution should have shape (T, ..., 16), got {tuple(y_solution.shape)}")
    if t_odeint.ndim != 1 or t_odeint.shape[0] != y_solution.shape[0]:
        raise ValueError("t_odeint should be one-dimensional and match y_solution time length")
    if jump_grad_ba.shape[-1] != 3:
        raise ValueError(f"jump_grad_ba should have last dimension 3, got {tuple(jump_grad_ba.shape)}")

    T = y_solution.shape[0]
    device = y_solution.device
    dtype = y_solution.dtype
    if jump_indices is None:
        jump_indices = torch.arange(jump_grad_ba.shape[0], device=device)
    else:
        jump_indices = jump_indices.to(device=device)
    if jump_indices.ndim != 1 or jump_indices.shape[0] != jump_grad_ba.shape[0]:
        raise ValueError("jump_indices should be one-dimensional and match jump_grad_ba length")
    if jump_indices.numel() and (jump_indices.min() < 0 or jump_indices.max() >= T):
        raise ValueError("jump_indices contains an index outside the forward grid")

    jumps = torch.zeros(T, *y_solution.shape[1:-1], 3, device=device, dtype=dtype)
    jumps.index_add_(0, jump_indices.long(), jump_grad_ba.to(device=device, dtype=dtype))

    if parameters is None:
        parameters = [p for p in model.biasfunc_a.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable accelerometer-bias parameters were provided")

    lambda_ba = jumps[-1].detach()
    for k in range(T - 2, -1, -1):
        lambda_next = lambda_ba.detach()
        dt = (t_odeint[k + 1] - t_odeint[k]).to(device=device, dtype=dtype)
        y_k_const = y_solution[k].detach()
        ba_k = y_k_const[..., 13:16].clone().detach().requires_grad_(True)
        y_k = torch.cat([y_k_const[..., :13], ba_k], dim=-1)
        t_abs = y_k_const[..., 3]
        imu_meas = model.u_func(t_abs).detach()
        imu_meas_dot = model.u_dot_func(t_abs).detach()
        f_ba = model.biasfunc_a(y_k, imu_meas, imu_meas_dot, model.R0)
        scalar = dt * (f_ba * lambda_next).sum()
        grads = torch.autograd.grad(
            scalar,
            [ba_k, *parameters],
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        grad_ba = grads[0]
        if grad_ba is None:
            grad_ba = torch.zeros_like(lambda_next)
        for parameter, grad in zip(parameters, grads[1:]):
            if grad is None:
                continue
            if parameter.grad is None:
                parameter.grad = grad.detach().clone()
            else:
                parameter.grad.add_(grad.detach())
        lambda_ba = lambda_next + grad_ba.detach()

        lambda_ba = lambda_ba + jumps[k].detach()

    return lambda_ba
