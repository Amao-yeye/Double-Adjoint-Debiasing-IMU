import torch


def accumulate_bg_euler_adjoint_gradients(
    model,
    y_solution: torch.Tensor,
    t_odeint: torch.Tensor,
    jump_grad_bg: torch.Tensor,
    jump_indices: torch.Tensor | None = None,
    parameters: list[torch.nn.Parameter] | None = None,
    vjp_mode: str = "local_ad",
) -> torch.Tensor:
    """Accumulate gyro-bias NODE parameter gradients with fixed-step adjoint.

    Forward recurrence:
        bg_{k+1} = bg_k + dt_k f_g(bg_k, w_k, w_dot_k; theta)

    Backward recurrence:
        mu_k = jump_k + (I + dt_k df_g/db_g)^T mu_{k+1}
        dL/dtheta += dt_k (df_g/dtheta)^T mu_{k+1}

    With vjp_mode="local_ad", local vector-Jacobian products are computed
    by PyTorch AD on the single-step bias MLP. This keeps the global ODE graph
    detached while preserving exact local derivatives.
    """
    if vjp_mode != "local_ad":
        raise NotImplementedError("Only vjp_mode='local_ad' is implemented for gyro inner adjoint.")
    if y_solution.ndim < 3 or y_solution.shape[-1] != 16:
        raise ValueError(f"y_solution should have shape (T, ..., 16), got {tuple(y_solution.shape)}")
    if t_odeint.ndim != 1 or t_odeint.shape[0] != y_solution.shape[0]:
        raise ValueError("t_odeint should be one-dimensional and match y_solution time length")
    if jump_grad_bg.shape[-1] != 3:
        raise ValueError(f"jump_grad_bg should have last dimension 3, got {tuple(jump_grad_bg.shape)}")

    T = y_solution.shape[0]
    device = y_solution.device
    dtype = y_solution.dtype
    if jump_indices is None:
        jump_indices = torch.arange(jump_grad_bg.shape[0], device=device)
    else:
        jump_indices = jump_indices.to(device=device)
    if jump_indices.ndim != 1 or jump_indices.shape[0] != jump_grad_bg.shape[0]:
        raise ValueError("jump_indices should be one-dimensional and match jump_grad_bg length")
    if jump_indices.numel() and (jump_indices.min() < 0 or jump_indices.max() >= T):
        raise ValueError("jump_indices contains an index outside the forward grid")

    jumps = torch.zeros(T, *y_solution.shape[1:-1], 3, device=device, dtype=dtype)
    jumps.index_add_(0, jump_indices.long(), jump_grad_bg.to(device=device, dtype=dtype))

    if parameters is None:
        parameters = [p for p in model.biasfunc_w.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable gyro-bias parameters were provided")

    lambda_bg = jumps[-1].detach()
    for k in range(T - 2, -1, -1):
        lambda_next = lambda_bg.detach()
        dt = (t_odeint[k + 1] - t_odeint[k]).to(device=device, dtype=dtype)
        y_k_const = y_solution[k].detach()
        bg_k = y_k_const[..., 4:7].clone().detach().requires_grad_(True)
        y_k = torch.cat([y_k_const[..., :4], bg_k, y_k_const[..., 7:]], dim=-1)
        t_abs = y_k_const[..., 3]
        imu_meas = model.u_func(t_abs).detach()
        imu_meas_dot = model.u_dot_func(t_abs).detach()
        f_bg = model.biasfunc_w(y_k, imu_meas, imu_meas_dot, model.R0)
        scalar = dt * (f_bg * lambda_next).sum()
        grads = torch.autograd.grad(
            scalar,
            [bg_k, *parameters],
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        grad_bg = grads[0]
        if grad_bg is None:
            grad_bg = torch.zeros_like(lambda_next)
        for parameter, grad in zip(parameters, grads[1:]):
            if grad is None:
                continue
            if parameter.grad is None:
                parameter.grad = grad.detach().clone()
            else:
                parameter.grad.add_(grad.detach())
        lambda_bg = lambda_next + grad_bg.detach()

        lambda_bg = lambda_bg + jumps[k].detach()

    return lambda_bg
