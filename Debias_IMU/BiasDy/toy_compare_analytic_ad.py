import argparse

import torch

import lie_algebra as Lie
from analytic_gradient import se23_mse_bias_gradient, se23_invariant_residual


def make_se23(R, v, p):
    X = torch.eye(5, device=R.device, dtype=R.dtype).expand(*R.shape[:-2], 5, 5).clone()
    X[..., :3, :3] = R
    X[..., :3, 3] = v
    X[..., :3, 4] = p
    return X


def bmv(A, x):
    return (A @ x.unsqueeze(-1)).squeeze(-1)


def integrate_discrete_imu(u, bias, dt, gravity):
    """Simple discrete IMU propagation for a toy gradient check.

    Args:
        u: IMU controls, shape (T-1, B, 6), order [omega, accel].
        bias: Explicit bias variables, shape (T-1, B, 6), order [ba, bg].
        dt: Scalar step size.
        gravity: Gravity vector, shape (3,).

    Returns:
        X, R, v, p with T states.
    """
    if u.shape != bias.shape:
        raise ValueError(f"u and bias should have the same shape, got {tuple(u.shape)} and {tuple(bias.shape)}")
    if u.shape[-1] != 6:
        raise ValueError("u and bias should have last dimension 6")

    T_minus_1, batch = u.shape[:2]
    R = torch.eye(3, device=u.device, dtype=u.dtype).expand(batch, 3, 3).clone()
    v = torch.zeros(batch, 3, device=u.device, dtype=u.dtype)
    p = torch.zeros(batch, 3, device=u.device, dtype=u.dtype)

    R_list = [R]
    v_list = [v]
    p_list = [p]
    for k in range(T_minus_1):
        ba = bias[k, :, :3]
        bg = bias[k, :, 3:]
        omega = u[k, :, :3] - bg
        accel = u[k, :, 3:] - ba

        R_next = R @ Lie.SO3exp(omega * dt)
        v_next = v + (bmv(R, accel) + gravity) * dt
        p_next = p + v * dt

        R, v, p = R_next, v_next, p_next
        R_list.append(R)
        v_list.append(v)
        p_list.append(p)

    R_all = torch.stack(R_list, dim=0)
    v_all = torch.stack(v_list, dim=0)
    p_all = torch.stack(p_list, dim=0)
    X_all = make_se23(R_all, v_all, p_all)
    return X_all, R_all, v_all, p_all


def compare_gradients(ad_grad, analytic_grad):
    ad_flat = ad_grad.reshape(-1)
    analytic_flat = analytic_grad.reshape(-1)
    eps = torch.tensor(1e-30, device=ad_flat.device, dtype=ad_flat.dtype)
    diff = analytic_flat - ad_flat
    neg_diff = -analytic_flat - ad_flat
    return {
        "ad_norm": ad_flat.norm().item(),
        "analytic_norm": analytic_flat.norm().item(),
        "max_abs_diff": diff.abs().max().item(),
        "rmse": diff.square().mean().sqrt().item(),
        "relative_error": (diff.norm() / (ad_flat.norm() + eps)).item(),
        "cosine": torch.nn.functional.cosine_similarity(ad_flat, analytic_flat, dim=0).item(),
        "neg_max_abs_diff": neg_diff.abs().max().item(),
        "neg_rmse": neg_diff.square().mean().sqrt().item(),
        "neg_relative_error": (neg_diff.norm() / (ad_flat.norm() + eps)).item(),
        "neg_cosine": torch.nn.functional.cosine_similarity(ad_flat, -analytic_flat, dim=0).item(),
    }


def print_group_metrics(ad_grad, analytic_grad):
    groups = {
        "ba dims [0:3]": (..., slice(0, 3)),
        "bg dims [3:6]": (..., slice(3, 6)),
    }
    print("\nGrouped metrics, comparing analytic vs autograd:")
    for name, index in groups.items():
        ad_group = ad_grad[index].reshape(-1)
        an_group = analytic_grad[index].reshape(-1)
        diff = an_group - ad_group
        rel = diff.norm() / (ad_group.norm() + torch.tensor(1e-30, device=ad_group.device, dtype=ad_group.dtype))
        cos = torch.nn.functional.cosine_similarity(ad_group, an_group, dim=0, eps=1e-30)
        print(
            f"  {name:13} "
            f"ad_norm={ad_group.norm().item():.8e} "
            f"an_norm={an_group.norm().item():.8e} "
            f"rel={rel.item():.8e} "
            f"cos={cos.item():.8e} "
            f"max_abs={diff.abs().max().item():.8e}"
        )


def finite_difference_one_entry(u, bias, X_gt, dt, gravity, index, eps, invariant_error):
    with torch.no_grad():
        bias_pos = bias.detach().clone()
        bias_neg = bias.detach().clone()
        bias_pos[index] += eps
        bias_neg[index] -= eps

    X_pos, _, _, _ = integrate_discrete_imu(u, bias_pos, dt, gravity)
    X_neg, _, _, _ = integrate_discrete_imu(u, bias_neg, dt, gravity)
    loss_pos = se23_invariant_residual(X_pos, X_gt, invariant_error=invariant_error).square().mean()
    loss_neg = se23_invariant_residual(X_neg, X_gt, invariant_error=invariant_error).square().mean()
    return ((loss_pos - loss_neg) / (2.0 * eps)).item()


def bias_from_toy_dynamics(u, theta, dt):
    """Small differentiable bias dynamics for testing chain-rule injection.

    b_{k+1} = b_k + dt * tanh(u_k @ theta)
    The returned transition bias b_k is used by the IMU propagation.
    """
    T_minus_1, batch = u.shape[:2]
    b = torch.zeros(batch, 6, device=u.device, dtype=u.dtype)
    biases = []
    for k in range(T_minus_1):
        biases.append(b)
        b_dot = torch.tanh(u[k] @ theta)
        b = b + dt * b_dot
    return torch.stack(biases, dim=0)


def compare_parameter_gradients(u, X_gt, dt, gravity, residual_jacobian, invariant_error):
    """Compare full AD dL/dtheta with analytic dL/db injected through b(theta)."""
    theta = (1.0e-3 * torch.randn(6, 6, device=u.device, dtype=u.dtype)).requires_grad_(True)
    bias = bias_from_toy_dynamics(u, theta, dt)
    X_pred, R_pred, v_pred, p_pred = integrate_discrete_imu(u, bias, dt, gravity)
    loss = se23_invariant_residual(X_pred, X_gt, invariant_error=invariant_error).square().mean()
    full_ad_theta_grad = torch.autograd.grad(loss, theta, retain_graph=True)[0].detach()

    analytic = se23_mse_bias_gradient(
        X_pred.detach(),
        X_gt.detach(),
        R_pred.detach(),
        v_pred.detach(),
        p_pred.detach(),
        dt=torch.full((u.shape[0],), dt, device=u.device, dtype=u.dtype),
        reduction="mean",
        detach=True,
        residual_jacobian=residual_jacobian,
        invariant_error=invariant_error,
    )

    injected_loss = (analytic.bias_gradient.detach() * bias).sum()
    injected_theta_grad = torch.autograd.grad(injected_loss, theta)[0].detach()
    return loss.detach(), full_ad_theta_grad, injected_theta_grad


def main():
    parser = argparse.ArgumentParser("Toy comparison between autograd and analytic dL/db.")
    parser.add_argument("--T", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bias_scale", type=float, default=1.0e-4)
    parser.add_argument("--motion_scale", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--finite_diff", action="store_true")
    parser.add_argument("--finite_diff_eps", type=float, default=1.0e-4)
    parser.add_argument("--finite_diff_samples", type=int, default=12)
    parser.add_argument("--param_grad", action="store_true", help="Also compare dL/dtheta through a toy bias dynamics.")
    parser.add_argument("--residual_jacobian", type=str, default="identity", choices=["identity", "left", "right"])
    parser.add_argument("--invariant_error", type=str, default="right", choices=["right", "left"])
    args = parser.parse_args()

    if args.T < 2:
        raise ValueError("--T should be at least 2")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float32
    gravity = torch.tensor([0.0, 0.0, -9.81], device=device, dtype=dtype)

    u = args.motion_scale * torch.randn(args.T - 1, args.batch, 6, device=device, dtype=dtype)
    zero_bias = torch.zeros_like(u)
    X_gt, _, _, _ = integrate_discrete_imu(u, zero_bias, args.dt, gravity)

    bias = (args.bias_scale * torch.randn_like(u)).requires_grad_(True)
    X_pred, R_pred, v_pred, p_pred = integrate_discrete_imu(u, bias, args.dt, gravity)
    residual = se23_invariant_residual(X_pred, X_gt, invariant_error=args.invariant_error)
    loss = residual.square().mean()
    ad_grad = torch.autograd.grad(loss, bias)[0].detach()

    analytic = se23_mse_bias_gradient(
        X_pred.detach(),
        X_gt.detach(),
        R_pred.detach(),
        v_pred.detach(),
        p_pred.detach(),
        dt=torch.full((args.T - 1,), args.dt, device=device, dtype=dtype),
        reduction="mean",
        detach=True,
        residual_jacobian=args.residual_jacobian,
        invariant_error=args.invariant_error,
    )
    analytic_grad = analytic.bias_gradient.detach()
    metrics = compare_gradients(ad_grad, analytic_grad)

    print("Toy SE23 MSE gradient check")
    print(f"T={args.T}, batch={args.batch}, dt={args.dt}, bias_scale={args.bias_scale}, motion_scale={args.motion_scale}, invariant_error={args.invariant_error}, residual_jacobian={args.residual_jacobian}")
    print(f"loss={loss.item():.8e}")
    print("\nCompare analytic vs autograd:")
    print(f"  ad_norm           {metrics['ad_norm']:.8e}")
    print(f"  analytic_norm     {metrics['analytic_norm']:.8e}")
    print(f"  max_abs_diff      {metrics['max_abs_diff']:.8e}")
    print(f"  rmse              {metrics['rmse']:.8e}")
    print(f"  relative_error    {metrics['relative_error']:.8e}")
    print(f"  cosine            {metrics['cosine']:.8e}")
    print("\nCompare -analytic vs autograd:")
    print(f"  max_abs_diff      {metrics['neg_max_abs_diff']:.8e}")
    print(f"  rmse              {metrics['neg_rmse']:.8e}")
    print(f"  relative_error    {metrics['neg_relative_error']:.8e}")
    print(f"  cosine            {metrics['neg_cosine']:.8e}")

    print("\nFirst transition, first batch, order [ba, bg]:")
    print(f"  autograd          {ad_grad[0, 0].tolist()}")
    print(f"  analytic          {analytic_grad[0, 0].tolist()}")
    print(f"  -analytic         {(-analytic_grad[0, 0]).tolist()}")
    print_group_metrics(ad_grad, analytic_grad)

    if args.finite_diff:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed + 12345)
        all_indices = [(0, 0, 0), (args.T - 2, 0, 0)]
        for _ in range(max(args.finite_diff_samples - len(all_indices), 0)):
            t_idx = torch.randint(0, args.T - 1, (1,), generator=generator).item()
            b_idx = torch.randint(0, args.batch, (1,), generator=generator).item()
            d_idx = torch.randint(0, 6, (1,), generator=generator).item()
            all_indices.append((t_idx, b_idx, d_idx))

        print("\nFinite difference checks for multiple bias entries, order [ba, bg]:")
        print("  idx(time,batch,dim)      finite_diff        autograd           analytic          -analytic         |fd-ad|")
        max_fd_ad = 0.0
        max_neg_an_ad = 0.0
        for index in all_indices:
            fd_grad = finite_difference_one_entry(
                u,
                bias,
                X_gt.detach(),
                args.dt,
                gravity,
                index,
                args.finite_diff_eps,
                args.invariant_error,
            )
            ad_value = ad_grad[index].item()
            an_value = analytic_grad[index].item()
            neg_an_value = -an_value
            fd_ad = abs(fd_grad - ad_value)
            neg_an_ad = abs(neg_an_value - ad_value)
            max_fd_ad = max(max_fd_ad, fd_ad)
            max_neg_an_ad = max(max_neg_an_ad, neg_an_ad)
            print(
                f"  {str(index):22} "
                f"{fd_grad: .8e}  {ad_value: .8e}  {an_value: .8e}  {neg_an_value: .8e}  {fd_ad:.3e}"
            )
        print(f"  max |finite_diff - autograd|: {max_fd_ad:.8e}")
        print(f"  max |-analytic - autograd|:  {max_neg_an_ad:.8e}")

    if args.param_grad:
        param_loss, full_ad_theta_grad, injected_theta_grad = compare_parameter_gradients(
            u,
            X_gt.detach(),
            args.dt,
            gravity,
            args.residual_jacobian,
            args.invariant_error,
        )
        theta_metrics = compare_gradients(full_ad_theta_grad, injected_theta_grad)
        print("\nToy bias-dynamics parameter gradient check")
        print(f"param_loss={param_loss.item():.8e}")
        print("Compare injected analytic-chain gradient vs full AD:")
        print(f"  full_ad_norm      {theta_metrics['ad_norm']:.8e}")
        print(f"  injected_norm     {theta_metrics['analytic_norm']:.8e}")
        print(f"  max_abs_diff      {theta_metrics['max_abs_diff']:.8e}")
        print(f"  rmse              {theta_metrics['rmse']:.8e}")
        print(f"  relative_error    {theta_metrics['relative_error']:.8e}")
        print(f"  cosine            {theta_metrics['cosine']:.8e}")


if __name__ == "__main__":
    main()
