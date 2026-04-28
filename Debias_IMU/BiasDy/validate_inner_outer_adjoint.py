import argparse
import copy
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analytic_gradient import se23_invariant_residual, se23_mse_bias_gradient
from bias_adjoint import accumulate_ba_euler_adjoint_gradients
from debug_compare_training_gradients import (
    build_model,
    collect_ba_grads,
    forward_batch,
    make_euroc_dataset,
    zero_ba_grads,
)


def flatten_grads(named_grads):
    return torch.cat([grad.detach().reshape(-1) for _, grad in named_grads])


def compare_vectors(reference, candidate):
    reference = reference.detach().reshape(-1)
    candidate = candidate.detach().reshape(-1)
    diff = candidate - reference
    denom = reference.norm() + torch.tensor(1e-30, device=reference.device, dtype=reference.dtype)
    return {
        "reference_norm": reference.norm().item(),
        "candidate_norm": candidate.norm().item(),
        "relative_error": (diff.norm() / denom).item(),
        "max_abs": diff.abs().max().item(),
        "cosine": torch.nn.functional.cosine_similarity(reference, candidate, dim=0, eps=1e-30).item(),
    }


def format_metrics(metrics):
    return (
        f"ref_norm={metrics['reference_norm']:.8e} "
        f"cand_norm={metrics['candidate_norm']:.8e} "
        f"rel={metrics['relative_error']:.8e} "
        f"cos={metrics['cosine']:.8e} "
        f"max_abs={metrics['max_abs']:.8e}"
    )


def selected_jump_indices(solution_full, selected_solution, eval_indices, device):
    if solution_full.shape[0] == selected_solution.shape[0]:
        return torch.arange(selected_solution.shape[0], device=device)
    return eval_indices[: selected_solution.shape[0]].to(device=device)


def loss_from_forward(model, batch, args, device):
    _, _, _, x_pred, x_gt_batch, _, _ = forward_batch(model, batch, device, args.integral_method)
    return args.loss_scale * se23_invariant_residual(
        x_pred,
        x_gt_batch,
        invariant_error=args.invariant_error,
    ).square().mean()


def run_inner_check(base_model, batch, args, device):
    """Check the inner discrete adjoint against AD for arbitrary ba jumps."""
    model_ad = copy.deepcopy(base_model)
    model_an = copy.deepcopy(base_model)

    zero_ba_grads(model_ad)
    solution_full, solution, _, _, _, t_odeint, eval_indices = forward_batch(
        model_ad,
        batch,
        device,
        args.integral_method,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 17)
    jump = torch.randn(solution[..., 13:16].shape, generator=generator, device=device, dtype=solution.dtype)
    jump = jump / (jump.norm() + 1e-12)
    pseudo_loss = (solution[..., 13:16] * jump).sum()
    pseudo_loss.backward()
    ad_grads = collect_ba_grads(model_ad)

    zero_ba_grads(model_an)
    with torch.no_grad():
        solution_full_an, solution_an, _, _, _, t_odeint_an, eval_indices_an = forward_batch(
            model_an,
            batch,
            device,
            args.integral_method,
        )
    jump_indices = selected_jump_indices(solution_full_an, solution_an, eval_indices_an, device)
    accumulate_ba_euler_adjoint_gradients(
        model_an,
        solution_full_an,
        t_odeint_an,
        jump.detach(),
        jump_indices=jump_indices,
        vjp_mode=args.inner_vjp,
    )
    analytic_grads = collect_ba_grads(model_an)
    return compare_vectors(flatten_grads(ad_grads), flatten_grads(analytic_grads))


def run_composed_check(base_model, batch, args, device):
    """Check full dL/dtheta AD against analytic outer jump plus inner adjoint."""
    model_ad = copy.deepcopy(base_model)
    model_an = copy.deepcopy(base_model)

    zero_ba_grads(model_ad)
    solution_full, solution, r_sol, x_pred, x_gt_batch, t_odeint, eval_indices = forward_batch(
        model_ad,
        batch,
        device,
        args.integral_method,
    )
    loss_ad = args.loss_scale * se23_invariant_residual(
        x_pred,
        x_gt_batch,
        invariant_error=args.invariant_error,
    ).square().mean()
    loss_ad.backward()
    ad_grads = collect_ba_grads(model_ad)

    zero_ba_grads(model_an)
    with torch.no_grad():
        solution_full_an, solution_an, r_sol_an, x_pred_an, x_gt_batch_an, t_odeint_an, eval_indices_an = forward_batch(
            model_an,
            batch,
            device,
            args.integral_method,
        )
    t_gt_batch = batch[0][0]
    analytic = se23_mse_bias_gradient(
        x_pred_an.detach(),
        x_gt_batch_an.detach(),
        r_sol_an.detach(),
        solution_an[..., 7:10].detach(),
        solution_an[..., 10:13].detach(),
        dt=(t_gt_batch[1:] - t_gt_batch[:-1]).detach(),
        reduction="mean",
        detach=True,
        residual_jacobian=args.residual_jacobian,
        invariant_error=args.invariant_error,
    )
    grad_ba = args.loss_scale * analytic.bias_gradient[..., :3]
    jump_indices = selected_jump_indices(solution_full_an, solution_an, eval_indices_an, device)
    if grad_ba.shape[0] != jump_indices.shape[0]:
        jump_indices = jump_indices[: grad_ba.shape[0]]
    accumulate_ba_euler_adjoint_gradients(
        model_an,
        solution_full_an,
        t_odeint_an,
        grad_ba,
        jump_indices=jump_indices,
        vjp_mode=args.inner_vjp,
    )
    analytic_grads = collect_ba_grads(model_an)
    metrics = compare_vectors(flatten_grads(ad_grads), flatten_grads(analytic_grads))
    return metrics, loss_ad.item(), grad_ba.norm().item(), analytic_grads


def run_directional_finite_difference(base_model, batch, analytic_grads, args, device):
    """Independent check: central finite difference directional derivative."""
    model_fd = copy.deepcopy(base_model)
    parameters = [p for p in model_fd.biasfunc_a.parameters() if p.requires_grad]
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 31)
    direction = [torch.randn(p.shape, generator=generator, device=device, dtype=p.dtype) for p in parameters]
    direction_norm = torch.sqrt(sum((d * d).sum() for d in direction))
    direction = [d / (direction_norm + 1e-30) for d in direction]

    analytic_dot = sum((grad * d).sum() for (_, grad), d in zip(analytic_grads, direction)).item()

    eps = args.finite_diff_eps
    with torch.no_grad():
        for parameter, d in zip(parameters, direction):
            parameter.add_(eps * d)
        loss_plus = loss_from_forward(model_fd, batch, args, device).item()
        for parameter, d in zip(parameters, direction):
            parameter.add_(-2.0 * eps * d)
        loss_minus = loss_from_forward(model_fd, batch, args, device).item()
        for parameter, d in zip(parameters, direction):
            parameter.add_(eps * d)

    finite_diff = (loss_plus - loss_minus) / (2.0 * eps)
    abs_diff = abs(finite_diff - analytic_dot)
    reliable = abs(finite_diff) >= args.finite_diff_min_abs
    rel = abs_diff / (abs(finite_diff) + 1e-30) if reliable else float("nan")
    return {
        "finite_diff": finite_diff,
        "analytic_dot": analytic_dot,
        "absolute_error": abs_diff,
        "relative_error": rel,
        "reliable": reliable,
        "loss_plus": loss_plus,
        "loss_minus": loss_minus,
    }


def parse_indices(raw):
    return [int(part) for part in raw.split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser("Validate inner-outer adjoint gradients on real EuRoC training batches.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sequence_indices", type=str, default="0")
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--loss_window", type=int, default=16)
    parser.add_argument("--integral_method", type=str, default="euler")
    parser.add_argument("--train_supervision", type=str, default="gt", choices=["gt", "vo"])
    parser.add_argument("--supervision_timing", type=str, default="imu", choices=["imu", "source"])
    parser.add_argument("--network_type_Gyro", type=str, default="bw_func_net")
    parser.add_argument("--network_type_Acc", type=str, default="ba_func_net")
    parser.add_argument("--bw_model_path", type=str, default=None)
    parser.add_argument("--invariant_error", type=str, default="right", choices=["right", "left"])
    parser.add_argument("--residual_jacobian", type=str, default="identity", choices=["identity", "left", "right"])
    parser.add_argument("--loss_scale", type=float, default=1e6)
    parser.add_argument("--inner_vjp", type=str, default="local_ad", choices=["local_ad"])
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--inner_rel_tol", type=float, default=1e-4)
    parser.add_argument("--inner_cos_tol", type=float, default=0.99999)
    parser.add_argument("--composed_rel_tol", type=float, default=1e-2)
    parser.add_argument("--composed_cos_tol", type=float, default=0.999)
    parser.add_argument("--finite_diff", action="store_true")
    parser.add_argument("--finite_diff_eps", type=float, default=1e-1)
    parser.add_argument("--finite_diff_min_abs", type=float, default=1e-8)
    args = parser.parse_args()

    if args.integral_method != "euler":
        raise NotImplementedError("This validator is for the fixed-step Euler inner adjoint.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu.")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    dataset = make_euroc_dataset(args)
    base_model = build_model(args, device)

    failures = []
    inner_metrics_all = []
    composed_metrics_all = []

    print("Inner-outer adjoint validation")
    print(
        f"device={args.device}, sequences={args.sequence_indices}, repeats={args.num_repeats}, "
        f"batch_size={args.batch_size}, loss_window={args.loss_window}"
    )
    print(f"invariant_error={args.invariant_error}, residual_jacobian={args.residual_jacobian}, inner_vjp={args.inner_vjp}, loss_scale={args.loss_scale:g}")
    print("")

    for sequence_index in parse_indices(args.sequence_indices):
        for repeat in range(args.num_repeats):
            torch.manual_seed(args.seed + 1000 * sequence_index + repeat)
            batch = dataset.get_data_for_biasDy(
                sequence_index,
                device=device,
                mask_flag=False,
                batch_size=args.batch_size,
            )

            inner_metrics = run_inner_check(base_model, batch, args, device)
            composed_metrics, loss_value, jump_norm, analytic_grads = run_composed_check(base_model, batch, args, device)
            inner_metrics_all.append(inner_metrics)
            composed_metrics_all.append(composed_metrics)

            tag = f"seq={sequence_index} repeat={repeat}"
            print(f"{tag} inner    {format_metrics(inner_metrics)}")
            print(f"{tag} composed loss={loss_value:.8e} jump_norm={jump_norm:.8e} {format_metrics(composed_metrics)}")

            if inner_metrics["relative_error"] > args.inner_rel_tol or inner_metrics["cosine"] < args.inner_cos_tol:
                failures.append(f"{tag} inner failed")
            if composed_metrics["relative_error"] > args.composed_rel_tol or composed_metrics["cosine"] < args.composed_cos_tol:
                failures.append(f"{tag} composed failed")

            if args.finite_diff:
                fd_metrics = run_directional_finite_difference(base_model, batch, analytic_grads, args, device)
                fd_status = "reliable" if fd_metrics["reliable"] else "unreliable"
                print(
                    f"{tag} finite_diff fd={fd_metrics['finite_diff']:.8e} "
                    f"analytic_dot={fd_metrics['analytic_dot']:.8e} "
                    f"rel={fd_metrics['relative_error']:.8e} "
                    f"abs={fd_metrics['absolute_error']:.8e} "
                    f"{fd_status}"
                )

    max_inner_rel = max(m["relative_error"] for m in inner_metrics_all)
    min_inner_cos = min(m["cosine"] for m in inner_metrics_all)
    max_composed_rel = max(m["relative_error"] for m in composed_metrics_all)
    min_composed_cos = min(m["cosine"] for m in composed_metrics_all)
    print("")
    print(
        "summary "
        f"max_inner_rel={max_inner_rel:.8e} min_inner_cos={min_inner_cos:.8e} "
        f"max_composed_rel={max_composed_rel:.8e} min_composed_cos={min_composed_cos:.8e}"
    )

    if failures:
        print("FAILED")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)
    print("PASSED")


if __name__ == "__main__":
    main()
