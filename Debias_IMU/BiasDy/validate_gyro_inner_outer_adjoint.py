import argparse
import copy
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from debug_compare_gyro_training_gradients import (
    build_model,
    collect_bg_grads,
    forward_batch,
    make_euroc_dataset,
    zero_bg_grads,
)
from gyro_adjoint import accumulate_bg_euler_adjoint_gradients
from so3_analytic_gradient import so3_gyro_bias_gradient, so3_invariant_residual


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


def run_inner_check(base_model, batch, args, device):
    model_ad = copy.deepcopy(base_model)
    model_an = copy.deepcopy(base_model)

    zero_bg_grads(model_ad)
    solution_full, solution, _, _, t_odeint, eval_indices = forward_batch(model_ad, batch, device, args.integral_method)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 17)
    jump = torch.randn(solution[..., 4:7].shape, generator=generator, device=device, dtype=solution.dtype)
    jump = jump / (jump.norm() + 1e-12)
    pseudo_loss = (solution[..., 4:7] * jump).sum()
    pseudo_loss.backward()
    ad_grads = collect_bg_grads(model_ad)

    zero_bg_grads(model_an)
    with torch.no_grad():
        solution_full_an, solution_an, _, _, t_odeint_an, eval_indices_an = forward_batch(model_an, batch, device, args.integral_method)
    jump_indices = selected_jump_indices(solution_full_an, solution_an, eval_indices_an, device)
    accumulate_bg_euler_adjoint_gradients(
        model_an,
        solution_full_an,
        t_odeint_an,
        jump.detach(),
        jump_indices=jump_indices,
        vjp_mode=args.inner_vjp,
    )
    analytic_grads = collect_bg_grads(model_an)
    return compare_vectors(flatten_grads(ad_grads), flatten_grads(analytic_grads))


def run_composed_check(base_model, batch, args, device):
    model_ad = copy.deepcopy(base_model)
    model_an = copy.deepcopy(base_model)

    zero_bg_grads(model_ad)
    _, _, R_sol, R_gt, _, _ = forward_batch(model_ad, batch, device, args.integral_method)
    loss_ad = args.loss_scale * so3_invariant_residual(
        R_sol,
        R_gt,
        invariant_error=args.invariant_error,
    ).square().mean()
    loss_ad.backward()
    ad_grads = collect_bg_grads(model_ad)

    zero_bg_grads(model_an)
    with torch.no_grad():
        solution_full_an, solution_an, R_sol_an, R_gt_an, t_odeint_an, eval_indices_an = forward_batch(model_an, batch, device, args.integral_method)
    t_gt_batch = batch[0][0]
    analytic = so3_gyro_bias_gradient(
        R_sol_an.detach(),
        R_gt_an.detach(),
        dt=(t_gt_batch[1:] - t_gt_batch[:-1]).detach(),
        reduction="mean",
        detach=True,
        residual_jacobian=args.residual_jacobian,
        invariant_error=args.invariant_error,
    )
    grad_bg = args.loss_scale * analytic.bias_gradient
    jump_indices = selected_jump_indices(solution_full_an, solution_an, eval_indices_an, device)
    if grad_bg.shape[0] != jump_indices.shape[0]:
        jump_indices = jump_indices[: grad_bg.shape[0]]
    accumulate_bg_euler_adjoint_gradients(
        model_an,
        solution_full_an,
        t_odeint_an,
        grad_bg,
        jump_indices=jump_indices,
        vjp_mode=args.inner_vjp,
    )
    analytic_grads = collect_bg_grads(model_an)
    metrics = compare_vectors(flatten_grads(ad_grads), flatten_grads(analytic_grads))
    return metrics, loss_ad.item(), grad_bg.norm().item()


def parse_indices(raw):
    return [int(part) for part in raw.split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser("Validate SO(3) gyro inner-outer adjoint gradients on real EuRoC batches.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sequence_indices", type=str, default="0")
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--loss_window", type=int, default=16)
    parser.add_argument("--integral_method", type=str, default="euler")
    parser.add_argument("--train_supervision", type=str, default="gt", choices=["gt", "vo"])
    parser.add_argument("--supervision_timing", type=str, default="imu", choices=["imu", "source"])
    parser.add_argument("--network_type_Gyro", type=str, default="bw_func_net")
    parser.add_argument("--invariant_error", type=str, default="right", choices=["right", "left"])
    parser.add_argument("--residual_jacobian", type=str, default="left", choices=["identity", "left", "right"])
    parser.add_argument("--loss_scale", type=float, default=1e6)
    parser.add_argument("--inner_vjp", type=str, default="local_ad", choices=["local_ad"])
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--inner_rel_tol", type=float, default=1e-4)
    parser.add_argument("--inner_cos_tol", type=float, default=0.99999)
    parser.add_argument("--composed_rel_tol", type=float, default=1e-2)
    parser.add_argument("--composed_cos_tol", type=float, default=0.999)
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

    print("SO(3) gyro inner-outer adjoint validation")
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
            composed_metrics, loss_value, jump_norm = run_composed_check(base_model, batch, args, device)
            inner_metrics_all.append(inner_metrics)
            composed_metrics_all.append(composed_metrics)

            tag = f"seq={sequence_index} repeat={repeat}"
            print(f"{tag} inner    {format_metrics(inner_metrics)}")
            print(f"{tag} composed loss={loss_value:.8e} jump_norm={jump_norm:.8e} {format_metrics(composed_metrics)}")

            if inner_metrics["relative_error"] > args.inner_rel_tol or inner_metrics["cosine"] < args.inner_cos_tol:
                failures.append(f"{tag} inner failed")
            if composed_metrics["relative_error"] > args.composed_rel_tol or composed_metrics["cosine"] < args.composed_cos_tol:
                failures.append(f"{tag} composed failed")

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
