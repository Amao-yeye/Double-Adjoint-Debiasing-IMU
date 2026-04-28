import argparse
import copy
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import Interpolation
from dataset import EUROCDataset
from gyro_adjoint import accumulate_bg_euler_adjoint_gradients
from learning import select_supervision_states
from network_for_BiadDy import VFSE23vbiasBase, bw_func_choose
from so3_analytic_gradient import so3_gyro_bias_gradient, so3_invariant_residual
from SO3diffeq import odeint_SO3


def grad_metrics(ad_grad, analytic_grad):
    ad = ad_grad.detach().reshape(-1)
    an = analytic_grad.detach().reshape(-1)
    diff = an - ad
    denom = ad.norm() + torch.tensor(1e-30, device=ad.device, dtype=ad.dtype)
    return {
        "ad_norm": ad.norm().item(),
        "analytic_norm": an.norm().item(),
        "relative_error": (diff.norm() / denom).item(),
        "max_abs": diff.abs().max().item(),
        "cosine": torch.nn.functional.cosine_similarity(ad, an, dim=0, eps=1e-30).item(),
    }


def make_euroc_dataset(args):
    train_seqs = [
        "MH_01_easy",
        "MH_03_medium",
        "MH_05_difficult",
        "V1_02_medium",
        "V2_01_easy",
        "V2_03_difficult",
    ]
    trainval_gt_root = "dataset/Euroc_VO" if args.train_supervision == "vo" else None
    trainval_gt_source = "stereo_estimate_aligned_body" if args.train_supervision == "vo" else "state_groundtruth_estimate0"
    cache_base = "data/EUROC_cache_vo_body" if args.train_supervision == "vo" else "data/EUROC_cache_gt"
    dataset_parameters = {
        "dataset_name": "EUROC",
        "data_dir": "data/EUROC",
        "train_seqs": train_seqs,
        "val_seqs": train_seqs,
        "test_seqs": ["MH_02_easy"],
        "dt": 0.005,
        "percent_for_val": 0.2,
        "loss_window": args.loss_window,
        "batch_size": args.batch_size,
        "sg_window_bw": 1,
        "sg_order_bw": 0,
        "sg_window_ba": 1,
        "sg_order_ba": 0,
        "trainval_gt_root": trainval_gt_root,
        "trainval_gt_source": trainval_gt_source,
        "trainval_supervision_timing": args.supervision_timing,
        "test_gt_source": "state_groundtruth_estimate0",
    }
    cache_dir = cache_base if args.supervision_timing == "imu" else f"{cache_base}_source"
    return EUROCDataset(**dataset_parameters, data_cache_dir=cache_dir, mode="train")


def build_model(args, device):
    bw_func = bw_func_choose(args.network_type_Gyro, device=device)
    model = VFSE23vbiasBase(biasfunc_w=bw_func, device=device)
    model.set_bw_zero(False)
    model.set_ba_zero(True)
    model.freeze_func_ba()
    return model


def collect_bg_grads(model):
    grads = []
    for name, param in model.biasfunc_w.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            grads.append((name, torch.zeros_like(param)))
        else:
            grads.append((name, param.grad.detach().clone()))
    return grads


def zero_bg_grads(model):
    for param in model.biasfunc_w.parameters():
        param.grad = None


def forward_batch(model, batch, device, integral_method):
    (t_gt_batch, X_gt_batch, bwa_gt_batch), t_odeint, coeff, spline_time, eval_indices = batch
    y0_batch, R0_batch = EUROCDataset.construct_init_forodeint(t_gt_batch, X_gt_batch, bwa_gt_batch)
    spline = Interpolation.CubicHermiteSpline(spline_time, coeff, device=device)
    model.set_u_func(spline.evaluate)
    model.set_u_dot_func(spline.derivative)
    model.set_R0(R0_batch)
    solution_full, R_full = odeint_SO3(model, y0_batch, R0_batch, t_odeint, method=integral_method)
    solution, R_sol = select_supervision_states(solution_full, R_full, X_gt_batch, eval_indices)
    return solution_full, solution, R_sol, X_gt_batch[..., :3, :3], t_odeint, eval_indices


def main():
    parser = argparse.ArgumentParser("Compare real gyro training-batch gradients: full AD vs SO3 analytic Euler adjoint.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sequence_index", type=int, default=0)
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
    args = parser.parse_args()

    if args.integral_method != "euler":
        raise NotImplementedError("This checker compares the fixed-step Euler adjoint; use --integral_method euler.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu.")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset = make_euroc_dataset(args)
    batch = dataset.get_data_for_biasDy(args.sequence_index, device=device, mask_flag=False, batch_size=args.batch_size)

    model_ad = build_model(args, device)
    model_an = copy.deepcopy(model_ad)

    zero_bg_grads(model_ad)
    solution_full, solution, R_sol, R_gt_batch, t_odeint, eval_indices = forward_batch(
        model_ad,
        batch,
        device,
        args.integral_method,
    )
    loss_ad = args.loss_scale * so3_invariant_residual(
        R_sol,
        R_gt_batch,
        invariant_error=args.invariant_error,
    ).square().mean()
    loss_ad.backward()
    ad_grads = collect_bg_grads(model_ad)

    zero_bg_grads(model_an)
    with torch.no_grad():
        solution_full_an, solution_an, R_sol_an, R_gt_batch_an, t_odeint_an, eval_indices_an = forward_batch(
            model_an,
            batch,
            device,
            args.integral_method,
        )
    t_gt_batch = batch[0][0]
    analytic = so3_gyro_bias_gradient(
        R_sol_an.detach(),
        R_gt_batch_an.detach(),
        dt=(t_gt_batch[1:] - t_gt_batch[:-1]).detach(),
        reduction="mean",
        detach=True,
        residual_jacobian=args.residual_jacobian,
        invariant_error=args.invariant_error,
    )
    grad_bg = args.loss_scale * analytic.bias_gradient
    if solution_full_an.shape[0] == R_gt_batch_an.shape[0]:
        jump_indices = torch.arange(grad_bg.shape[0], device=device)
    else:
        jump_indices = eval_indices_an[: grad_bg.shape[0]].to(device=device)
    accumulate_bg_euler_adjoint_gradients(
        model_an,
        solution_full_an,
        t_odeint_an,
        grad_bg,
        jump_indices=jump_indices,
        vjp_mode=args.inner_vjp,
    )
    analytic_grads = collect_bg_grads(model_an)

    print("Real gyro training-batch gradient comparison")
    print(f"device={args.device}, sequence_index={args.sequence_index}, batch_size={args.batch_size}, loss_window={args.loss_window}")
    print(f"invariant_error={args.invariant_error}, residual_jacobian={args.residual_jacobian}, inner_vjp={args.inner_vjp}, loss_scale={args.loss_scale:g}")
    print(f"loss_ad={loss_ad.item():.8e}, analytic_jump_norm={grad_bg.norm().item():.8e}")
    print("")
    for (name_ad, grad_ad), (name_an, grad_an) in zip(ad_grads, analytic_grads):
        if name_ad != name_an:
            raise RuntimeError(f"Parameter mismatch: {name_ad} vs {name_an}")
        metrics = grad_metrics(grad_ad, grad_an)
        print(
            f"{name_ad:30} "
            f"ad_norm={metrics['ad_norm']:.8e} "
            f"analytic_norm={metrics['analytic_norm']:.8e} "
            f"rel={metrics['relative_error']:.8e} "
            f"cos={metrics['cosine']:.8e} "
            f"max_abs={metrics['max_abs']:.8e}"
        )


if __name__ == "__main__":
    main()
