#!/usr/bin/env python3
"""Run EuRoC Debias_IMU training with PLATO-owned bias dynamics networks.

This is PLATO glue code.  It keeps external/Debias_IMU untouched while reusing
its EuRoC dataset loader, losses, and evaluation routines.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
from pathlib import Path

import torch

from run_liasuw_debias_ad_pipeline import (
    ACC_CONSERVATIVE_RESIDUAL_NAME,
    ACC_M2_NAME,
    ACC_PHYS_DELTA_NAME,
    ACC_PHYS_EQ_NAME,
    ACC_PHYS_RATE_NAME,
    DEBIAS_ROOT,
    GYRO_CONSERVATIVE_RESIDUAL_NAME,
    GYRO_M2_NAME,
    GYRO_PHYS_DELTA_NAME,
    GYRO_PHYS_EQ_NAME,
    GYRO_PHYS_RATE_NAME,
    PLATO_ROOT,
    Acc_train_SE23_with_readable_metrics,
    Acc_train_interval_preintegration_NLL_with_readable_metrics,
    Gyro_train_with_readable_metrics,
    Test_with_configured_aux,
    _load_ba_model_initialization,
    _requires_spline_derivative,
    _resolve_initial_bias,
    _set_model_imu_functions,
    _uses_delta_imu,
    ba_func_choose,
    bw_func_choose,
    resolve_checkpoint,
)
import Interpolation
import learning as debias_learning
from dataset import EUROCDataset, pload
from evalutation import AOE, ATE
from learning import Acc_train, Acc_train_SE23, Gyro_train, format_training_profile, plot_bias, simple_visualization
from learning import Acc_train_SE23_Analytic, Gyro_train_Analytic


DEFAULT_TRAIN_SEQS = [
    "MH_01_easy",
    "MH_03_medium",
    "MH_05_difficult",
    "V1_02_medium",
    "V2_01_easy",
    "V2_03_difficult",
]
DEFAULT_TEST_SEQS = [
    "MH_02_easy",
    "MH_04_difficult",
    "V2_02_medium",
    "V1_03_difficult",
    "V1_01_easy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("EuRoC Debias_IMU pipeline with PLATO bias networks")
    parser.add_argument("--data-root", type=Path, default=DEBIAS_ROOT / "data" / "EUROC")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--test-cache-dir",
        type=Path,
        default=PLATO_ROOT / "results" / "debias_imu" / "cache" / "EUROC_test_gt",
    )
    parser.add_argument(
        "--outputdir",
        type=Path,
        default=PLATO_ROOT / "results" / "debias_imu" / "EUROC_phys_delta_e1801_w16_b1000",
    )
    parser.add_argument("--train-supervision", choices=["gt", "vo"], default="gt")
    parser.add_argument(
        "--trainval-gt-root",
        type=Path,
        default=None,
        help="External EuRoC VO teacher root. Defaults to external Debias_IMU dataset/Euroc_VO when --train-supervision=vo.",
    )
    parser.add_argument("--trainval-gt-source", type=str, default=None)
    parser.add_argument(
        "--supervision-timing",
        choices=["imu", "source"],
        default="imu",
        help="Use dense IMU-timing supervision or source-timing VO/GT poses for train/val.",
    )
    parser.add_argument("--exclude-train-seq", nargs="*", default=[])
    parser.add_argument("--train-seqs", nargs="+", default=DEFAULT_TRAIN_SEQS)
    parser.add_argument(
        "--val-seqs",
        nargs="+",
        default=None,
        help="Optional sequence-level validation split. Defaults to train_seqs for original Debias_IMU behavior.",
    )
    parser.add_argument("--test-seqs", nargs="+", default=DEFAULT_TEST_SEQS)
    parser.add_argument("--integral_method", type=str, default="euler")
    parser.add_argument(
        "--network_type_Gyro",
        type=str,
        default=GYRO_PHYS_DELTA_NAME,
        help=(
            f"Gyro network. PLATO options: {GYRO_M2_NAME}, {GYRO_PHYS_EQ_NAME}, "
            f"{GYRO_PHYS_RATE_NAME}, {GYRO_PHYS_DELTA_NAME}, {GYRO_CONSERVATIVE_RESIDUAL_NAME}. "
            "External default: bw_func_net."
        ),
    )
    parser.add_argument(
        "--network_type_Acc",
        type=str,
        default=ACC_PHYS_DELTA_NAME,
        help=(
            f"Accel network. PLATO options: {ACC_M2_NAME}, {ACC_PHYS_EQ_NAME}, "
            f"{ACC_PHYS_RATE_NAME}, {ACC_PHYS_DELTA_NAME}, {ACC_CONSERVATIVE_RESIDUAL_NAME}. "
            "External default: ba_func_net."
        ),
    )
    parser.add_argument("--loss_window", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--epoch", type=int, default=1801)
    parser.add_argument("--lr_Gyro", type=float, default=5e-3)
    parser.add_argument("--weight_decay_Gyro", type=float, default=1e-6)
    parser.add_argument("--gyro_lr_step_size", type=int, default=300)
    parser.add_argument("--gyro_lr_gamma", type=float, default=0.5)
    parser.add_argument("--lr_Acc", type=float, default=5e-3)
    parser.add_argument("--weight_decay_Acc", type=float, default=1e-6)
    parser.add_argument("--acc_lr_step_size", type=int, default=300)
    parser.add_argument("--acc_lr_gamma", type=float, default=0.5)
    parser.add_argument("--gyro-loss", choices=["ad", "analytic"], default="ad")
    parser.add_argument("--acc-loss", choices=["se23", "analytic", "rss"], default="se23")
    parser.add_argument(
        "--acc-loss-type",
        choices=["se23", "rss", "interval_nll", "trajectory_nll"],
        default="se23",
        help=(
            "Accelerometer-stage loss. rss uses original Debias_IMU Acc_train; "
            "interval_nll reuses the PLATO interval preintegration-factor NLL trainer; "
            "trajectory_nll uses the stacked trajectory-state NLL with cross-time covariance."
        ),
    )
    parser.add_argument("--ba-init-model-path", type=Path, default=None)
    parser.add_argument("--nll-train-gyro", action="store_true")
    parser.add_argument("--acc-loss-rot-weight", type=float, default=1.0)
    parser.add_argument("--acc-loss-vel-weight", type=float, default=1.0)
    parser.add_argument("--acc-loss-pos-weight", type=float, default=1.0)
    parser.add_argument("--gyro-bias-rate-regularization-weight", type=float, default=0.0)
    parser.add_argument("--acc-bias-rate-regularization-weight", type=float, default=0.0)
    parser.add_argument("--bw-init-model-path", type=Path, default=None)
    parser.add_argument("--interval-nll-mode", choices=["mse_only", "nll_cov_only", "nll_joint"], default="nll_cov_only")
    parser.add_argument("--interval-sigma-g-nominal", type=float, default=1.6968e-4)
    parser.add_argument("--interval-sigma-a-nominal", type=float, default=2.0e-3)
    parser.add_argument("--interval-sigma-g-floor2", type=float, default=1.0e-12)
    parser.add_argument("--interval-sigma-a-floor2", type=float, default=1.0e-10)
    parser.add_argument("--interval-target-cov-floor", type=float, default=1.0e-9)
    parser.add_argument("--interval-jitter", type=float, default=1.0e-8)
    parser.add_argument("--interval-prior-lambda-g", type=float, default=0.0)
    parser.add_argument("--interval-prior-lambda-a", type=float, default=0.0)
    parser.add_argument("--interval-lr-sigma", type=float, default=1.0e-2)
    parser.add_argument(
        "--interval-detach-covariance-theta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detach covariance propagation from network theta while retaining residual gradients.",
    )
    parser.add_argument(
        "--interval-covariance-gradient-mode",
        choices=["analytic_sensitivity", "ad"],
        default="analytic_sensitivity",
    )
    parser.add_argument("--interval-debug-covariance-gradients", action="store_true")
    parser.add_argument("--interval-noise-init-model-path", type=Path, default=None)
    parser.add_argument("--trajectory-supervision-mode", choices=["sparse", "endpoint"], default="sparse")
    parser.add_argument("--trajectory-supervision-stride", type=int, default=1)
    parser.add_argument("--trajectory-max-supervised-states", type=int, default=0)
    parser.add_argument("--analytic_grad_sign", type=float, default=1.0, choices=[-1.0, 1.0])
    parser.add_argument("--analytic_grad_scale", type=float, default=1.0)
    parser.add_argument("--analytic_grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--analytic_residual_jacobian", choices=["identity", "left", "right"], default="left")
    parser.add_argument("--analytic_invariant_error", choices=["right", "left"], default="right")
    parser.add_argument("--gyro_analytic_grad_sign", type=float, default=1.0, choices=[-1.0, 1.0])
    parser.add_argument("--gyro_analytic_grad_scale", type=float, default=1.0)
    parser.add_argument("--gyro_analytic_grad_clip_norm", type=float, default=0.0)
    parser.add_argument("--gyro_analytic_residual_jacobian", choices=["identity", "left", "right"], default="left")
    parser.add_argument("--gyro_analytic_invariant_error", choices=["right", "left"], default="right")
    parser.add_argument("--inner_vjp", choices=["local_ad"], default="local_ad")
    parser.add_argument("--val_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=100)
    parser.add_argument("--train_batch_strategy", choices=["random_sequence", "balanced_sequences"], default="random_sequence")
    parser.add_argument("--percent_for_val", type=float, default=0.2)
    parser.add_argument("--sg_window_bw", type=int, default=1)
    parser.add_argument("--sg_order_bw", type=int, default=0)
    parser.add_argument("--sg_window_ba", type=int, default=1)
    parser.add_argument("--sg_order_ba", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test-device", type=str, default="cpu")
    parser.add_argument(
        "--initial-bias-mode",
        choices=["first", "mean_window", "median_window", "train_global_mean", "train_global_median"],
        default="first",
    )
    parser.add_argument("--initial-bias-window-s", type=float, default=1.0)
    parser.add_argument("--recompute-cache", action="store_true")
    parser.add_argument("--test-recompute", action="store_true")
    parser.add_argument("--only-test", action="store_true")
    parser.add_argument("--skip-gyro-train", action="store_true")
    parser.add_argument("--stop-after-gyro", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--bw_model_path", type=Path, default=None)
    parser.add_argument("--ba_model_path", type=Path, default=None)
    parser.add_argument("--use-best-val", action="store_true")
    parser.add_argument(
        "--full-rollout-select",
        action="store_true",
        help="Select saved MSE checkpoints by held-out full-rollout AOE/position instead of short-window validation.",
    )
    parser.add_argument(
        "--full-rollout-select-only",
        action="store_true",
        help="Run full-rollout checkpoint selection on existing checkpoints, then test the selected checkpoint.",
    )
    parser.add_argument(
        "--full-rollout-select-gyro",
        action="store_true",
        help="Also select the gyro checkpoint by held-out full-rollout orientation before Acc training.",
    )
    parser.add_argument("--full-rollout-val-seqs", nargs="+", default=None)
    parser.add_argument("--full-rollout-selection-cache-dir", type=Path, default=None)
    parser.add_argument("--full-rollout-selection-device", type=str, default=None)
    parser.add_argument("--full-rollout-score-aoe-weight", type=float, default=1.0)
    parser.add_argument("--full-rollout-score-vel-weight", type=float, default=0.0)
    parser.add_argument("--full-rollout-score-pos-weight", type=float, default=1.0)
    parser.add_argument("--full-rollout-score-aoe-scale", type=float, default=3.4944)
    parser.add_argument("--full-rollout-score-vel-scale", type=float, default=16.2950)
    parser.add_argument("--full-rollout-score-pos-scale", type=float, default=678.0098)
    parser.add_argument(
        "--readable-val-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print validation RMSE metrics alongside the original scaled Debias_IMU losses.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    if args.cache_dir is None:
        source = "vo_body" if args.train_supervision == "vo" else "gt"
        timing = f"_{args.supervision_timing}" if args.supervision_timing != "imu" else ""
        args.cache_dir = PLATO_ROOT / "results" / "debias_imu" / "cache" / f"EUROC_{source}{timing}_w{args.loss_window}"
    return args


def make_dataset_parameters(args: argparse.Namespace) -> dict:
    train_seqs = [seq for seq in args.train_seqs if seq not in set(args.exclude_train_seq)]
    val_seqs = list(args.val_seqs) if args.val_seqs is not None else list(train_seqs)
    trainval_gt_root = None
    trainval_gt_source = "state_groundtruth_estimate0"
    if args.train_supervision == "vo":
        trainval_gt_root = args.trainval_gt_root or (DEBIAS_ROOT / "dataset" / "Euroc_VO")
        trainval_gt_source = args.trainval_gt_source or "stereo_estimate_aligned_body"
    elif args.trainval_gt_root is not None:
        trainval_gt_root = args.trainval_gt_root
    elif args.trainval_gt_source is not None:
        trainval_gt_source = args.trainval_gt_source
    return {
        "dataset_name": "EUROC",
        "data_dir": str(args.data_root),
        "train_seqs": train_seqs,
        "val_seqs": val_seqs,
        "test_seqs": list(args.test_seqs),
        "percent_for_val": args.percent_for_val,
        "batch_size": args.batch_size,
        "loss_window": args.loss_window,
        "dt": args.dt,
        "sg_window_bw": args.sg_window_bw,
        "sg_order_bw": args.sg_order_bw,
        "sg_window_ba": args.sg_window_ba,
        "sg_order_ba": args.sg_order_ba,
        "trainval_gt_root": str(trainval_gt_root) if trainval_gt_root is not None else None,
        "test_gt_root": None,
        "trainval_gt_source": trainval_gt_source,
        "test_gt_source": "state_groundtruth_estimate0",
        "trainval_supervision_timing": args.supervision_timing,
    }


def _train_global_bias_from_cache(
    cache_dir: Path,
    train_seqs: list[str],
    stat: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    bw_values = []
    ba_values = []
    missing = []
    for seq in train_seqs:
        path = cache_dir / f"{seq}.p"
        if not path.exists():
            missing.append(str(path))
            continue
        data = pload(str(path))
        bw_values.append(data["w_bias"].float())
        ba_values.append(data["a_bias"].float())
    if missing:
        raise FileNotFoundError(
            "train-global initial bias requires cached train sequence bias files. Missing: "
            + ", ".join(missing)
        )
    if not bw_values:
        raise ValueError("train-global initial bias requires at least one train sequence")
    bw = torch.cat(bw_values, dim=0)
    ba = torch.cat(ba_values, dim=0)
    if stat == "mean":
        bw0 = bw.mean(dim=0)
        ba0 = ba.mean(dim=0)
    elif stat == "median":
        bw0 = bw.median(dim=0).values
        ba0 = ba.median(dim=0).values
    else:
        raise ValueError(f"unsupported train-global stat: {stat}")
    return bw0.to(device), ba0.to(device)


def _checkpoint_sort_key(path: Path) -> tuple[int, int, str]:
    if path.name == "model_initial.pt":
        return (0, -1, path.name)
    if path.name == "best_val_model.pt":
        return (1, 10**9, path.name)
    match = re.match(r"model_epoch_(\d+)\.pt$", path.name)
    if match:
        return (0, int(match.group(1)), path.name)
    return (2, 10**9, path.name)


def _checkpoint_label(path: Path) -> str:
    if path.name == "model_initial.pt":
        return "initial"
    if path.name == "best_val_model.pt":
        return "best_val"
    match = re.match(r"model_epoch_(\d+)\.pt$", path.name)
    if match:
        return f"epoch_{match.group(1)}"
    return path.stem


def _checkpoint_candidates(stage_dir: Path) -> list[Path]:
    candidates = []
    initial = stage_dir / "model_initial.pt"
    if initial.exists():
        candidates.append(initial)
    candidates.extend(sorted(stage_dir.glob("model_epoch_*.pt"), key=_checkpoint_sort_key))
    best = stage_dir / "best_val_model.pt"
    if best.exists():
        candidates.append(best)
    unique = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _make_full_rollout_dataset(
    args: argparse.Namespace,
    params: dict,
    eval_seqs: list[str],
) -> EUROCDataset:
    cache_dir = args.full_rollout_selection_cache_dir or args.test_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = any(not (cache_dir / f"{seq}.p").exists() or not (cache_dir / f"{seq}_gt.p").exists() for seq in eval_seqs)
    eval_params = dict(params)
    eval_params["train_seqs"] = []
    eval_params["val_seqs"] = []
    eval_params["test_seqs"] = list(eval_seqs)
    return EUROCDataset(
        **eval_params,
        data_cache_dir=str(cache_dir),
        mode="test",
        recompute=missing,
    )


def _full_rollout_score(summary: dict[str, float], args: argparse.Namespace) -> float:
    return (
        args.full_rollout_score_aoe_weight * summary["mean_aoe_deg"] / args.full_rollout_score_aoe_scale
        + args.full_rollout_score_vel_weight * summary["mean_ate_vel"] / args.full_rollout_score_vel_scale
        + args.full_rollout_score_pos_weight * summary["mean_ate_pos_m"] / args.full_rollout_score_pos_scale
    )


def _evaluate_full_rollout_metrics(
    dataset_eval: EUROCDataset,
    bw_func,
    ba_func,
    integral_method: str,
    device: str,
    use_delta_imu: bool,
    initial_bias_mode: str,
    initial_bias_window_s: float,
    fixed_bw0: torch.Tensor | None,
    fixed_ba0: torch.Tensor | None,
) -> tuple[list[dict[str, float | str]], dict[str, float]]:
    model = debias_learning.VFSE23vbiasBase(biasfunc_w=bw_func, biasfunc_a=ba_func, device=device)
    model.set_bw_zero(False)
    model.set_ba_zero(False)
    model.eval()

    rows = []
    for index in range(dataset_eval.length()):
        seq_name = dataset_eval.sequences[index]
        coeff, time_tensor = dataset_eval.get_coeff(index)
        spline = Interpolation.CubicHermiteSpline(time_tensor, coeff, device)
        _t_imu, _u_imu, X_gt, t_gt = dataset_eval.get_full_trajectory(index, device=device)
        R0 = X_gt[0, :3, :3].clone()
        bias_w_gt, t_w_bias = dataset_eval.get_w_bias(index)
        bias_a_gt, t_a_bias = dataset_eval.get_a_bias(index)
        bw0 = _resolve_initial_bias(
            bias_w_gt,
            t_w_bias,
            initial_bias_mode,
            initial_bias_window_s,
            device,
            fixed_bias=fixed_bw0,
        )
        ba0 = _resolve_initial_bias(
            bias_a_gt,
            t_a_bias,
            initial_bias_mode,
            initial_bias_window_s,
            device,
            fixed_bias=fixed_ba0,
        )
        vec3zeros = torch.zeros(3, dtype=X_gt.dtype, device=device)
        y0 = torch.cat([vec3zeros, t_gt[0].unsqueeze(-1), bw0, X_gt[0, :3, 3], X_gt[0, :3, 4], ba0], dim=-1).unsqueeze(0)

        _set_model_imu_functions(model, spline, dataset_eval.dt, use_delta_imu)
        model.set_R0(R0)
        sol_pred, R_pred = debias_learning.odeint_SO3(model, y0, R0, t_gt, method=integral_method)
        sol_pred = sol_pred.squeeze(1)
        R_pred = R_pred.squeeze(1)
        rows.append(
            {
                "sequence": seq_name,
                "aoe_deg": float(AOE(X_gt[:, :3, :3], R_pred).item()),
                "ate_vel": float(ATE(X_gt[:, :3, 3], sol_pred[:, 7:10]).item()),
                "ate_pos_m": float(ATE(X_gt[:, :3, 4], sol_pred[:, 10:13]).item()),
            }
        )

    summary = {
        "mean_aoe_deg": sum(float(row["aoe_deg"]) for row in rows) / len(rows),
        "mean_ate_vel": sum(float(row["ate_vel"]) for row in rows) / len(rows),
        "mean_ate_pos_m": sum(float(row["ate_pos_m"]) for row in rows) / len(rows),
    }
    return rows, summary


def Acc_train_RSS_with_init(
    dataset_train: EUROCDataset,
    dataset_val: EUROCDataset,
    outpur_dir: str,
    bw_model,
    ba_model_name: str,
    integral_method="euler",
    device="cuda",
    lr=0.005,
    weight_decay=1e-6,
    epoch=1801,
    lr_step_size: int = 300,
    lr_gamma: float = 0.5,
    train_batch_strategy: str = "random_sequence",
    val_freq: int = 100,
    save_freq: int = 100,
    ba_init_model_path: str | Path | None = None,
):
    """PLATO wrapper for original Debias_IMU Acc_train with optional ba initialization.

    The loss and sampling intentionally match the original RSS Acc_train: one
    random sequence/window per epoch and MSE on velocity/position states.
    """
    del train_batch_strategy
    num_iterations = epoch
    torch.manual_seed(3407)

    mask_flag = {"EUROC": False, "TUM": True, "Fetch": False}[dataset_train.dataset_name]
    model_save_path = Path(outpur_dir) / "Acc_weights"
    model_save_path.mkdir(parents=True, exist_ok=True)
    log_writer_path = Path(outpur_dir) / "Acc_logs"

    ba_model = ba_func_choose(ba_model_name, device)
    _load_ba_model_initialization(ba_model, ba_init_model_path, device)
    model = debias_learning.VFSE23vbiasBase(biasfunc_w=bw_model, biasfunc_a=ba_model, device=device)
    best_loss = float("inf")
    writter = debias_learning.SummaryWriter(str(log_writer_path))
    model.set_bw_zero(False)
    model.set_ba_zero(False)
    model.freeze_func_bw()

    debias_learning.write_parameters("Acc", dataset_train, outpur_dir, ba_model_name, lr, weight_decay, epoch)

    gpu_memory = []
    epoch_times = []
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    loss_total = float("nan")
    for epoch_idx in range(num_iterations):
        epoch_start = debias_learning.start_epoch_profile(device)
        optimizer.zero_grad()
        model.train()

        index = torch.randint(0, dataset_train.length(), (1,)).item()
        (t_gt_batch, X_gt_batch, bwa_gt_batch), t_odeint, coeff, spline_time, eval_indices = dataset_train.get_data_for_biasDy(
            index,
            device,
            mask_flag=mask_flag,
        )
        y0_batch, R0_batch = dataset_train.construct_init_forodeint(t_gt_batch, X_gt_batch, bwa_gt_batch)
        spline = Interpolation.CubicHermiteSpline(spline_time, coeff, device=device)
        model.set_u_func(spline.evaluate)
        model.set_u_dot_func(spline.derivative)
        model.set_R0(R0_batch)
        solution, _ = debias_learning.odeint_SO3(model, y0_batch, R0_batch, t_odeint, method=integral_method)
        if solution.shape[0] != X_gt_batch.shape[0]:
            solution = solution[eval_indices]

        train_loss = 1e6 * torch.nn.MSELoss()(
            solution[..., 7:13],
            torch.cat([X_gt_batch[..., :3, 3], X_gt_batch[..., :3, 4]], dim=-1),
        )
        train_loss.backward()
        optimizer.step()

        writter.add_scalar("train_loss", train_loss.item(), epoch_idx)
        writter.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch_idx)
        scheduler.step()

        if epoch_idx % val_freq == 0:
            print(f"epoch: {epoch_idx}, train_loss: {train_loss.item()}")
            with torch.no_grad():
                model.eval()
                loss_total = 0.0
                for index in range(dataset_val.length()):
                    (t_gt_batch, X_gt_batch, bwa_gt_batch), t_odeint, coeff, spline_time, eval_indices = dataset_val.get_data_for_biasDy(
                        index,
                        device,
                        mask_flag=mask_flag,
                    )
                    y0_batch, R0_batch = dataset_train.construct_init_forodeint(t_gt_batch, X_gt_batch, bwa_gt_batch)
                    spline = Interpolation.CubicHermiteSpline(spline_time, coeff, device=device)
                    model.set_u_func(spline.evaluate)
                    model.set_u_dot_func(spline.derivative)
                    model.set_R0(R0_batch)
                    solution, _ = debias_learning.odeint_SO3(model, y0_batch, R0_batch, t_odeint, method=integral_method)
                    if solution.shape[0] != X_gt_batch.shape[0]:
                        solution = solution[eval_indices]
                    loss = 1e6 * torch.nn.MSELoss()(
                        solution[..., 7:13],
                        torch.cat([X_gt_batch[..., :3, 3], X_gt_batch[..., :3, 4]], dim=-1),
                    )
                    loss_total += loss.item()
                writter.add_scalar("val_loss", loss_total, epoch_idx)
                print(f"epoch: {epoch_idx}, val_loss: {loss_total}")

            if loss_total < best_loss:
                best_loss = loss_total
                torch.save(
                    {
                        "func_ba_model": model.biasfunc_a.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch_idx,
                        "best_loss": best_loss,
                    },
                    model_save_path / "best_val_model.pt",
                )

        if epoch_idx % save_freq == 0:
            torch.save(
                {
                    "func_ba_model": model.biasfunc_a.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch_idx,
                    "valloss": loss_total,
                },
                model_save_path / f"model_epoch_{epoch_idx}.pt",
            )
        epoch_time, peak_memory_mb = debias_learning.finish_epoch_profile(device, epoch_start)
        epoch_times.append(epoch_time)
        gpu_memory.append(peak_memory_mb)
    writter.close()
    return debias_learning.write_stage_profile(outpur_dir, "Acc", epoch_times, gpu_memory)


def _select_full_rollout_checkpoint(
    *,
    args: argparse.Namespace,
    params: dict,
    stage: str,
    gyro_checkpoint: Path | None,
    use_delta_imu: bool,
) -> Path | None:
    eval_seqs = list(args.full_rollout_val_seqs or params["val_seqs"])
    if not eval_seqs:
        raise ValueError("--full-rollout-select requires at least one validation sequence.")

    stage_dir = args.outputdir / ("Gyro_weights" if stage == "gyro" else "Acc_weights")
    candidates = _checkpoint_candidates(stage_dir)
    if not candidates:
        print(f"full-rollout selection skipped for {stage}: no saved checkpoints in {stage_dir}", flush=True)
        return None

    device = args.full_rollout_selection_device or args.test_device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"full-rollout selection device={device} was requested, but CUDA is not available.")
    dataset_eval = _make_full_rollout_dataset(args, params, eval_seqs)

    fixed_bw0 = None
    fixed_ba0 = None
    if args.initial_bias_mode in {"train_global_mean", "train_global_median"}:
        stat = "mean" if args.initial_bias_mode.endswith("mean") else "median"
        fixed_bw0, fixed_ba0 = _train_global_bias_from_cache(args.cache_dir, list(params["train_seqs"]), stat, device)

    selection_rows = []
    best_row = None
    best_path = None
    for checkpoint in candidates:
        if stage == "gyro":
            bw_func = bw_func_choose(args.network_type_Gyro, device=device)
            bw_func.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["func_bw_model"])
            ba_func = ba_func_choose(args.network_type_Acc, device=device)
        else:
            if gyro_checkpoint is None:
                raise ValueError("Acc full-rollout selection requires a gyro checkpoint.")
            bw_func = bw_func_choose(args.network_type_Gyro, device=device)
            bw_func.load_state_dict(torch.load(gyro_checkpoint, map_location=device, weights_only=True)["func_bw_model"])
            ba_func = ba_func_choose(args.network_type_Acc, device=device)
            ba_func.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["func_ba_model"])
        with torch.no_grad():
            per_sequence, summary = _evaluate_full_rollout_metrics(
                dataset_eval,
                bw_func,
                ba_func,
                args.integral_method,
                device,
                use_delta_imu,
                args.initial_bias_mode,
                args.initial_bias_window_s,
                fixed_bw0,
                fixed_ba0,
            )
        score = _full_rollout_score(summary, args)
        row = {
            "stage": stage,
            "label": _checkpoint_label(checkpoint),
            "checkpoint": str(checkpoint),
            "score": score,
            **summary,
            "per_sequence": per_sequence,
        }
        selection_rows.append(row)
        print(
            f"full_rollout_select stage={stage} label={row['label']} "
            f"score={score:.6g} AOE={summary['mean_aoe_deg']:.3f} "
            f"vel={summary['mean_ate_vel']:.3f} pos={summary['mean_ate_pos_m']:.3f}",
            flush=True,
        )
        if best_row is None or score < float(best_row["score"]):
            best_row = row
            best_path = checkpoint

    assert best_row is not None and best_path is not None
    selection_dir = args.outputdir / "full_rollout_selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    json_path = selection_dir / f"{stage}_selection.json"
    json_path.write_text(json.dumps({"eval_seqs": eval_seqs, "selected": best_row, "candidates": selection_rows}, indent=2) + "\n")
    csv_path = selection_dir / f"{stage}_selection.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["stage", "label", "checkpoint", "score", "mean_aoe_deg", "mean_ate_vel", "mean_ate_pos_m"],
        )
        writer.writeheader()
        for row in selection_rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    selected_path = stage_dir / "best_full_rollout_model.pt"
    shutil.copy2(best_path, selected_path)
    print(
        f"selected full-rollout {stage} checkpoint: {_checkpoint_label(best_path)} -> {selected_path}",
        flush=True,
    )
    return selected_path


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device={args.device} was requested, but CUDA is not available.")
    if args.test_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"test-device={args.test_device} was requested, but CUDA is not available.")
    if not args.only_test and args.epoch <= 0:
        raise ValueError("--epoch must be positive unless --only-test is used")
    if args.stop_after_gyro and args.skip_gyro_train:
        raise ValueError("--stop-after-gyro cannot be combined with --skip-gyro-train")
    if args.only_test and (args.stop_after_gyro or args.skip_gyro_train or args.skip_test):
        raise ValueError("--only-test cannot be combined with train-stage control flags")
    if args.acc_loss_type in {"interval_nll", "trajectory_nll"} and not args.readable_val_metrics:
        raise ValueError("PLATO preintegration NLL variants are implemented in the readable-metrics trainer.")
    if args.acc_loss_type in {"interval_nll", "trajectory_nll"} and args.acc_loss != "se23":
        raise ValueError("--acc-loss must be se23 for PLATO preintegration NLL variants.")
    if args.acc_loss == "rss" and args.acc_loss_type != "rss":
        raise ValueError("--acc-loss rss requires --acc-loss-type rss.")
    if args.acc_loss_type == "rss" and args.acc_loss != "rss":
        raise ValueError("--acc-loss-type rss requires --acc-loss rss.")
    if args.acc_loss == "rss" and args.readable_val_metrics:
        raise ValueError("--acc-loss rss uses original Debias_IMU Acc_train and requires --no-readable-val-metrics.")
    if args.acc_loss_type in {"interval_nll", "trajectory_nll"} and args.interval_nll_mode != "nll_joint" and args.nll_train_gyro:
        raise ValueError("--nll-train-gyro is only meaningful with --interval-nll-mode=nll_joint.")
    if args.acc_loss_type == "trajectory_nll" and args.interval_covariance_gradient_mode != "analytic_sensitivity":
        raise ValueError("--acc-loss-type=trajectory_nll currently requires --interval-covariance-gradient-mode=analytic_sensitivity.")
    if (args.full_rollout_select or args.full_rollout_select_only) and args.only_test:
        raise ValueError("full-rollout checkpoint selection cannot be combined with --only-test.")
    if args.full_rollout_select_only:
        args.full_rollout_select = True
    if args.ba_init_model_path is not None and not args.readable_val_metrics and args.acc_loss != "rss":
        raise ValueError("--ba-init-model-path requires the PLATO readable-metrics trainer unless --acc-loss=rss.")
    if args.bw_init_model_path is not None and args.gyro_loss != "ad":
        raise ValueError("--bw-init-model-path is only implemented for AD gyro training.")
    if _uses_delta_imu(args.network_type_Acc) and _requires_spline_derivative(args.network_type_Gyro):
        raise ValueError(
            "Accel finite-difference ablation cannot be combined with a gyro model that requires spline derivatives."
        )
    if _uses_delta_imu(args.network_type_Gyro, args.network_type_Acc) and (
        args.gyro_loss == "analytic" or args.acc_loss == "analytic"
    ):
        raise ValueError("Analytic training is only wired for non-delta IMU networks in this EuRoC runner.")

    use_delta_imu = _uses_delta_imu(args.network_type_Gyro, args.network_type_Acc)
    args.outputdir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.test_cache_dir.mkdir(parents=True, exist_ok=True)
    params = make_dataset_parameters(args)

    print("EuRoC Debias_IMU pipeline")
    print(f"train_seqs={params['train_seqs']}")
    print(f"val_seqs={params['val_seqs']}")
    print(f"test_seqs={args.test_seqs}")
    print(f"train_supervision={args.train_supervision}")
    print(f"trainval_gt_root={params['trainval_gt_root']}")
    print(f"trainval_gt_source={params['trainval_gt_source']}")
    print(f"supervision_timing={args.supervision_timing}")
    print(f"network_type_Gyro={args.network_type_Gyro}, network_type_Acc={args.network_type_Acc}")
    print(f"gyro_loss={args.gyro_loss}, acc_loss={args.acc_loss}, acc_loss_type={args.acc_loss_type}")
    ode_grid_window_s = (args.loss_window - 1) * args.dt
    print(f"device={args.device}, test_device={args.test_device}, epoch={args.epoch}, loss_window={args.loss_window}")
    if args.supervision_timing == "source":
        print(
            f"source_timing_window={args.loss_window} source poses; physical span is read from source timestamps, "
            f"not assumed from IMU dt. ode_grid_window_s={ode_grid_window_s:.6g} from dt={args.dt}"
        )
    else:
        print(f"nominal_imu_window_s={ode_grid_window_s:.6g} from loss_window={args.loss_window}, dt={args.dt}")
    print(f"batch_size={args.batch_size}, train_batch_strategy={args.train_batch_strategy}")
    if args.acc_loss_type in {"interval_nll", "trajectory_nll"}:
        print(
            f"{args.acc_loss_type} "
            f"mode={args.interval_nll_mode}, train_gyro={args.nll_train_gyro}, "
            f"sigma_a_nominal={args.interval_sigma_a_nominal}, sigma_g_nominal={args.interval_sigma_g_nominal}, "
            f"target_cov_floor={args.interval_target_cov_floor}, jitter={args.interval_jitter}, "
            f"lr_sigma={args.interval_lr_sigma}, detach_covariance_theta={args.interval_detach_covariance_theta}, "
            f"covariance_gradient_mode={args.interval_covariance_gradient_mode}, "
            f"noise_init={args.interval_noise_init_model_path}, "
            f"trajectory_mode={args.trajectory_supervision_mode}, "
            f"trajectory_stride={args.trajectory_supervision_stride}, "
            f"trajectory_max_states={args.trajectory_max_supervised_states}"
        )

    gyro_ckpt = resolve_checkpoint(
        args.outputdir,
        "Gyro_weights",
        args.bw_model_path,
        args.epoch,
        args.save_freq,
        args.use_best_val,
    )
    acc_ckpt = resolve_checkpoint(
        args.outputdir,
        "Acc_weights",
        args.ba_model_path,
        args.epoch,
        args.save_freq,
        args.use_best_val,
    )

    if not args.only_test and not args.full_rollout_select_only:
        dataset_train = EUROCDataset(
            **params,
            data_cache_dir=str(args.cache_dir),
            mode="train",
            recompute=args.recompute_cache,
        )
        dataset_val = EUROCDataset(
            **params,
            data_cache_dir=str(args.cache_dir),
            mode="val",
            recompute=False,
        )

        t_start = time.time()
        gyro_profile = None
        if not args.skip_gyro_train:
            gyro_train_fn = (
                Gyro_train_Analytic
                if args.gyro_loss == "analytic"
                else (
                    Gyro_train_with_readable_metrics
                    if args.readable_val_metrics or args.bw_init_model_path is not None
                    else Gyro_train
                )
            )
            gyro_kwargs = {}
            if args.gyro_loss == "analytic":
                gyro_kwargs = {
                    "analytic_grad_sign": args.gyro_analytic_grad_sign,
                    "analytic_grad_scale": args.gyro_analytic_grad_scale,
                    "analytic_grad_clip_norm": args.gyro_analytic_grad_clip_norm,
                    "analytic_residual_jacobian": args.gyro_analytic_residual_jacobian,
                    "analytic_invariant_error": args.gyro_analytic_invariant_error,
                    "inner_vjp": args.inner_vjp,
                }
            elif args.readable_val_metrics or args.bw_init_model_path is not None:
                if args.bw_init_model_path is not None:
                    gyro_kwargs["bw_init_model_path"] = args.bw_init_model_path
                gyro_kwargs["gyro_bias_rate_regularization_weight"] = args.gyro_bias_rate_regularization_weight
            gyro_profile = gyro_train_fn(
                dataset_train,
                dataset_val,
                str(args.outputdir),
                args.network_type_Gyro,
                args.integral_method,
                args.device,
                args.lr_Gyro,
                args.weight_decay_Gyro,
                epoch=args.epoch,
                lr_step_size=args.gyro_lr_step_size,
                lr_gamma=args.gyro_lr_gamma,
                train_batch_strategy=args.train_batch_strategy,
                val_freq=args.val_freq,
                save_freq=args.save_freq,
                **gyro_kwargs,
            )
            if args.full_rollout_select and args.full_rollout_select_gyro and args.bw_model_path is None:
                selected_gyro = _select_full_rollout_checkpoint(
                    args=args,
                    params=params,
                    stage="gyro",
                    gyro_checkpoint=None,
                    use_delta_imu=use_delta_imu,
                )
                if selected_gyro is not None:
                    gyro_ckpt = selected_gyro
        if args.stop_after_gyro:
            return

        bw_func = bw_func_choose(args.network_type_Gyro, device=args.device)
        bw_func.load_state_dict(torch.load(gyro_ckpt, map_location=args.device, weights_only=True)["func_bw_model"])
        if args.acc_loss_type in {"interval_nll", "trajectory_nll"}:
            acc_train_fn = Acc_train_interval_preintegration_NLL_with_readable_metrics
        elif args.acc_loss == "analytic":
            acc_train_fn = Acc_train_SE23_Analytic
        elif args.acc_loss == "rss":
            acc_train_fn = Acc_train_RSS_with_init
        else:
            acc_train_fn = Acc_train_SE23_with_readable_metrics if args.readable_val_metrics else Acc_train_SE23
        acc_kwargs = {}
        if args.acc_loss_type in {"interval_nll", "trajectory_nll"}:
            acc_kwargs.update(
                {
                    "interval_nll_mode": args.interval_nll_mode,
                    "interval_sigma_g_nominal": args.interval_sigma_g_nominal,
                    "interval_sigma_a_nominal": args.interval_sigma_a_nominal,
                    "interval_sigma_g_floor2": args.interval_sigma_g_floor2,
                    "interval_sigma_a_floor2": args.interval_sigma_a_floor2,
                    "interval_target_cov_floor": args.interval_target_cov_floor,
                    "interval_jitter": args.interval_jitter,
                    "interval_prior_lambda_g": args.interval_prior_lambda_g,
                    "interval_prior_lambda_a": args.interval_prior_lambda_a,
                    "interval_lr_sigma": args.interval_lr_sigma,
                    "interval_detach_covariance_theta": args.interval_detach_covariance_theta,
                    "interval_covariance_gradient_mode": args.interval_covariance_gradient_mode,
                    "interval_debug_covariance_gradients": args.interval_debug_covariance_gradients,
                    "interval_noise_init_model_path": args.interval_noise_init_model_path,
                    "nll_train_gyro": args.nll_train_gyro,
                    "ba_init_model_path": args.ba_init_model_path,
                    "nll_family": "trajectory_state" if args.acc_loss_type == "trajectory_nll" else "interval",
                    "trajectory_supervision_mode": args.trajectory_supervision_mode,
                    "trajectory_supervision_stride": args.trajectory_supervision_stride,
                    "trajectory_max_supervised_states": args.trajectory_max_supervised_states,
                }
            )
        elif args.acc_loss == "analytic":
            acc_kwargs = {
                "analytic_grad_sign": args.analytic_grad_sign,
                "analytic_grad_scale": args.analytic_grad_scale,
                "analytic_grad_clip_norm": args.analytic_grad_clip_norm,
                "analytic_residual_jacobian": args.analytic_residual_jacobian,
                "analytic_invariant_error": args.analytic_invariant_error,
                "inner_vjp": args.inner_vjp,
            }
        if args.readable_val_metrics and (args.acc_loss_type == "interval_nll" or args.acc_loss != "analytic"):
            acc_kwargs["bw_model_name"] = args.network_type_Gyro
            if args.ba_init_model_path is not None:
                acc_kwargs["ba_init_model_path"] = args.ba_init_model_path
            if args.acc_loss_type in {"se23", "interval_nll", "trajectory_nll"}:
                acc_kwargs["acc_loss_rot_weight"] = args.acc_loss_rot_weight
                acc_kwargs["acc_loss_vel_weight"] = args.acc_loss_vel_weight
                acc_kwargs["acc_loss_pos_weight"] = args.acc_loss_pos_weight
            if args.acc_loss_type == "se23" and args.acc_loss != "analytic":
                acc_kwargs["acc_bias_rate_regularization_weight"] = args.acc_bias_rate_regularization_weight
        if args.acc_loss == "rss":
            if args.ba_init_model_path is not None:
                acc_kwargs["ba_init_model_path"] = args.ba_init_model_path
            acc_profile = acc_train_fn(
                dataset_train,
                dataset_val,
                str(args.outputdir),
                bw_func,
                args.network_type_Acc,
                args.integral_method,
                args.device,
                args.lr_Acc,
                args.weight_decay_Acc,
                epoch=args.epoch,
                lr_step_size=args.acc_lr_step_size,
                lr_gamma=args.acc_lr_gamma,
                train_batch_strategy=args.train_batch_strategy,
                val_freq=args.val_freq,
                save_freq=args.save_freq,
                **acc_kwargs,
            )
        else:
            acc_profile = acc_train_fn(
                dataset_train,
                dataset_val,
                str(args.outputdir),
                bw_func,
                args.network_type_Acc,
                args.integral_method,
                args.device,
                args.lr_Acc,
                args.weight_decay_Acc,
                epoch=args.epoch,
                lr_step_size=args.acc_lr_step_size,
                lr_gamma=args.acc_lr_gamma,
                train_batch_strategy=args.train_batch_strategy,
                val_freq=args.val_freq,
                save_freq=args.save_freq,
                **acc_kwargs,
            )
        if args.full_rollout_select and args.ba_model_path is None and args.acc_loss_type == "se23":
            selected_acc = _select_full_rollout_checkpoint(
                args=args,
                params=params,
                stage="acc",
                gyro_checkpoint=gyro_ckpt,
                use_delta_imu=use_delta_imu,
            )
            if selected_acc is not None:
                acc_ckpt = selected_acc

        elapsed = time.time() - t_start
        with open(args.outputdir / "time_memory_record.txt", "w", encoding="utf-8") as f:
            f.write(
                format_training_profile(
                    elapsed,
                    gyro_profile,
                    acc_profile,
                    extra={
                        "dataset": "EUROC",
                        "train_supervision": args.train_supervision,
                        "trainval_gt_root": params["trainval_gt_root"],
                        "trainval_gt_source": params["trainval_gt_source"],
                        "supervision_timing": args.supervision_timing,
                        "test_supervision_source": "state_groundtruth_estimate0",
                        "data_root": str(args.data_root),
                        "train_seqs": list(params["train_seqs"]),
                        "val_seqs": list(params["val_seqs"]),
                        "test_seqs": list(args.test_seqs),
                        "loss_window": args.loss_window,
                        "ode_grid_window_s": (args.loss_window - 1) * args.dt,
                        "nominal_imu_window_s": (args.loss_window - 1) * args.dt
                        if args.supervision_timing == "imu"
                        else None,
                        "source_timing_window_note": (
                            "loss_window counts source poses; physical span is read from source timestamps"
                            if args.supervision_timing == "source"
                            else None
                        ),
                        "batch_size": args.batch_size,
                        "integral_method": args.integral_method,
                        "network_type_Gyro": args.network_type_Gyro,
                        "network_type_Acc": args.network_type_Acc,
                        "gyro_loss": args.gyro_loss,
                        "acc_loss": args.acc_loss,
                        "acc_loss_type": args.acc_loss_type,
                        "acc_loss_rot_weight": args.acc_loss_rot_weight,
                        "acc_loss_vel_weight": args.acc_loss_vel_weight,
                        "acc_loss_pos_weight": args.acc_loss_pos_weight,
                        "gyro_bias_rate_regularization_weight": args.gyro_bias_rate_regularization_weight,
                        "acc_bias_rate_regularization_weight": args.acc_bias_rate_regularization_weight,
                        "full_rollout_select": args.full_rollout_select,
                        "full_rollout_select_only": args.full_rollout_select_only,
                        "full_rollout_select_gyro": args.full_rollout_select_gyro,
                        "full_rollout_val_seqs": list(args.full_rollout_val_seqs or params["val_seqs"]),
                        "full_rollout_selection_cache_dir": str(args.full_rollout_selection_cache_dir),
                        "full_rollout_selection_device": args.full_rollout_selection_device,
                        "full_rollout_score_aoe_weight": args.full_rollout_score_aoe_weight,
                        "full_rollout_score_vel_weight": args.full_rollout_score_vel_weight,
                        "full_rollout_score_pos_weight": args.full_rollout_score_pos_weight,
                        "full_rollout_score_aoe_scale": args.full_rollout_score_aoe_scale,
                        "full_rollout_score_vel_scale": args.full_rollout_score_vel_scale,
                        "full_rollout_score_pos_scale": args.full_rollout_score_pos_scale,
                        "interval_nll_mode": args.interval_nll_mode,
                        "interval_sigma_g_nominal": args.interval_sigma_g_nominal,
                        "interval_sigma_a_nominal": args.interval_sigma_a_nominal,
                        "interval_sigma_g_floor2": args.interval_sigma_g_floor2,
                        "interval_sigma_a_floor2": args.interval_sigma_a_floor2,
                        "interval_target_cov_floor": args.interval_target_cov_floor,
                        "interval_jitter": args.interval_jitter,
                        "interval_prior_lambda_g": args.interval_prior_lambda_g,
                        "interval_prior_lambda_a": args.interval_prior_lambda_a,
                        "interval_lr_sigma": args.interval_lr_sigma,
                        "interval_detach_covariance_theta": args.interval_detach_covariance_theta,
                        "interval_covariance_gradient_mode": args.interval_covariance_gradient_mode,
                        "interval_debug_covariance_gradients": args.interval_debug_covariance_gradients,
                        "interval_noise_init_model_path": str(args.interval_noise_init_model_path),
                        "trajectory_supervision_mode": args.trajectory_supervision_mode,
                        "trajectory_supervision_stride": args.trajectory_supervision_stride,
                        "trajectory_max_supervised_states": args.trajectory_max_supervised_states,
                        "nll_train_gyro": args.nll_train_gyro,
                        "ba_init_model_path": str(args.ba_init_model_path),
                        "use_delta_imu": use_delta_imu,
                        "readable_val_metrics": args.readable_val_metrics,
                        "device": args.device,
                        "test_device": args.test_device,
                        "initial_bias_mode": args.initial_bias_mode,
                        "initial_bias_window_s": args.initial_bias_window_s,
                        "skip_gyro_train": args.skip_gyro_train,
                        "stop_after_gyro": args.stop_after_gyro,
                        "skip_test": args.skip_test,
                        "gyro_checkpoint": str(gyro_ckpt),
                        "acc_checkpoint": str(acc_ckpt),
                        "exclude_train_seq": list(args.exclude_train_seq),
                        "inner_vjp": args.inner_vjp,
                        "analytic_grad_sign": args.analytic_grad_sign,
                        "analytic_grad_scale": args.analytic_grad_scale,
                        "analytic_grad_clip_norm": args.analytic_grad_clip_norm,
                        "analytic_residual_jacobian": args.analytic_residual_jacobian,
                        "analytic_invariant_error": args.analytic_invariant_error,
                        "gyro_analytic_grad_sign": args.gyro_analytic_grad_sign,
                        "gyro_analytic_grad_scale": args.gyro_analytic_grad_scale,
                        "gyro_analytic_grad_clip_norm": args.gyro_analytic_grad_clip_norm,
                        "gyro_analytic_residual_jacobian": args.gyro_analytic_residual_jacobian,
                        "gyro_analytic_invariant_error": args.gyro_analytic_invariant_error,
                    },
                )
            )
    elif args.full_rollout_select_only:
        if args.full_rollout_select_gyro and args.bw_model_path is None:
            selected_gyro = _select_full_rollout_checkpoint(
                args=args,
                params=params,
                stage="gyro",
                gyro_checkpoint=None,
                use_delta_imu=use_delta_imu,
            )
            if selected_gyro is not None:
                gyro_ckpt = selected_gyro
        if args.ba_model_path is None:
            selected_acc = _select_full_rollout_checkpoint(
                args=args,
                params=params,
                stage="acc",
                gyro_checkpoint=gyro_ckpt,
                use_delta_imu=use_delta_imu,
            )
            if selected_acc is not None:
                acc_ckpt = selected_acc

    if args.skip_test:
        return

    dataset_test = EUROCDataset(
        **params,
        data_cache_dir=str(args.test_cache_dir),
        mode="test",
        recompute=args.test_recompute,
    )
    fixed_bw0 = None
    fixed_ba0 = None
    if args.initial_bias_mode in {"train_global_mean", "train_global_median"}:
        stat = "mean" if args.initial_bias_mode.endswith("mean") else "median"
        fixed_bw0, fixed_ba0 = _train_global_bias_from_cache(
            args.cache_dir,
            list(params["train_seqs"]),
            stat,
            args.test_device,
        )
        print(
            f"train-global initial bias ({stat}) from {len(params['train_seqs'])} train sequences: "
            f"bw0={fixed_bw0.cpu().tolist()}, ba0={fixed_ba0.cpu().tolist()}"
        )

    bw_func = bw_func_choose(args.network_type_Gyro, device=args.test_device)
    bw_func.load_state_dict(torch.load(gyro_ckpt, map_location=args.test_device, weights_only=True)["func_bw_model"])
    ba_func = ba_func_choose(args.network_type_Acc, device=args.test_device)
    ba_func.load_state_dict(torch.load(acc_ckpt, map_location=args.test_device, weights_only=True)["func_ba_model"])
    with torch.no_grad():
        Test_with_configured_aux(
            dataset_test,
            str(args.outputdir),
            bw_func,
            ba_func,
            args.integral_method,
            device=args.test_device,
            recompute=args.test_recompute,
            use_delta_imu=use_delta_imu,
            initial_bias_mode=args.initial_bias_mode,
            initial_bias_window_s=args.initial_bias_window_s,
            fixed_bw0=fixed_bw0,
            fixed_ba0=fixed_ba0,
        )

    result_root = args.outputdir / "Sequnces-results"
    lines = []
    for seq_dir in sorted(path for path in result_root.iterdir() if path.is_dir()):
        path_tmp = seq_dir / "FullTra_result.p"
        if args.skip_plots:
            data = pload(str(path_tmp))
            aoe = float(AOE(data["X_gt"][:, :3, :3], data["R_pred"]).item())
            ate_vel = float(ATE(data["X_gt"][:, :3, 3], data["sol_pred"][:, 7:10]).item())
            ate_pos = float(ATE(data["X_gt"][:, :3, 4], data["sol_pred"][:, 10:13]).item())
        else:
            aoe, ate_vel, ate_pos = simple_visualization(str(path_tmp), return_pos=True)
            plot_bias(str(path_tmp), dataset_test)
        lines.append(f"sequence: {seq_dir.name:20} AOE: {aoe:.3f} ATE vel: {ate_vel:.3f} ATE pos: {ate_pos:.3f}")
    if lines:
        with open(args.outputdir / "evaluation_summary.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            f.write("\n")


if __name__ == "__main__":
    main()
