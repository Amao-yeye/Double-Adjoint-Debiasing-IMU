#!/usr/bin/env python3
"""Run EuRoC Debias_IMU training with PLATO-owned bias dynamics networks.

This is PLATO glue code.  It keeps external/Debias_IMU untouched while reusing
its EuRoC dataset loader, losses, and evaluation routines.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from run_liasuw_debias_ad_pipeline import (
    ACC_M2_NAME,
    ACC_PHYS_DELTA_NAME,
    ACC_PHYS_EQ_NAME,
    ACC_PHYS_RATE_NAME,
    DEBIAS_ROOT,
    GYRO_M2_NAME,
    GYRO_PHYS_DELTA_NAME,
    GYRO_PHYS_EQ_NAME,
    GYRO_PHYS_RATE_NAME,
    PLATO_ROOT,
    Acc_train_SE23_with_readable_metrics,
    Gyro_train_with_readable_metrics,
    Test_with_configured_aux,
    _requires_spline_derivative,
    _uses_delta_imu,
    ba_func_choose,
    bw_func_choose,
    resolve_checkpoint,
)
from dataset import EUROCDataset
from learning import Acc_train_SE23, Gyro_train, format_training_profile, plot_bias, simple_visualization


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
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PLATO_ROOT / "results" / "debias_imu" / "cache" / "EUROC_gt_w16",
    )
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
    parser.add_argument("--train-seqs", nargs="+", default=DEFAULT_TRAIN_SEQS)
    parser.add_argument("--test-seqs", nargs="+", default=DEFAULT_TEST_SEQS)
    parser.add_argument("--integral_method", type=str, default="euler")
    parser.add_argument(
        "--network_type_Gyro",
        type=str,
        default=GYRO_PHYS_DELTA_NAME,
        help=(
            f"Gyro network. PLATO options: {GYRO_M2_NAME}, {GYRO_PHYS_EQ_NAME}, "
            f"{GYRO_PHYS_RATE_NAME}, {GYRO_PHYS_DELTA_NAME}. External default: bw_func_net."
        ),
    )
    parser.add_argument(
        "--network_type_Acc",
        type=str,
        default=ACC_PHYS_DELTA_NAME,
        help=(
            f"Accel network. PLATO options: {ACC_M2_NAME}, {ACC_PHYS_EQ_NAME}, "
            f"{ACC_PHYS_RATE_NAME}, {ACC_PHYS_DELTA_NAME}. External default: ba_func_net."
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
    parser.add_argument("--recompute-cache", action="store_true")
    parser.add_argument("--test-recompute", action="store_true")
    parser.add_argument("--only-test", action="store_true")
    parser.add_argument("--bw_model_path", type=Path, default=None)
    parser.add_argument("--ba_model_path", type=Path, default=None)
    parser.add_argument("--use-best-val", action="store_true")
    parser.add_argument(
        "--readable-val-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print validation RMSE metrics alongside the original scaled Debias_IMU losses.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def make_dataset_parameters(args: argparse.Namespace) -> dict:
    return {
        "dataset_name": "EUROC",
        "data_dir": str(args.data_root),
        "train_seqs": list(args.train_seqs),
        "val_seqs": list(args.train_seqs),
        "test_seqs": list(args.test_seqs),
        "percent_for_val": args.percent_for_val,
        "batch_size": args.batch_size,
        "loss_window": args.loss_window,
        "dt": args.dt,
        "sg_window_bw": args.sg_window_bw,
        "sg_order_bw": args.sg_order_bw,
        "sg_window_ba": args.sg_window_ba,
        "sg_order_ba": args.sg_order_ba,
        "trainval_gt_root": None,
        "test_gt_root": None,
        "trainval_gt_source": "state_groundtruth_estimate0",
        "test_gt_source": "state_groundtruth_estimate0",
        "trainval_supervision_timing": "imu",
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device={args.device} was requested, but CUDA is not available.")
    if args.test_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"test-device={args.test_device} was requested, but CUDA is not available.")
    if not args.only_test and args.epoch <= 0:
        raise ValueError("--epoch must be positive unless --only-test is used")
    if _uses_delta_imu(args.network_type_Acc) and _requires_spline_derivative(args.network_type_Gyro):
        raise ValueError(
            "Accel finite-difference ablation cannot be combined with a gyro model that requires spline derivatives."
        )

    use_delta_imu = _uses_delta_imu(args.network_type_Gyro, args.network_type_Acc)
    args.outputdir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.test_cache_dir.mkdir(parents=True, exist_ok=True)
    params = make_dataset_parameters(args)

    print("EuRoC Debias_IMU pipeline")
    print(f"train_seqs={args.train_seqs}")
    print(f"test_seqs={args.test_seqs}")
    print("trainval_supervision_source=state_groundtruth_estimate0")
    print(f"network_type_Gyro={args.network_type_Gyro}, network_type_Acc={args.network_type_Acc}")
    print(f"device={args.device}, test_device={args.test_device}, epoch={args.epoch}, loss_window={args.loss_window}")
    print(f"batch_size={args.batch_size}, train_batch_strategy={args.train_batch_strategy}")

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

    if not args.only_test:
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
        gyro_train_fn = Gyro_train_with_readable_metrics if args.readable_val_metrics else Gyro_train
        acc_train_fn = Acc_train_SE23_with_readable_metrics if args.readable_val_metrics else Acc_train_SE23
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
        )

        bw_func = bw_func_choose(args.network_type_Gyro, device=args.device)
        bw_func.load_state_dict(torch.load(gyro_ckpt, map_location=args.device, weights_only=True)["func_bw_model"])
        acc_kwargs = {}
        if args.readable_val_metrics:
            acc_kwargs["bw_model_name"] = args.network_type_Gyro
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

        elapsed = time.time() - t_start
        with open(args.outputdir / "time_memory_record.txt", "w", encoding="utf-8") as f:
            f.write(
                format_training_profile(
                    elapsed,
                    gyro_profile,
                    acc_profile,
                    extra={
                        "dataset": "EUROC",
                        "trainval_supervision_source": "state_groundtruth_estimate0",
                        "test_supervision_source": "state_groundtruth_estimate0",
                        "data_root": str(args.data_root),
                        "train_seqs": list(args.train_seqs),
                        "test_seqs": list(args.test_seqs),
                        "loss_window": args.loss_window,
                        "batch_size": args.batch_size,
                        "integral_method": args.integral_method,
                        "network_type_Gyro": args.network_type_Gyro,
                        "network_type_Acc": args.network_type_Acc,
                        "use_delta_imu": use_delta_imu,
                        "readable_val_metrics": args.readable_val_metrics,
                        "device": args.device,
                        "test_device": args.test_device,
                        "gyro_checkpoint": str(gyro_ckpt),
                        "acc_checkpoint": str(acc_ckpt),
                    },
                )
            )

    dataset_test = EUROCDataset(
        **params,
        data_cache_dir=str(args.test_cache_dir),
        mode="test",
        recompute=args.test_recompute,
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
        )

    result_root = args.outputdir / "Sequnces-results"
    lines = []
    for seq_dir in sorted(path for path in result_root.iterdir() if path.is_dir()):
        path_tmp = seq_dir / "FullTra_result.p"
        if args.skip_plots:
            continue
        aoe, ate_vel, ate_pos = simple_visualization(str(path_tmp), return_pos=True)
        lines.append(f"sequence: {seq_dir.name:20} AOE: {aoe:.3f} ATE vel: {ate_vel:.3f} ATE pos: {ate_pos:.3f}")
        plot_bias(str(path_tmp), dataset_test)
    if lines:
        with open(args.outputdir / "evaluation_summary.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            f.write("\n")


if __name__ == "__main__":
    main()
