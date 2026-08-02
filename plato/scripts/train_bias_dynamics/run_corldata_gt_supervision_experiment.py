#!/usr/bin/env python3
"""Run a config-driven corldata GT-supervision Debias_IMU experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


PLATO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE = PLATO_ROOT / "plato" / "scripts" / "train_bias_dynamics" / "run_liasuw_debias_ad_pipeline.py"

INPUT_MODE_DEFAULT_NETWORKS = {
    "u": ("bw_func_m2_no_deriv", "ba_func_m2_no_deriv"),
    "u_dot_u": ("bw_func_net", "ba_func_net"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--test-device", type=str, default=None)
    parser.add_argument("--outputdir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--train-seqs", nargs="+", default=None)
    parser.add_argument("--val-seqs", nargs="+", default=None)
    parser.add_argument("--test-seqs", nargs="+", default=None)
    parser.add_argument("--recompute-cache", action="store_true")
    parser.add_argument("--test-recompute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def add_option(argv: list[str], name: str, value) -> None:
    if value is None:
        return
    argv.extend([name, str(value)])


def add_list_option(argv: list[str], name: str, values) -> None:
    items = as_list(values)
    if items:
        argv.append(name)
        argv.extend(items)


def add_bool_flag(argv: list[str], name: str, enabled: bool) -> None:
    if enabled:
        argv.append(name)


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"{args.config} must contain a YAML mapping")
    if cfg.get("dataset") != "corldata":
        raise ValueError("This runner only accepts dataset: corldata")
    if cfg.get("supervision") != "gt":
        raise ValueError("This runner is intentionally limited to supervision: gt")
    if cfg.get("loss_type") != "se23_gt_supervision":
        raise ValueError("Expected loss_type: se23_gt_supervision")

    input_mode = cfg.get("input_mode", "u")
    if input_mode not in INPUT_MODE_DEFAULT_NETWORKS:
        raise ValueError(f"Unsupported input_mode {input_mode!r}; expected one of {sorted(INPUT_MODE_DEFAULT_NETWORKS)}")
    default_gyro, default_acc = INPUT_MODE_DEFAULT_NETWORKS[input_mode]
    network_type_gyro = cfg.get("network_type_Gyro", default_gyro)
    network_type_acc = cfg.get("network_type_Acc", default_acc)

    outputdir = args.outputdir or Path(cfg["outputdir"])
    cache_dir = args.cache_dir or Path(cfg["cache_dir"])
    outputdir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    resolved = dict(cfg)
    train_seqs = args.train_seqs if args.train_seqs is not None else as_list(cfg.get("train_seqs"))
    val_seqs = args.val_seqs if args.val_seqs is not None else as_list(cfg.get("val_seqs", train_seqs))
    test_seqs = args.test_seqs if args.test_seqs is not None else as_list(cfg.get("test_seqs"))
    resolved.update(
        {
            "config": str(args.config),
            "outputdir": str(outputdir),
            "cache_dir": str(cache_dir),
            "network_type_Gyro": network_type_gyro,
            "network_type_Acc": network_type_acc,
            "train_seqs": train_seqs,
            "val_seqs": val_seqs,
            "test_seqs": test_seqs,
            "epoch": int(args.epoch if args.epoch is not None else cfg.get("epoch", 1)),
            "device": args.device or cfg.get("device", "cpu"),
            "test_device": args.test_device or cfg.get("test_device", "cpu"),
        }
    )
    (outputdir / "resolved_config.json").write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (outputdir / "split_used.json").write_text(
        json.dumps(
            {
                "train": train_seqs,
                "val": val_seqs,
                "test": test_seqs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    argv = [sys.executable, str(PIPELINE)]
    add_option(argv, "--data-root", cfg.get("data_root"))
    add_option(argv, "--cache-dir", cache_dir)
    add_option(argv, "--outputdir", outputdir)
    add_option(argv, "--trainval-supervision-source", cfg.get("trainval_supervision_source", "mocap_200hz"))
    add_option(argv, "--pseudo-gt-root", cfg.get("pseudo_gt_root"))
    add_option(argv, "--vo-supervision-world-frame", cfg.get("vo_supervision_world_frame"))
    add_option(argv, "--vo-gap-tolerance-s", cfg.get("vo_gap_tolerance_s"))
    add_option(argv, "--source-position-sg-window", cfg.get("source_position_sg_window"))
    add_option(argv, "--source-position-sg-order", cfg.get("source_position_sg_order"))
    add_list_option(argv, "--gravity-accel-world", cfg.get("gravity_accel_world"))
    add_list_option(argv, "--train-seqs", train_seqs)
    add_list_option(argv, "--val-seqs", val_seqs)
    add_list_option(argv, "--test-seqs", test_seqs)
    add_option(argv, "--network_type_Gyro", network_type_gyro)
    add_option(argv, "--network_type_Acc", network_type_acc)
    add_option(argv, "--loss_window", cfg.get("window_size", cfg.get("loss_window", 64)))
    add_option(argv, "--batch-size", cfg.get("batch_size", 64))
    add_option(argv, "--epoch", resolved["epoch"])
    add_option(argv, "--dt", cfg.get("dt", 0.001))
    add_option(argv, "--percent_for_val", cfg.get("percent_for_val", 0.2))
    add_option(argv, "--train_batch_strategy", cfg.get("train_batch_strategy", "balanced_sequences"))
    add_option(argv, "--val_freq", cfg.get("val_freq", 10))
    add_option(argv, "--save_freq", cfg.get("save_freq", 10))
    add_option(argv, "--lr_Gyro", cfg.get("lr_Gyro"))
    add_option(argv, "--lr_Acc", cfg.get("lr_Acc"))
    add_option(argv, "--weight_decay_Gyro", cfg.get("weight_decay_Gyro"))
    add_option(argv, "--weight_decay_Acc", cfg.get("weight_decay_Acc"))
    add_option(argv, "--gyro_lr_step_size", cfg.get("gyro_lr_step_size"))
    add_option(argv, "--acc_lr_step_size", cfg.get("acc_lr_step_size"))
    add_option(argv, "--gyro_lr_gamma", cfg.get("gyro_lr_gamma"))
    add_option(argv, "--acc_lr_gamma", cfg.get("acc_lr_gamma"))
    add_option(argv, "--acc-loss-rot-weight", cfg.get("acc_loss_rot_weight"))
    add_option(argv, "--acc-loss-vel-weight", cfg.get("acc_loss_vel_weight"))
    add_option(argv, "--acc-loss-pos-weight", cfg.get("acc_loss_pos_weight"))
    add_option(argv, "--acc-loss-type", cfg.get("acc_loss_type"))
    add_option(argv, "--nll-covariance-strategy", cfg.get("nll_covariance_strategy"))
    add_option(argv, "--nll-sigma-a", cfg.get("nll_sigma_a"))
    add_option(argv, "--nll-sigma-g", cfg.get("nll_sigma_g"))
    add_option(argv, "--nll-w-rot", cfg.get("nll_w_rot"))
    add_option(argv, "--nll-w-vel", cfg.get("nll_w_vel"))
    add_option(argv, "--nll-w-pos", cfg.get("nll_w_pos"))
    add_option(argv, "--nll-q-block-epochs", cfg.get("nll_q_block_epochs"))
    add_option(argv, "--nll-lr-q", cfg.get("nll_lr_q"))
    add_option(argv, "--nll-sigma-a-min", cfg.get("nll_sigma_a_min"))
    add_option(argv, "--nll-sigma-a-max", cfg.get("nll_sigma_a_max"))
    add_option(argv, "--nll-sigma-g-min", cfg.get("nll_sigma_g_min"))
    add_option(argv, "--nll-sigma-g-max", cfg.get("nll_sigma_g_max"))
    add_option(argv, "--nll-supervision-mode", cfg.get("nll_supervision_mode"))
    add_option(argv, "--nll-mean-anchor-weight", cfg.get("nll_mean_anchor_weight"))
    add_option(argv, "--nll-mean-anchor-mode", cfg.get("nll_mean_anchor_mode"))
    add_option(argv, "--interval-nll-mode", cfg.get("interval_nll_mode"))
    add_option(argv, "--interval-sigma-g-nominal", cfg.get("interval_sigma_g_nominal"))
    add_option(argv, "--interval-sigma-a-nominal", cfg.get("interval_sigma_a_nominal"))
    add_option(argv, "--interval-sigma-g-floor2", cfg.get("interval_sigma_g_floor2"))
    add_option(argv, "--interval-sigma-a-floor2", cfg.get("interval_sigma_a_floor2"))
    add_option(argv, "--interval-target-cov-floor", cfg.get("interval_target_cov_floor"))
    add_option(argv, "--interval-jitter", cfg.get("interval_jitter"))
    add_option(argv, "--interval-prior-lambda-g", cfg.get("interval_prior_lambda_g"))
    add_option(argv, "--interval-prior-lambda-a", cfg.get("interval_prior_lambda_a"))
    add_option(argv, "--interval-lr-sigma", cfg.get("interval_lr_sigma"))
    add_option(argv, "--interval-covariance-gradient-mode", cfg.get("interval_covariance_gradient_mode"))
    add_option(argv, "--interval-noise-init-model-path", cfg.get("interval_noise_init_model_path"))
    add_option(argv, "--trajectory-supervision-mode", cfg.get("trajectory_supervision_mode"))
    add_option(argv, "--trajectory-supervision-stride", cfg.get("trajectory_supervision_stride"))
    add_option(argv, "--trajectory-max-supervised-states", cfg.get("trajectory_max_supervised_states"))
    add_option(argv, "--sg_window_bw", cfg.get("sg_window_bw"))
    add_option(argv, "--sg_order_bw", cfg.get("sg_order_bw"))
    add_option(argv, "--sg_window_ba", cfg.get("sg_window_ba"))
    add_option(argv, "--sg_order_ba", cfg.get("sg_order_ba"))
    add_option(argv, "--initial-bias-mode", cfg.get("initial_bias_mode"))
    add_option(argv, "--initial-bias-window-s", cfg.get("initial_bias_window_s"))
    add_option(argv, "--train-initial-bw-mode", cfg.get("train_initial_bw_mode"))
    add_option(argv, "--train-initial-ba-mode", cfg.get("train_initial_ba_mode"))
    add_option(argv, "--bw_model_path", cfg.get("bw_model_path"))
    add_option(argv, "--ba_model_path", cfg.get("ba_model_path"))
    add_option(argv, "--ba-init-model-path", cfg.get("ba_init_model_path"))
    add_option(argv, "--device", resolved["device"])
    add_option(argv, "--test-device", resolved["test_device"])
    add_bool_flag(argv, "--use-best-val", bool(cfg.get("use_best_val", True)))
    add_bool_flag(argv, "--nll-optimize-q", bool(cfg.get("nll_optimize_q", False)))
    add_bool_flag(argv, "--nll-train-gyro", bool(cfg.get("nll_train_gyro", False)))
    if "interval_detach_covariance_theta" in cfg:
        argv.append(
            "--interval-detach-covariance-theta"
            if cfg["interval_detach_covariance_theta"]
            else "--no-interval-detach-covariance-theta"
        )
    add_bool_flag(
        argv,
        "--interval-debug-covariance-gradients",
        bool(cfg.get("interval_debug_covariance_gradients", False)),
    )
    add_bool_flag(argv, "--skip-gyro-train", bool(cfg.get("skip_gyro_train", False)))
    add_bool_flag(argv, "--stop-after-gyro", bool(cfg.get("stop_after_gyro", False)))
    add_bool_flag(argv, "--skip-test", bool(cfg.get("skip_test", False)))
    add_bool_flag(argv, "--only-test", bool(cfg.get("only_test", False)))
    add_bool_flag(argv, "--recompute-cache", bool(cfg.get("recompute_cache", False) or args.recompute_cache))
    add_bool_flag(argv, "--test-recompute", bool(cfg.get("test_recompute", False) or args.test_recompute))
    if "readable_val_metrics" in cfg:
        argv.append("--readable-val-metrics" if cfg["readable_val_metrics"] else "--no-readable-val-metrics")

    command_text = " ".join(str(item) for item in argv)
    (outputdir / "command.txt").write_text(command_text + "\n", encoding="utf-8")
    print(command_text, flush=True)
    if args.dry_run:
        return
    subprocess.run(argv, check=True)


if __name__ == "__main__":
    main()
