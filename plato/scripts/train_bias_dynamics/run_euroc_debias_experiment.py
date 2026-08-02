#!/usr/bin/env python3
"""Run a config-driven EuRoC Debias_IMU experiment through the PLATO runner."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


PLATO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE = PLATO_ROOT / "plato" / "scripts" / "train_bias_dynamics" / "run_euroc_debias_phys_delta_pipeline.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--test-device", type=str, default=None)
    parser.add_argument("--outputdir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--recompute-cache", action="store_true")
    parser.add_argument("--test-recompute", action="store_true")
    parser.add_argument("--only-test", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def add_option(argv: list[str], name: str, value) -> None:
    if value is not None:
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
    if cfg.get("dataset") != "euroc":
        raise ValueError("Expected dataset: euroc")

    outputdir = args.outputdir or Path(cfg["outputdir"])
    cache_dir = args.cache_dir or Path(cfg["cache_dir"])
    epoch = int(args.epoch if args.epoch is not None else cfg.get("epoch", 1801))
    device = args.device or cfg.get("device", "cuda")
    test_device = args.test_device or cfg.get("test_device", "cpu")
    outputdir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    resolved = dict(cfg)
    resolved.update(
        {
            "config": str(args.config),
            "outputdir": str(outputdir),
            "cache_dir": str(cache_dir),
            "epoch": epoch,
            "device": device,
            "test_device": test_device,
        }
    )
    (outputdir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    argv = [sys.executable, str(PIPELINE)]
    add_option(argv, "--data-root", cfg.get("data_root"))
    add_option(argv, "--cache-dir", cache_dir)
    add_option(argv, "--test-cache-dir", cfg.get("test_cache_dir"))
    add_option(argv, "--outputdir", outputdir)
    add_option(argv, "--train-supervision", cfg.get("train_supervision", "gt"))
    add_option(argv, "--trainval-gt-root", cfg.get("trainval_gt_root"))
    add_option(argv, "--trainval-gt-source", cfg.get("trainval_gt_source"))
    add_option(argv, "--supervision-timing", cfg.get("supervision_timing", "imu"))
    add_list_option(argv, "--exclude-train-seq", cfg.get("exclude_train_seq"))
    add_list_option(argv, "--train-seqs", cfg.get("train_seqs"))
    add_list_option(argv, "--val-seqs", cfg.get("val_seqs"))
    add_list_option(argv, "--test-seqs", cfg.get("test_seqs"))
    add_option(argv, "--network_type_Gyro", cfg.get("network_type_Gyro", "bw_func_net_no_udot"))
    add_option(argv, "--network_type_Acc", cfg.get("network_type_Acc", "ba_func_net_no_udot"))
    add_option(argv, "--loss_window", cfg.get("loss_window", cfg.get("window_size", 64)))
    add_option(argv, "--batch-size", cfg.get("batch_size", 1000))
    add_option(argv, "--epoch", epoch)
    add_option(argv, "--lr_Gyro", cfg.get("lr_Gyro"))
    add_option(argv, "--lr_Acc", cfg.get("lr_Acc"))
    add_option(argv, "--weight_decay_Gyro", cfg.get("weight_decay_Gyro"))
    add_option(argv, "--weight_decay_Acc", cfg.get("weight_decay_Acc"))
    add_option(argv, "--gyro_lr_step_size", cfg.get("gyro_lr_step_size"))
    add_option(argv, "--acc_lr_step_size", cfg.get("acc_lr_step_size"))
    add_option(argv, "--gyro_lr_gamma", cfg.get("gyro_lr_gamma"))
    add_option(argv, "--acc_lr_gamma", cfg.get("acc_lr_gamma"))
    add_option(argv, "--gyro-loss", cfg.get("gyro_loss", "ad"))
    add_option(argv, "--acc-loss", cfg.get("acc_loss", "se23"))
    add_option(argv, "--acc-loss-type", cfg.get("acc_loss_type"))
    add_option(argv, "--ba-init-model-path", cfg.get("ba_init_model_path"))
    add_option(argv, "--acc-loss-rot-weight", cfg.get("acc_loss_rot_weight"))
    add_option(argv, "--acc-loss-vel-weight", cfg.get("acc_loss_vel_weight"))
    add_option(argv, "--acc-loss-pos-weight", cfg.get("acc_loss_pos_weight"))
    add_option(argv, "--gyro-bias-rate-regularization-weight", cfg.get("gyro_bias_rate_regularization_weight"))
    add_option(argv, "--acc-bias-rate-regularization-weight", cfg.get("acc_bias_rate_regularization_weight"))
    add_option(argv, "--bw-init-model-path", cfg.get("bw_init_model_path"))
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
    add_option(argv, "--analytic_grad_sign", cfg.get("analytic_grad_sign"))
    add_option(argv, "--analytic_grad_scale", cfg.get("analytic_grad_scale"))
    add_option(argv, "--analytic_grad_clip_norm", cfg.get("analytic_grad_clip_norm"))
    add_option(argv, "--analytic_residual_jacobian", cfg.get("analytic_residual_jacobian"))
    add_option(argv, "--analytic_invariant_error", cfg.get("analytic_invariant_error"))
    add_option(argv, "--gyro_analytic_grad_sign", cfg.get("gyro_analytic_grad_sign"))
    add_option(argv, "--gyro_analytic_grad_scale", cfg.get("gyro_analytic_grad_scale"))
    add_option(argv, "--gyro_analytic_grad_clip_norm", cfg.get("gyro_analytic_grad_clip_norm"))
    add_option(argv, "--gyro_analytic_residual_jacobian", cfg.get("gyro_analytic_residual_jacobian"))
    add_option(argv, "--gyro_analytic_invariant_error", cfg.get("gyro_analytic_invariant_error"))
    add_option(argv, "--inner_vjp", cfg.get("inner_vjp"))
    add_option(argv, "--val_freq", cfg.get("val_freq", 100))
    add_option(argv, "--save_freq", cfg.get("save_freq", 100))
    add_option(argv, "--train_batch_strategy", cfg.get("train_batch_strategy", "random_sequence"))
    add_option(argv, "--percent_for_val", cfg.get("percent_for_val"))
    add_option(argv, "--sg_window_bw", cfg.get("sg_window_bw"))
    add_option(argv, "--sg_order_bw", cfg.get("sg_order_bw"))
    add_option(argv, "--sg_window_ba", cfg.get("sg_window_ba"))
    add_option(argv, "--sg_order_ba", cfg.get("sg_order_ba"))
    add_option(argv, "--dt", cfg.get("dt"))
    add_option(argv, "--device", device)
    add_option(argv, "--test-device", test_device)
    add_option(argv, "--initial-bias-mode", cfg.get("initial_bias_mode"))
    add_option(argv, "--initial-bias-window-s", cfg.get("initial_bias_window_s"))
    add_option(argv, "--bw_model_path", cfg.get("bw_model_path"))
    add_option(argv, "--ba_model_path", cfg.get("ba_model_path"))
    add_list_option(argv, "--full-rollout-val-seqs", cfg.get("full_rollout_val_seqs"))
    add_option(argv, "--full-rollout-selection-cache-dir", cfg.get("full_rollout_selection_cache_dir"))
    add_option(argv, "--full-rollout-selection-device", cfg.get("full_rollout_selection_device"))
    add_option(argv, "--full-rollout-score-aoe-weight", cfg.get("full_rollout_score_aoe_weight"))
    add_option(argv, "--full-rollout-score-vel-weight", cfg.get("full_rollout_score_vel_weight"))
    add_option(argv, "--full-rollout-score-pos-weight", cfg.get("full_rollout_score_pos_weight"))
    add_option(argv, "--full-rollout-score-aoe-scale", cfg.get("full_rollout_score_aoe_scale"))
    add_option(argv, "--full-rollout-score-vel-scale", cfg.get("full_rollout_score_vel_scale"))
    add_option(argv, "--full-rollout-score-pos-scale", cfg.get("full_rollout_score_pos_scale"))
    add_bool_flag(argv, "--use-best-val", bool(cfg.get("use_best_val", False)))
    add_bool_flag(argv, "--full-rollout-select", bool(cfg.get("full_rollout_select", False)))
    add_bool_flag(argv, "--full-rollout-select-only", bool(cfg.get("full_rollout_select_only", False)))
    add_bool_flag(argv, "--full-rollout-select-gyro", bool(cfg.get("full_rollout_select_gyro", False)))
    add_bool_flag(argv, "--only-test", bool(cfg.get("only_test", False) or args.only_test))
    add_bool_flag(argv, "--skip-gyro-train", bool(cfg.get("skip_gyro_train", False)))
    add_bool_flag(argv, "--nll-train-gyro", bool(cfg.get("nll_train_gyro", False)))
    add_bool_flag(argv, "--interval-debug-covariance-gradients", bool(cfg.get("interval_debug_covariance_gradients", False)))
    add_bool_flag(argv, "--stop-after-gyro", bool(cfg.get("stop_after_gyro", False)))
    add_bool_flag(argv, "--skip-test", bool(cfg.get("skip_test", False)))
    add_bool_flag(argv, "--skip-plots", bool(cfg.get("skip_plots", False) or args.skip_plots))
    add_bool_flag(argv, "--recompute-cache", bool(cfg.get("recompute_cache", False) or args.recompute_cache))
    add_bool_flag(argv, "--test-recompute", bool(cfg.get("test_recompute", False) or args.test_recompute))
    if "readable_val_metrics" in cfg:
        argv.append("--readable-val-metrics" if cfg["readable_val_metrics"] else "--no-readable-val-metrics")
    if "interval_detach_covariance_theta" in cfg:
        argv.append(
            "--interval-detach-covariance-theta"
            if cfg["interval_detach_covariance_theta"]
            else "--no-interval-detach-covariance-theta"
        )

    command_text = " ".join(str(item) for item in argv)
    (outputdir / "command.txt").write_text(command_text + "\n", encoding="utf-8")
    print(command_text, flush=True)
    if args.dry_run:
        return
    subprocess.run(argv, check=True)


if __name__ == "__main__":
    main()
