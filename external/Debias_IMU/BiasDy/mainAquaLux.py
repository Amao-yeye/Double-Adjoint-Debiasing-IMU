#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

BIASDY_DIR = os.path.abspath(os.path.dirname(__file__))
DEBIAS_ROOT = os.path.abspath(os.path.join(BIASDY_DIR, ".."))
if BIASDY_DIR not in sys.path:
    sys.path.insert(0, BIASDY_DIR)
if DEBIAS_ROOT not in sys.path:
    sys.path.insert(0, DEBIAS_ROOT)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import lie_algebra as Lie
from aqualux_dataset import (
    DEFAULT_DATA_DIR,
    DEFAULT_GRAVITY_NORM,
    AquaLuxDataset,
    analyze_timing,
    estimate_native_dt,
    sorted_aqualux_sequences,
    write_timing_csv,
)
from dataset import pload
from learning import Acc_train, Gyro_train, Test
from network_for_BiadDy import ba_func_choose, bw_func_choose


DEFAULT_OUTPUT_DIR = "Results/AquaLux/preintegration/bias_dynamic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Learning AquaLux Bias Dynamics")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--outputdir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--integral_method", type=str, default="euler")
    parser.add_argument("--network_type_Gyro", type=str, default="bw_func_net")
    parser.add_argument("--network_type_Acc", type=str, default="ba_func_net")
    parser.add_argument("--loss_window", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--epoch", type=int, default=1801)
    parser.add_argument("--lr_Gyro", type=float, default=5e-3)
    parser.add_argument("--weight_decay_Gyro", type=float, default=1e-6)
    parser.add_argument("--lr_Acc", type=float, default=5e-3)
    parser.add_argument("--weight_decay_Acc", type=float, default=1e-6)
    parser.add_argument("--dt", type=float, default=0.0, help="Fixed native-rate dt. <=0 estimates median IMU dt.")
    parser.add_argument("--percent-for-val", type=float, default=0.2)
    parser.add_argument("--gravity-norm", type=float, default=DEFAULT_GRAVITY_NORM)
    parser.add_argument("--gt-valid-max-dt", type=float, default=0.02)
    parser.add_argument("--velocity-smooth-sec", type=float, default=0.12)
    parser.add_argument("--sg-window-bw", type=int, default=1)
    parser.add_argument("--sg-order-bw", type=int, default=0)
    parser.add_argument("--sg-window-ba", type=int, default=1)
    parser.add_argument("--sg-order-ba", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--test-device", type=str, default="cpu")
    parser.add_argument("--train-seq", action="append", default=None)
    parser.add_argument("--test-seq", action="append", default=None)
    parser.add_argument("--recompute-cache", action="store_true")
    parser.add_argument("--test-recompute", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--skip-gyro-train", action="store_true")
    parser.add_argument("--skip-acc-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--gyro-checkpoint", type=str, default=None)
    parser.add_argument("--acc-checkpoint", type=str, default=None)
    parser.add_argument("--bias-init-mode", choices=["first", "mean"], default="first")
    parser.add_argument("--bias-init-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputdir = Path(args.outputdir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else outputdir / "cache"
    outputdir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_seqs = args.train_seq or sorted_aqualux_sequences(args.data_dir, "Train_")
    test_seqs = args.test_seq or sorted_aqualux_sequences(args.data_dir, "Test_")
    val_seqs = list(train_seqs)
    all_seqs = sorted(set(train_seqs + test_seqs), key=lambda name: (name.split("_", 1)[0], int(name.split("_", 1)[1])))

    timing_rows = analyze_timing(args.data_dir, all_seqs)
    write_timing_csv(outputdir / "timing_report.csv", timing_rows)
    assert_no_time_backjump_or_duplicates(timing_rows)

    dt = float(args.dt) if args.dt > 0 else estimate_native_dt(args.data_dir, all_seqs)
    print(f"AquaLux native dt: {dt:.12g} s ({1.0 / dt:.3f} Hz)")
    print(f"Timing report: {outputdir / 'timing_report.csv'}")

    device = choose_device(args.device)
    test_device = choose_device(args.test_device)
    config_path = outputdir / "run_config.json"
    write_run_config(config_path, args, train_seqs, val_seqs, test_seqs, dt, device, test_device)

    dataset_parameters = {
        "dataset_name": "AquaLux",
        "data_dir": args.data_dir,
        "data_cache_dir": str(cache_dir),
        "train_seqs": train_seqs,
        "val_seqs": val_seqs,
        "test_seqs": test_seqs,
        "dt": dt,
        "percent_for_val": args.percent_for_val,
        "loss_window": args.loss_window,
        "batch_size": args.batch_size,
        "sg_window_bw": args.sg_window_bw,
        "sg_order_bw": args.sg_order_bw,
        "sg_window_ba": args.sg_window_ba,
        "sg_order_ba": args.sg_order_ba,
        "gravity_norm": args.gravity_norm,
        "gt_valid_max_dt": args.gt_valid_max_dt,
        "velocity_smooth_sec": args.velocity_smooth_sec,
    }
    dataset_train = AquaLuxDataset(**dataset_parameters, mode="train", recompute=args.recompute_cache)
    dataset_val = AquaLuxDataset(**dataset_parameters, mode="val", recompute=False)
    dataset_test = AquaLuxDataset(**dataset_parameters, mode="test", recompute=False)

    if args.cache_only:
        print("Cache-only run complete.")
        return

    t_start = time.time()
    gyro_checkpoint = args.gyro_checkpoint
    if not args.skip_gyro_train and gyro_checkpoint is None:
        Gyro_train(
            dataset_train,
            dataset_val,
            str(outputdir),
            args.network_type_Gyro,
            args.integral_method,
            device,
            args.lr_Gyro,
            args.weight_decay_Gyro,
            epoch=args.epoch,
        )
        gyro_checkpoint = find_checkpoint(outputdir / "Gyro_weights", args.epoch)
    elif gyro_checkpoint is None:
        gyro_checkpoint = find_checkpoint(outputdir / "Gyro_weights", args.epoch)

    acc_checkpoint = args.acc_checkpoint
    if not args.skip_acc_train and acc_checkpoint is None:
        bw_func = bw_func_choose(args.network_type_Gyro, device=device)
        bw_func.load_state_dict(load_checkpoint(gyro_checkpoint, device)["func_bw_model"])
        gpu_memory = Acc_train(
            dataset_train,
            dataset_val,
            str(outputdir),
            bw_func,
            args.network_type_Acc,
            args.integral_method,
            device,
            args.lr_Acc,
            args.weight_decay_Acc,
            epoch=args.epoch,
        )
        acc_checkpoint = find_checkpoint(outputdir / "Acc_weights", args.epoch)
    elif acc_checkpoint is None:
        acc_checkpoint = find_checkpoint(outputdir / "Acc_weights", args.epoch)
        gpu_memory = []
    else:
        gpu_memory = []

    t_end = time.time()
    write_time_memory(outputdir / "time_memory_record.txt", t_end - t_start, gpu_memory)

    if not args.skip_test:
        bw_func = bw_func_choose(args.network_type_Gyro, device=test_device)
        bw_func.load_state_dict(load_checkpoint(gyro_checkpoint, test_device)["func_bw_model"])
        ba_func = ba_func_choose(args.network_type_Acc, device=test_device)
        ba_func.load_state_dict(load_checkpoint(acc_checkpoint, test_device)["func_ba_model"])
        with torch.no_grad():
            Test(
                dataset_test,
                str(outputdir),
                bw_func,
                ba_func,
                args.integral_method,
                device=test_device,
                recompute=args.test_recompute,
                bias_init_mode=args.bias_init_mode,
                bias_init_seconds=args.bias_init_seconds,
            )
        rows = export_preintegration_results(dataset_test, outputdir)
        write_metrics_csv(outputdir / "metrics.csv", rows)
        aggregate = aggregate_rows(rows)
        write_method_summary(outputdir / "summary.md", rows, aggregate, args, dt)
        write_global_preintegration_summary(outputdir.parent / "summary.md", outputdir, rows, aggregate, args, dt)
        print(
            "bias_dynamic aggregate: "
            f"position_rmse_m={aggregate['position_rmse_m']:.6f}, "
            f"rotation_rmse_deg={aggregate['rotation_rmse_deg']:.6f}"
        )


def choose_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is not available")
    return device


def assert_no_time_backjump_or_duplicates(rows: list[dict[str, float | int | str]]) -> None:
    bad = [
        row["sequence"]
        for row in rows
        if int(row["imu_duplicate_dt"]) or int(row["imu_time_backjump"]) or int(row["gt_duplicate_dt"]) or int(row["gt_time_backjump"])
    ]
    if bad:
        raise RuntimeError(f"AquaLux time audit failed for sequences: {bad}")


def write_run_config(
    path: Path,
    args: argparse.Namespace,
    train_seqs: list[str],
    val_seqs: list[str],
    test_seqs: list[str],
    dt: float,
    device: str,
    test_device: str,
) -> None:
    payload = vars(args).copy()
    payload.update(
        {
            "train_seqs": train_seqs,
            "val_seqs": val_seqs,
            "test_seqs": test_seqs,
            "effective_dt": dt,
            "effective_rate_hz": 1.0 / dt,
            "device": device,
            "test_device": test_device,
            "method": "bias_dynamic",
            "axb_used": False,
        }
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_checkpoint(path: str | os.PathLike, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def find_checkpoint(weights_dir: Path, epoch_count: int) -> str:
    preferred = weights_dir / f"model_epoch_{max(0, epoch_count - 1)}.pt"
    if preferred.exists():
        return str(preferred)
    checkpoints = sorted(weights_dir.glob("model_epoch_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if checkpoints:
        return str(checkpoints[-1])
    best = weights_dir / "best_val_model.pt"
    if best.exists():
        return str(best)
    raise FileNotFoundError(f"No checkpoint found in {weights_dir}")


def write_time_memory(path: Path, seconds: float, gpu_memory: list[float]) -> None:
    ave_gpu_m = float(np.mean(gpu_memory)) if gpu_memory else 0.0
    path.write_text(
        f"Time for training: {seconds}\n"
        f"Average max GPU memory allocated: {ave_gpu_m:.2f} MB\n",
        encoding="utf-8",
    )


def export_preintegration_results(dataset_test: AquaLuxDataset, outputdir: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for index, sequence in enumerate(dataset_test.sequences):
        result_path = outputdir / "Sequnces-results" / sequence / "FullTra_result.p"
        result = pload(result_path)
        seq_dir = outputdir / sequence
        seq_dir.mkdir(parents=True, exist_ok=True)

        t = to_numpy(result["t_gt"]).astype(np.float64)
        timestamp_start = float(result.get("timestamp_start", dataset_test.get_start_timestamp(index)))
        stamps = np.float64(timestamp_start) + t
        x_gt = result["X_gt"].detach().cpu()
        r_pred = result["R_pred"].detach().cpu()
        sol_pred = result["sol_pred"].detach().cpu()
        p_gt = x_gt[:, :3, 4]
        r_gt = x_gt[:, :3, :3]
        p_pred = sol_pred[:, 10:13]
        net_imu = result["net_imu"].detach().cpu()

        imudict = dataset_test.load_imu(index)
        gt_nearest_dt = to_numpy(imudict.get("gt_nearest_dt", torch.zeros_like(result["t_gt"])))
        valid = gt_nearest_dt <= dataset_test.gt_valid_max_dt
        n = min(len(valid), len(stamps))
        valid = valid[:n]
        stamps = stamps[:n]
        gt_nearest_dt = gt_nearest_dt[:n]
        p_gt = p_gt[:n]
        r_gt = r_gt[:n]
        p_pred = p_pred[:n]
        r_pred = r_pred[:n]
        net_imu = net_imu[:n]

        pos_err = torch.linalg.norm(p_pred - p_gt, dim=1).numpy()
        rot_err = (
            torch.linalg.norm(Lie.SO3log(r_gt @ r_pred.transpose(-1, -2)), dim=1) * 180.0 / torch.pi
        ).numpy()

        gt_path = seq_dir / "gt_resampled.tum"
        pred_path = seq_dir / "bias_dynamic_preintegrated.tum"
        errors_path = seq_dir / "errors.csv"
        imu_path = seq_dir / "corrected_imu.csv"
        write_tum(gt_path, stamps, p_gt.numpy(), r_gt.numpy())
        write_tum(pred_path, stamps, p_pred.numpy(), r_pred.numpy())
        write_corrected_imu_csv(imu_path, stamps, net_imu.numpy())
        write_errors_csv(errors_path, stamps, valid, gt_nearest_dt, pos_err, rot_err)

        if int(np.sum(valid)) < 10:
            raise RuntimeError(f"{sequence}: too few valid GT metric samples")
        rows.append(
            {
                "sequence": sequence,
                "samples": int(n),
                "valid_metric_samples": int(np.sum(valid)),
                "duration_sec": float(stamps[-1] - stamps[0]),
                "gt_interp_max_dt_sec": float(np.max(gt_nearest_dt)),
                "gt_valid_max_dt_sec": float(dataset_test.gt_valid_max_dt),
                "position_rmse_m": rmse(pos_err[valid]),
                "rotation_rmse_deg": rmse(rot_err[valid]),
                "position_mean_m": float(np.mean(pos_err[valid])),
                "rotation_mean_deg": float(np.mean(rot_err[valid])),
                "position_max_m": float(np.max(pos_err[valid])),
                "rotation_max_deg": float(np.max(rot_err[valid])),
                "gt_resampled": str(gt_path),
                "bias_dynamic_preintegrated": str(pred_path),
                "corrected_imu": str(imu_path),
                "errors_csv": str(errors_path),
            }
        )
    return rows


def to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def rmse(values: np.ndarray) -> float:
    if len(values) == 0:
        return math.nan
    return float(np.sqrt(np.mean(values * values)))


def write_tum(path: Path, stamps: np.ndarray, positions: np.ndarray, rotations: np.ndarray) -> None:
    quats = Rotation.from_matrix(rotations).as_quat()
    with path.open("w", encoding="utf-8") as f:
        for stamp, pos, quat in zip(stamps, positions, quats):
            f.write(
                f"{stamp:.9f} "
                f"{pos[0]:.12g} {pos[1]:.12g} {pos[2]:.12g} "
                f"{quat[0]:.12g} {quat[1]:.12g} {quat[2]:.12g} {quat[3]:.12g}\n"
            )


def write_corrected_imu_csv(path: Path, stamps: np.ndarray, imu: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "wx", "wy", "wz", "ax", "ay", "az"])
        for stamp, row in zip(stamps, imu):
            writer.writerow([f"{stamp:.9f}"] + [f"{value:.12g}" for value in row])


def write_errors_csv(
    path: Path,
    stamps: np.ndarray,
    valid: np.ndarray,
    gt_nearest_dt: np.ndarray,
    pos_err: np.ndarray,
    rot_err: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "valid_metric_sample", "gt_nearest_dt_sec", "position_error_m", "rotation_error_deg"])
        for row in zip(stamps, valid, gt_nearest_dt, pos_err, rot_err):
            writer.writerow([f"{row[0]:.9f}", int(row[1]), f"{row[2]:.12g}", f"{row[3]:.12g}", f"{row[4]:.12g}"])


def write_metrics_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fields = [
        "sequence",
        "samples",
        "valid_metric_samples",
        "duration_sec",
        "gt_interp_max_dt_sec",
        "gt_valid_max_dt_sec",
        "position_rmse_m",
        "rotation_rmse_deg",
        "position_mean_m",
        "rotation_mean_deg",
        "position_max_m",
        "rotation_max_deg",
        "gt_resampled",
        "bias_dynamic_preintegrated",
        "corrected_imu",
        "errors_csv",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, float | int | str]]) -> dict[str, float | int]:
    pos_parts: list[np.ndarray] = []
    rot_parts: list[np.ndarray] = []
    for row in rows:
        data = np.genfromtxt(row["errors_csv"], delimiter=",", names=True, dtype=float)
        if data.ndim == 0:
            data = data.reshape(1)
        valid = data["valid_metric_sample"] > 0.5
        pos_parts.append(np.asarray(data["position_error_m"][valid], dtype=float))
        rot_parts.append(np.asarray(data["rotation_error_deg"][valid], dtype=float))
    pos = np.concatenate(pos_parts)
    rot = np.concatenate(rot_parts)
    return {
        "valid_metric_samples": int(len(pos)),
        "position_rmse_m": rmse(pos),
        "rotation_rmse_deg": rmse(rot),
        "position_mean_m": float(np.mean(pos)),
        "rotation_mean_deg": float(np.mean(rot)),
        "position_max_m": float(np.max(pos)),
        "rotation_max_deg": float(np.max(rot)),
    }


def write_method_summary(
    path: Path,
    rows: list[dict[str, float | int | str]],
    aggregate: dict[str, float | int],
    args: argparse.Namespace,
    dt: float,
) -> None:
    mean_pos = float(np.mean([float(row["position_rmse_m"]) for row in rows]))
    mean_rot = float(np.mean([float(row["rotation_rmse_deg"]) for row in rows]))
    lines = [
        "# AquaLux Bias Dynamic Preintegration Summary",
        "",
        "## Scope",
        "",
        "- Method: `bias_dynamic`, the original BiasDy learned bias dynamics model.",
        "- Axb linear baseline is not used in this run.",
        "- Split: Train sequences use first 80 percent for training and last 20 percent for validation; Test sequences are evaluation only.",
        f"- Loss window: `{args.loss_window}` samples.",
        f"- Effective IMU rate: `{1.0 / dt:.3f} Hz` (`dt={dt:.12g} s`).",
        f"- Gravity: `[0, 0, -{args.gravity_norm:g}] m/s^2`.",
        f"- Valid metric samples require nearest source GT pose within `{args.gt_valid_max_dt:g} s`.",
        "",
        "## Aggregate",
        "",
        "| Method | Sequences | Valid samples | Position RMSE (m) | Rotation RMSE (deg) | Mean seq pos RMSE (m) | Mean seq rot RMSE (deg) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| `bias_dynamic` | {len(rows)} | {aggregate['valid_metric_samples']} | "
            f"{aggregate['position_rmse_m']:.6f} | {aggregate['rotation_rmse_deg']:.6f} | "
            f"{mean_pos:.6f} | {mean_rot:.6f} |"
        ),
        "",
        "## Per-Sequence Results",
        "",
        "| Sequence | Samples | Valid samples | Duration (s) | GT max nearest dt (s) | Position RMSE (m) | Rotation RMSE (deg) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['sequence']}` | {row['samples']} | {row['valid_metric_samples']} | "
            f"{float(row['duration_sec']):.3f} | {float(row['gt_interp_max_dt_sec']):.6f} | "
            f"{float(row['position_rmse_m']):.6f} | {float(row['rotation_rmse_deg']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Output directory: `{path.parent}`.",
            "- Per-sequence files:",
            "  - `gt_resampled.tum`",
            "  - `bias_dynamic_preintegrated.tum`",
            "  - `corrected_imu.csv`",
            "  - `errors.csv`",
            "- Machine-readable metrics: `metrics.csv`.",
            "- Timing audit: `timing_report.csv`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_global_preintegration_summary(
    path: Path,
    outputdir: Path,
    rows: list[dict[str, float | int | str]],
    aggregate: dict[str, float | int],
    args: argparse.Namespace,
    dt: float,
) -> None:
    method_rows = read_existing_raw_row(outputdir.parent)
    mean_pos = float(np.mean([float(row["position_rmse_m"]) for row in rows]))
    mean_rot = float(np.mean([float(row["rotation_rmse_deg"]) for row in rows]))
    method_rows.append(
        {
            "method": "bias_dynamic",
            "rate_hz": 1.0 / dt,
            "sequences": len(rows),
            "valid_samples": aggregate["valid_metric_samples"],
            "position_rmse_m": aggregate["position_rmse_m"],
            "rotation_rmse_deg": aggregate["rotation_rmse_deg"],
            "mean_seq_pos_rmse_m": mean_pos,
            "mean_seq_rot_rmse_deg": mean_rot,
        }
    )
    lines = [
        "# AquaLux Preintegration Summary",
        "",
        "## Scope",
        "",
        "- Split: Test sequences only for reported metrics.",
        "- `bias_dynamic` uses the original BiasDy learned bias dynamics model trained with GT supervision.",
        "- Axb linear baseline is intentionally not included in Step 8.",
        f"- Bias Dynamic effective IMU rate: `{1.0 / dt:.3f} Hz`.",
        f"- Bias Dynamic gravity: `[0, 0, -{args.gravity_norm:g}] m/s^2`.",
        "",
        "## Method Comparison",
        "",
        "| Method | Rate (Hz) | Sequences | Valid samples | Position RMSE (m) | Rotation RMSE (deg) | Mean seq pos RMSE (m) | Mean seq rot RMSE (deg) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in method_rows:
        lines.append(
            f"| `{row['method']}` | {float(row['rate_hz']):.3f} | {row['sequences']} | {row['valid_samples']} | "
            f"{float(row['position_rmse_m']):.6f} | {float(row['rotation_rmse_deg']):.6f} | "
            f"{float(row['mean_seq_pos_rmse_m']):.6f} | {float(row['mean_seq_rot_rmse_deg']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- Raw method directory: `Results/AquaLux/preintegration/raw`.",
            "- Bias Dynamic method directory: `Results/AquaLux/preintegration/bias_dynamic`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_existing_raw_row(preintegration_root: Path) -> list[dict[str, float | int | str]]:
    raw_metrics = preintegration_root / "raw" / "metrics.csv"
    if not raw_metrics.exists():
        return []
    with raw_metrics.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    raw_aggregate = aggregate_rows(rows)
    mean_pos = float(np.mean([float(row["position_rmse_m"]) for row in rows]))
    mean_rot = float(np.mean([float(row["rotation_rmse_deg"]) for row in rows]))
    rates = [float(row["samples"]) / float(row["duration_sec"]) for row in rows if float(row["duration_sec"]) > 0]
    return [
        {
            "method": "raw_imu",
            "rate_hz": float(np.mean(rates)) if rates else math.nan,
            "sequences": len(rows),
            "valid_samples": raw_aggregate["valid_metric_samples"],
            "position_rmse_m": raw_aggregate["position_rmse_m"],
            "rotation_rmse_deg": raw_aggregate["rotation_rmse_deg"],
            "mean_seq_pos_rmse_m": mean_pos,
            "mean_seq_rot_rmse_deg": mean_rot,
        }
    ]


if __name__ == "__main__":
    main()
