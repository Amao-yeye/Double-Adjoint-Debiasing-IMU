#!/usr/bin/env python3
"""Generate AquaLux hybrid IMU for segment-only baseline compensation.

Light-on intervals keep raw IMU unchanged. Light-off intervals are detected
from the raw OpenVINS timing log, seeded with the last raw OpenVINS bg/ba, and
compensated by the BiasDy bias dynamics model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from evaluate_aqualux_openvins_compensate1 import (  # noqa: E402
    DEFAULT_BASELINE_ROOT,
    DEFAULT_RAW_ROOT,
    TEST_SEQUENCES,
    bias_derivative,
    choose_device,
    load_models,
    read_openvins_estimate,
    read_weak_visual_segments,
)
from run_aqualux_openvins_raw import DEFAULT_DATA_ROOT  # noqa: E402

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Results" / "AquaLux" / "openvins_test" / "baseline" / "segment_only" / "hybrid_imu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--raw-openvins-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--gyro-checkpoint", type=Path, default=None)
    parser.add_argument("--acc-checkpoint", type=Path, default=None)
    parser.add_argument("--sequences", nargs="+", default=list(TEST_SEQUENCES))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--msckf-update-threshold", type=float, default=5e-4)
    parser.add_argument("--min-weak-duration", type=float, default=0.5)
    parser.add_argument("--merge-gap", type=float, default=0.2)
    parser.add_argument("--correction-mode", choices=["absolute", "delta"], default="delta")
    parser.add_argument("--components", choices=["both", "gyro", "acc"], default="both")
    parser.add_argument("--gyro-correction-scale", type=float, default=1.0)
    parser.add_argument("--acc-correction-scale", type=float, default=1.0)
    parser.add_argument(
        "--gate-max-position-cov-trace",
        type=float,
        default=0.0,
        help="If >0, compensate a sequence only when raw OpenVINS max Pt trace exceeds this threshold.",
    )
    return parser.parse_args()


def read_imu_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if rows.ndim == 0:
        rows = rows.reshape(1)
    required = ("timestamp", "wx", "wy", "wz", "ax", "ay", "az")
    if rows.dtype.names is None or any(name not in rows.dtype.names for name in required):
        raise ValueError(f"{path}: missing required IMU columns")
    stamps = np.asarray(rows["timestamp"], dtype=np.float64)
    imu = np.column_stack([rows[name] for name in required[1:]]).astype(np.float64)
    order = np.argsort(stamps)
    stamps = stamps[order]
    imu = imu[order]
    keep = np.r_[True, np.diff(stamps) > 0]
    return stamps[keep], imu[keep]


def write_imu_csv(path: Path, stamps: np.ndarray, imu: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "wx", "wy", "wz", "ax", "ay", "az"])
        for stamp, row in zip(stamps, imu):
            writer.writerow([f"{stamp:.9f}", *[f"{value:.12g}" for value in row]])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def raw_max_position_cov_trace(odom_path: Path) -> float:
    max_trace = 0.0
    if not odom_path.exists():
        return max_trace
    with odom_path.open(errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 20:
                continue
            try:
                values = [float(x) for x in parts[:20]]
            except ValueError:
                continue
            pt_trace = values[14] + values[17] + values[19]
            max_trace = max(max_trace, pt_trace)
    return float(max_trace)


def compensate_segment(
    stamps: np.ndarray,
    raw_imu: np.ndarray,
    hybrid_imu: np.ndarray,
    mask: np.ndarray,
    bg0: np.ndarray,
    ba0: np.ndarray,
    bw_model,
    ba_model,
    device: str,
    correction_mode: str,
    components: str,
    gyro_correction_scale: float,
    acc_correction_scale: float,
) -> dict[str, float | int]:
    idx = np.where(mask)[0]
    if len(idx) < 2:
        return {"samples": int(len(idx)), "status": "skipped_too_few_samples"}

    local_t = stamps[idx]
    local_imu = raw_imu[idx]
    local_dot = np.gradient(local_imu, local_t, axis=0, edge_order=1)
    bg = bg0.astype(np.float64).copy()
    ba = ba0.astype(np.float64).copy()
    bias = np.zeros((len(idx), 6), dtype=np.float64)

    for local_i in range(len(idx)):
        bias[local_i, :3] = bg
        bias[local_i, 3:] = ba
        if local_i + 1 >= len(idx):
            break
        y = np.zeros(16, dtype=np.float64)
        y[3] = local_t[local_i] - local_t[0]
        y[4:7] = bg
        y[13:16] = ba
        bg_dot, ba_dot = bias_derivative(bw_model, ba_model, y, local_imu[local_i], local_dot[local_i], device)
        dt = float(local_t[local_i + 1] - local_t[local_i])
        if dt <= 0:
            continue
        bg = bg + bg_dot * dt
        ba = ba + ba_dot * dt

    if correction_mode == "absolute":
        correction = bias
    elif correction_mode == "delta":
        seed_bias = np.r_[bg0, ba0][None, :]
        correction = bias - seed_bias
    else:
        raise ValueError(f"unknown correction mode: {correction_mode}")

    if components == "gyro":
        correction[:, 3:] = 0.0
    elif components == "acc":
        correction[:, :3] = 0.0
    elif components != "both":
        raise ValueError(f"unknown components: {components}")

    correction[:, :3] *= gyro_correction_scale
    correction[:, 3:] *= acc_correction_scale

    hybrid_imu[idx] = raw_imu[idx] - correction
    corr_norm = np.linalg.norm(correction, axis=1)
    gyro_norm = np.linalg.norm(correction[:, :3], axis=1)
    acc_norm = np.linalg.norm(correction[:, 3:], axis=1)
    return {
        "samples": int(len(idx)),
        "status": "ok",
        "mean_bias_norm": float(np.mean(corr_norm)),
        "max_bias_norm": float(np.max(corr_norm)),
        "mean_gyro_bias_norm": float(np.mean(gyro_norm)),
        "max_gyro_bias_norm": float(np.max(gyro_norm)),
        "mean_acc_bias_norm": float(np.mean(acc_norm)),
        "max_acc_bias_norm": float(np.max(acc_norm)),
        "final_bg_x": float(bg[0]),
        "final_bg_y": float(bg[1]),
        "final_bg_z": float(bg[2]),
        "final_ba_x": float(ba[0]),
        "final_ba_y": float(ba[1]),
        "final_ba_z": float(ba[2]),
    }


def process_sequence(sequence: str, args: argparse.Namespace, bw_model, ba_model, device: str) -> dict[str, object]:
    raw_dir = args.raw_openvins_root / sequence
    raw_imu_path = args.data_root / "0524" / sequence / "imu_diagnostics" / "imu_left.csv"
    out_dir = args.output_root / sequence
    out_dir.mkdir(parents=True, exist_ok=True)

    stamps, raw_imu = read_imu_csv(raw_imu_path)
    hybrid_imu = raw_imu.copy()
    est = read_openvins_estimate(raw_dir / "estimate.txt")
    max_pt_trace = raw_max_position_cov_trace(raw_dir / "odomimu.txt")
    gate_enabled = args.gate_max_position_cov_trace > 0.0
    gate_pass = (not gate_enabled) or max_pt_trace > args.gate_max_position_cov_trace
    segments = (
        read_weak_visual_segments(
            raw_dir / "timing.txt",
            args.msckf_update_threshold,
            args.min_weak_duration,
            args.merge_gap,
        )
        if gate_pass
        else []
    )

    segment_rows: list[dict[str, object]] = []
    total_changed = 0
    for seg_index, (start, end, weak_count) in enumerate(segments):
        seed_idx = int(np.searchsorted(est["t"], start, side="right") - 1)
        mask = (stamps >= start) & (stamps <= end)
        row: dict[str, object] = {
            "sequence": sequence,
            "segment_index": seg_index,
            "start": start,
            "end": end,
            "duration_s": end - start,
            "weak_update_rows": weak_count,
            "imu_samples_in_segment": int(np.sum(mask)),
        }
        if seed_idx < 0 or not np.any(mask):
            row["status"] = "skipped_no_seed_or_imu"
            segment_rows.append(row)
            continue
        bg0 = est["bg"][seed_idx]
        ba0 = est["ba"][seed_idx]
        meta = compensate_segment(
            stamps,
            raw_imu,
            hybrid_imu,
            mask,
            bg0,
            ba0,
            bw_model,
            ba_model,
            device,
            args.correction_mode,
            args.components,
            args.gyro_correction_scale,
            args.acc_correction_scale,
        )
        row.update(
            {
                "seed_time": float(est["t"][seed_idx]),
                "seed_bg_x": float(bg0[0]),
                "seed_bg_y": float(bg0[1]),
                "seed_bg_z": float(bg0[2]),
                "seed_ba_x": float(ba0[0]),
                "seed_ba_y": float(ba0[1]),
                "seed_ba_z": float(ba0[2]),
                **meta,
            }
        )
        if meta.get("status") == "ok":
            total_changed += int(meta["samples"])
        segment_rows.append(row)

    hybrid_path = out_dir / "hybrid_imu.csv"
    write_imu_csv(hybrid_path, stamps, hybrid_imu)
    write_csv(out_dir / "segments.csv", segment_rows)
    max_delta = float(np.max(np.linalg.norm(hybrid_imu - raw_imu, axis=1))) if len(raw_imu) else math.nan
    mean_delta_changed = (
        float(np.mean(np.linalg.norm((hybrid_imu - raw_imu)[np.linalg.norm(hybrid_imu - raw_imu, axis=1) > 0], axis=1)))
        if np.any(np.linalg.norm(hybrid_imu - raw_imu, axis=1) > 0)
        else 0.0
    )
    run_config = {
        "sequence": sequence,
        "raw_imu": str(raw_imu_path),
        "hybrid_imu": str(hybrid_path),
        "raw_openvins_dir": str(raw_dir),
        "segments_csv": str(out_dir / "segments.csv"),
        "msckf_update_threshold": args.msckf_update_threshold,
        "min_weak_duration": args.min_weak_duration,
        "merge_gap": args.merge_gap,
        "correction_mode": args.correction_mode,
        "components": args.components,
        "gyro_correction_scale": args.gyro_correction_scale,
        "acc_correction_scale": args.acc_correction_scale,
        "gate_max_position_cov_trace": args.gate_max_position_cov_trace,
        "raw_max_position_cov_trace": max_pt_trace,
        "gate_pass": gate_pass,
        "samples": int(len(stamps)),
        "changed_samples": int(total_changed),
        "changed_fraction": float(total_changed / len(stamps)) if len(stamps) else 0.0,
        "max_delta_norm": max_delta,
        "mean_delta_norm_changed": mean_delta_changed,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    return {
        "sequence": sequence,
        "segments": len(segments),
        "segments_ok": sum(1 for row in segment_rows if row.get("status") == "ok"),
        "raw_max_position_cov_trace": max_pt_trace,
        "gate_pass": gate_pass,
        "samples": int(len(stamps)),
        "changed_samples": int(total_changed),
        "changed_fraction": float(total_changed / len(stamps)) if len(stamps) else 0.0,
        "max_delta_norm": max_delta,
        "mean_delta_norm_changed": mean_delta_changed,
        "hybrid_imu": str(hybrid_path),
    }


def write_summary(args: argparse.Namespace, rows: list[dict[str, object]], gyro_ckpt: Path, acc_ckpt: Path) -> None:
    total_samples = sum(int(row["samples"]) for row in rows)
    total_changed = sum(int(row["changed_samples"]) for row in rows)
    lines = [
        "# AquaLux Segment-Only Hybrid IMU Summary",
        "",
        "Method: keep raw IMU in light-on intervals and compensate only raw-OpenVINS-detected light-off intervals.",
        "",
        f"- Correction mode: `{args.correction_mode}`.",
        f"- Components: `{args.components}`.",
        f"- Gyro correction scale: `{args.gyro_correction_scale:.12g}`.",
        f"- Acc correction scale: `{args.acc_correction_scale:.12g}`.",
        f"- Gate max position covariance trace: `{args.gate_max_position_cov_trace:.12g}`.",
        f"- Gyro checkpoint: `{gyro_ckpt}`",
        f"- Acc checkpoint: `{acc_ckpt}`",
        f"- Changed samples: `{total_changed}/{total_samples}` (`{(total_changed / total_samples if total_samples else 0.0):.6g}`).",
        "",
        "Light-off detection:",
        "",
        "```text",
        f"msckf_update_time < {args.msckf_update_threshold:.12g}",
        f"min_weak_duration = {args.min_weak_duration:.12g} s",
        f"merge_gap = {args.merge_gap:.12g} s",
        "```",
        "",
        "| sequence | gate pass | raw max Pt trace | segments | ok segments | samples | changed samples | changed fraction | max delta norm | mean delta norm changed |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['sequence']}` | {row['gate_pass']} | {float(row['raw_max_position_cov_trace']):.6g} | "
            f"{row['segments']} | {row['segments_ok']} | {row['samples']} | {row['changed_samples']} | {float(row['changed_fraction']):.6g} | "
            f"{float(row['max_delta_norm']):.6g} | {float(row['mean_delta_norm_changed']):.6g} |"
        )
    lines.extend(
        [
            "",
            "Artifacts:",
            "",
            f"- Metrics CSV: `{args.output_root / 'hybrid_imu_metrics.csv'}`",
            f"- Per-sequence hybrid IMU: `{args.output_root}/Test_*/hybrid_imu.csv`",
        ]
    )
    (args.output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    bw_model, ba_model, gyro_ckpt, acc_ckpt = load_models(args, device)
    rows: list[dict[str, object]] = []
    for index, sequence in enumerate(args.sequences, start=1):
        print(f"[{index}/{len(args.sequences)}] segment-only hybrid IMU/{sequence}", flush=True)
        row = process_sequence(sequence, args, bw_model, ba_model, device)
        rows.append(row)
        write_csv(args.output_root / "hybrid_imu_metrics.csv", rows)
        (args.output_root / "hybrid_imu_metrics.json").write_text(
            json.dumps(rows, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        write_summary(args, rows, gyro_ckpt, acc_ckpt)
        print(
            f"  segments={row['segments_ok']}/{row['segments']} "
            f"changed={row['changed_samples']} ({float(row['changed_fraction']):.4%}) "
            f"max_delta={float(row['max_delta_norm']):.4g}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
