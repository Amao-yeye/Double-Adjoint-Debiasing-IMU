#!/usr/bin/env python3
"""Run raw IMU preintegration for AquaLux Test sequences.

The raw method intentionally applies no bias compensation. It initializes from
the GT pose and a GT-derived initial velocity, then integrates the measured IMU
with fixed gravity in the AquaLux gravity-aligned world.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(
    os.environ.get("AQUALUX_DATA_ROOT", REPO_ROOT / "datasets" / "AquaLux" / "0524_data")
)
DEFAULT_OUTPUT_ROOT = Path("Results/AquaLux/preintegration")
DEFAULT_GRAVITY_NORM = 9.7879


def sorted_test_sequences(seq_root: Path) -> list[str]:
    return sorted(
        [p.name for p in seq_root.iterdir() if p.is_dir() and p.name.startswith("Test_")],
        key=lambda name: int(name.split("_", 1)[1]),
    )


def read_tum(path: Path) -> tuple[np.ndarray, np.ndarray, Rotation]:
    stamps: list[float] = []
    positions: list[list[float]] = []
    quats: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                raise ValueError(f"{path}:{line_no}: expected TUM format with 8 fields")
            vals = [float(x) for x in parts]
            stamps.append(vals[0])
            positions.append(vals[1:4])
            quats.append(vals[4:8])
    if len(stamps) < 2:
        raise ValueError(f"{path}: need at least two poses")
    stamps_arr = np.asarray(stamps, dtype=float)
    order = np.argsort(stamps_arr)
    stamps_arr = stamps_arr[order]
    if np.any(np.diff(stamps_arr) <= 0):
        raise ValueError(f"{path}: timestamps must be strictly increasing")
    positions_arr = np.asarray(positions, dtype=float)[order]
    quats_arr = np.asarray(quats, dtype=float)[order]
    quats_arr = quats_arr / np.linalg.norm(quats_arr, axis=1, keepdims=True)
    return stamps_arr, positions_arr, Rotation.from_quat(quats_arr)


def read_imu_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if rows.ndim == 0:
        rows = rows.reshape(1)
    required = ("timestamp", "wx", "wy", "wz", "ax", "ay", "az")
    if rows.dtype.names is None or any(name not in rows.dtype.names for name in required):
        raise ValueError(f"{path}: missing required IMU columns")
    stamps = np.asarray(rows["timestamp"], dtype=float)
    imu = np.column_stack([rows[name] for name in required[1:]]).astype(float)
    order = np.argsort(stamps)
    stamps = stamps[order]
    imu = imu[order]
    keep = np.r_[True, np.diff(stamps) > 0]
    return stamps[keep], imu[keep]


def make_uniform_grid(start: float, end: float, rate_hz: float) -> np.ndarray:
    dt = 1.0 / rate_hz
    first = math.ceil(start / dt) * dt
    last = math.floor(end / dt) * dt
    if last <= first:
        raise ValueError("no valid overlap for uniform grid")
    count = int(math.floor((last - first) / dt)) + 1
    return first + np.arange(count, dtype=float) * dt


def interpolate_imu(stamps: np.ndarray, imu: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(query, stamps, imu[:, i]) for i in range(6)])


def interpolate_gt(
    gt_stamps: np.ndarray,
    gt_positions: np.ndarray,
    gt_rotations: Rotation,
    query: np.ndarray,
) -> tuple[np.ndarray, Rotation, np.ndarray]:
    positions = np.column_stack([np.interp(query, gt_stamps, gt_positions[:, i]) for i in range(3)])
    rotations = Slerp(gt_stamps, gt_rotations)(query)
    nearest_idx = np.searchsorted(gt_stamps, query)
    nearest_idx = np.clip(nearest_idx, 1, len(gt_stamps) - 1)
    left = gt_stamps[nearest_idx - 1]
    right = gt_stamps[nearest_idx]
    nearest_dt = np.minimum(np.abs(query - left), np.abs(right - query))
    return positions, rotations, nearest_dt


def smooth_velocity(positions: np.ndarray, dt: float, smooth_sec: float, polyorder: int) -> np.ndarray:
    requested = max(polyorder + 2, int(round(smooth_sec / dt)))
    if requested % 2 == 0:
        requested += 1
    if requested >= len(positions):
        requested = len(positions) - 1 if len(positions) % 2 == 0 else len(positions)
    if requested <= polyorder or requested < 3:
        return np.gradient(positions, dt, axis=0, edge_order=1)
    velocity = np.zeros_like(positions)
    for axis in range(3):
        velocity[:, axis] = savgol_filter(
            positions[:, axis],
            requested,
            polyorder,
            deriv=1,
            delta=dt,
            mode="interp",
        )
    return velocity


def integrate_raw_imu(
    stamps: np.ndarray,
    imu: np.ndarray,
    initial_position: np.ndarray,
    initial_rotation: Rotation,
    initial_velocity: np.ndarray,
    gravity: np.ndarray,
) -> tuple[np.ndarray, Rotation]:
    n = len(stamps)
    positions = np.zeros((n, 3), dtype=float)
    rotations = [initial_rotation]
    velocity = np.asarray(initial_velocity, dtype=float).copy()
    positions[0] = initial_position
    for i in range(n - 1):
        dt = float(stamps[i + 1] - stamps[i])
        rot_i = rotations[-1]
        acc_world = rot_i.apply(imu[i, 3:6]) + gravity
        positions[i + 1] = positions[i] + velocity * dt + 0.5 * acc_world * dt * dt
        velocity = velocity + acc_world * dt
        rotations.append(rot_i * Rotation.from_rotvec(imu[i, :3] * dt))
    return positions, Rotation.concatenate(rotations)


def rotation_errors_deg(pred: Rotation, gt: Rotation) -> np.ndarray:
    return (gt.inv() * pred).magnitude() * 180.0 / math.pi


def rmse(values: np.ndarray) -> float:
    if len(values) == 0:
        return math.nan
    return float(np.sqrt(np.mean(values * values)))


def write_tum(path: Path, stamps: np.ndarray, positions: np.ndarray, rotations: Rotation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    quats = rotations.as_quat()
    with path.open("w", encoding="utf-8") as f:
        for stamp, pos, quat in zip(stamps, positions, quats):
            f.write(
                f"{stamp:.9f} "
                f"{pos[0]:.12g} {pos[1]:.12g} {pos[2]:.12g} "
                f"{quat[0]:.12g} {quat[1]:.12g} {quat[2]:.12g} {quat[3]:.12g}\n"
            )


def write_errors_csv(
    path: Path,
    stamps: np.ndarray,
    valid_mask: np.ndarray,
    pos_err: np.ndarray,
    rot_err: np.ndarray,
    gt_nearest_dt: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "valid_metric_sample",
                "gt_nearest_dt_sec",
                "position_error_m",
                "rotation_error_deg",
            ]
        )
        for row in zip(stamps, valid_mask, gt_nearest_dt, pos_err, rot_err):
            writer.writerow(
                [
                    f"{row[0]:.9f}",
                    int(row[1]),
                    f"{row[2]:.12g}",
                    f"{row[3]:.12g}",
                    f"{row[4]:.12g}",
                ]
            )


def process_sequence(
    seq: str,
    data_root: Path,
    raw_root: Path,
    rate_hz: float,
    gravity_norm: float,
    gt_valid_max_dt: float,
    velocity_smooth_sec: float,
) -> dict[str, float | int | str]:
    seq_dir = data_root / "0524" / seq
    gt_path = seq_dir / "ATO" / "gt_imu_gravity_aligned.tum"
    imu_path = seq_dir / "imu_diagnostics" / "imu_left.csv"
    gt_stamps, gt_positions_raw, gt_rotations_raw = read_tum(gt_path)
    imu_stamps, imu_raw = read_imu_csv(imu_path)

    start = max(float(gt_stamps[0]), float(imu_stamps[0]))
    end = min(float(gt_stamps[-1]), float(imu_stamps[-1]))
    stamps = make_uniform_grid(start, end, rate_hz)
    dt = 1.0 / rate_hz
    gt_positions, gt_rotations, gt_nearest_dt = interpolate_gt(
        gt_stamps,
        gt_positions_raw,
        gt_rotations_raw,
        stamps,
    )
    imu = interpolate_imu(imu_stamps, imu_raw, stamps)
    gt_velocity = smooth_velocity(gt_positions, dt, velocity_smooth_sec, polyorder=3)
    gravity = np.array([0.0, 0.0, -float(gravity_norm)], dtype=float)
    pred_positions, pred_rotations = integrate_raw_imu(
        stamps,
        imu,
        gt_positions[0],
        gt_rotations[0],
        gt_velocity[0],
        gravity,
    )

    pos_err = np.linalg.norm(pred_positions - gt_positions, axis=1)
    rot_err = rotation_errors_deg(pred_rotations, gt_rotations)
    valid_mask = gt_nearest_dt <= gt_valid_max_dt
    if valid_mask.sum() < 10:
        raise RuntimeError(f"{seq}: too few valid GT metric samples")

    seq_raw_dir = raw_root / seq
    write_tum(seq_raw_dir / "gt_resampled.tum", stamps, gt_positions, gt_rotations)
    write_tum(seq_raw_dir / "raw_imu_preintegrated.tum", stamps, pred_positions, pred_rotations)
    write_errors_csv(seq_raw_dir / "errors.csv", stamps, valid_mask, pos_err, rot_err, gt_nearest_dt)

    return {
        "sequence": seq,
        "samples": int(len(stamps)),
        "valid_metric_samples": int(valid_mask.sum()),
        "duration_sec": float(stamps[-1] - stamps[0]),
        "gt_interp_max_dt_sec": float(gt_nearest_dt.max()),
        "gt_valid_max_dt_sec": float(gt_valid_max_dt),
        "position_rmse_m": rmse(pos_err[valid_mask]),
        "rotation_rmse_deg": rmse(rot_err[valid_mask]),
        "position_mean_m": float(np.mean(pos_err[valid_mask])),
        "rotation_mean_deg": float(np.mean(rot_err[valid_mask])),
        "position_max_m": float(np.max(pos_err[valid_mask])),
        "rotation_max_deg": float(np.max(rot_err[valid_mask])),
        "gt_resampled": str(seq_raw_dir / "gt_resampled.tum"),
        "raw_preintegrated": str(seq_raw_dir / "raw_imu_preintegrated.tum"),
        "errors_csv": str(seq_raw_dir / "errors.csv"),
    }


def write_metrics_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        "raw_preintegrated",
        "errors_csv",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_from_error_files(rows: list[dict[str, float | int | str]]) -> dict[str, float | int]:
    pos_parts: list[np.ndarray] = []
    rot_parts: list[np.ndarray] = []
    for row in rows:
        path = Path(str(row["errors_csv"]))
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
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


def write_summary_md(
    path: Path,
    rows: list[dict[str, float | int | str]],
    aggregate: dict[str, float | int],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mean_pos = float(np.mean([float(row["position_rmse_m"]) for row in rows]))
    mean_rot = float(np.mean([float(row["rotation_rmse_deg"]) for row in rows]))
    lines = [
        "# AquaLux Preintegration Summary",
        "",
        "## Scope",
        "",
        "- Split: Test sequences only.",
        "- Method currently evaluated: `raw_imu`.",
        "- Raw IMU means no learned, pseudo, or constant bias compensation.",
        f"- IMU/GT resampling rate: `{args.rate_hz:g} Hz`.",
        f"- Gravity: `[0, 0, -{args.gravity_norm:g}] m/s^2`.",
        "- Initialization: GT pose and GT-derived initial velocity at the first overlap sample.",
        (
            "- RMSE metrics are computed only at samples whose nearest source GT pose is within "
            f"`{args.gt_valid_max_dt:g} s`; trajectory files still contain the full IMU/GT overlap."
        ),
        "",
        "## Method Comparison",
        "",
        "| Method | Sequences | Valid samples | Position RMSE (m) | Rotation RMSE (deg) | Mean seq pos RMSE (m) | Mean seq rot RMSE (deg) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| `raw_imu` | {len(rows)} | {aggregate['valid_metric_samples']} | "
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
            "- Raw method directory: `Results/AquaLux/preintegration/raw`.",
            "- Per-sequence files:",
            "  - `gt_resampled.tum`",
            "  - `raw_imu_preintegrated.tum`",
            "  - `errors.csv`",
            "- Machine-readable metrics: `Results/AquaLux/preintegration/raw/metrics.csv`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seq", action="append", help="Test sequence to process; repeat for multiple")
    parser.add_argument("--rate-hz", type=float, default=125.0)
    parser.add_argument("--gravity-norm", type=float, default=DEFAULT_GRAVITY_NORM)
    parser.add_argument("--gt-valid-max-dt", type=float, default=0.02)
    parser.add_argument("--velocity-smooth-sec", type=float, default=0.12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seq_root = args.data_root / "0524"
    sequences = args.seq or sorted_test_sequences(seq_root)
    raw_root = args.output_root / "raw"
    rows = [
        process_sequence(
            seq,
            args.data_root,
            raw_root,
            args.rate_hz,
            args.gravity_norm,
            args.gt_valid_max_dt,
            args.velocity_smooth_sec,
        )
        for seq in sequences
    ]
    metrics_path = raw_root / "metrics.csv"
    write_metrics_csv(metrics_path, rows)
    aggregate = aggregate_from_error_files(rows)
    write_summary_md(args.output_root / "summary.md", rows, aggregate, args)
    print(f"Wrote {metrics_path}")
    print(f"Wrote {args.output_root / 'summary.md'}")
    print(
        "raw_imu aggregate: "
        f"position_rmse_m={aggregate['position_rmse_m']:.6f}, "
        f"rotation_rmse_deg={aggregate['rotation_rmse_deg']:.6f}"
    )


if __name__ == "__main__":
    main()
