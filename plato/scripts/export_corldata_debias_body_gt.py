#!/usr/bin/env python3
"""Export corldata mocap GT as Debias_IMU-ready IMU/body supervision.

This exporter is intentionally separate from the VO pseudo-GT exporter.  It
uses corldata `Body_0.tum` mocap poses plus raw DAVIS IMU messages, estimates a
constant Body_0-to-IMU rotation from angular rates, interpolates the GT pose to
a fixed IMU-rate grid, and writes a TUM-VI-style tree:

    <aggregate>/<sequence>/mav0/imu0/data.csv
    <aggregate>/<sequence>/mav0/mocap0/data.csv

The supervision target is `T_W_I`, the IMU/body pose in a gravity-aligned world.
No VO trajectory is used as the supervised target.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


PLATO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ESTIMATE = PLATO_ROOT / "tools" / "estimate_mocap_body_gt.py"
DEFAULT_DATASET_ROOT = Path("/home/amao/datasets/corldata")
DEFAULT_PSEUDO_ROOT = PLATO_ROOT / "results" / "pseudo_gt" / "corldata"
DEFAULT_AGGREGATE_ROOT = PLATO_ROOT / "results" / "debias_imu" / "corldata_gt_body_1000hz"
DEFAULT_TRAIN_SEQS = ["S00", "S02", "S03", "S07"]
DEFAULT_TEST_SEQS = ["S13", "S15", "S20", "S21", "S22", "S23"]
DEFAULT_GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float64)


spec = importlib.util.spec_from_file_location("plato_estimate_mocap_body_gt", TOOLS_ESTIMATE)
estimate_tool = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = estimate_tool
assert spec.loader is not None
spec.loader.exec_module(estimate_tool)
estimate_tool.require_scipy()


@dataclass
class PoseSeries:
    t: np.ndarray
    p: np.ndarray
    rot: Rotation
    quat_norm_max_error: float
    duplicate_or_nonmonotonic_count: int


@dataclass
class SequenceExport:
    sequence: str
    split: str
    rows: int
    duration_s: float
    output_rate_hz: float
    raw_window_count: int
    valid_window_count: int
    dropped_window_count: int
    drop_reason_statistics: dict[str, int]
    body_imu_rotation_rmse_rad_s: float | None
    body_imu_gyro_bias_rad_s: list[float] | None
    gravity_before_alignment_m_s2: list[float]
    output_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--pseudo-root", type=Path, default=DEFAULT_PSEUDO_ROOT)
    parser.add_argument("--aggregate-root", type=Path, default=DEFAULT_AGGREGATE_ROOT)
    parser.add_argument("--train-seqs", nargs="+", default=DEFAULT_TRAIN_SEQS)
    parser.add_argument("--test-seqs", nargs="+", default=DEFAULT_TEST_SEQS)
    parser.add_argument("--output-rate-hz", type=float, default=1000.0)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--body-imu-alignment", choices=["gyro", "identity"], default="gyro")
    parser.add_argument("--gravity-magnitude", type=float, default=9.81)
    parser.add_argument("--max-gt-gap-s", type=float, default=0.03)
    parser.add_argument("--max-imu-gap-s", type=float, default=0.02)
    parser.add_argument("--gap-pad-s", type=float, default=0.01)
    parser.add_argument("--max-speed-m-s", type=float, default=3.0)
    parser.add_argument("--max-accel-m-s2", type=float, default=80.0)
    parser.add_argument("--max-angular-speed-rad-s", type=float, default=12.0)
    parser.add_argument("--min-align-angular-rate-rad-s", type=float, default=0.15)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_tum_xyzw(path: Path) -> PoseSeries:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            values = [float(item) for item in parts[:8]]
            if np.all(np.isfinite(values)):
                rows.append(values)
    if len(rows) < 2:
        raise ValueError(f"{path} has fewer than two finite poses")

    data = np.asarray(rows, dtype=np.float64)
    order = np.argsort(data[:, 0], kind="stable")
    data = data[order]
    keep = np.concatenate(([True], np.diff(data[:, 0]) > 0.0))
    duplicate_count = int(np.count_nonzero(~keep))
    data = data[keep]
    if data.shape[0] < 2:
        raise ValueError(f"{path} has fewer than two strictly increasing poses")

    q = data[:, 4:8]
    norms = np.linalg.norm(q, axis=1)
    if np.any(norms < 1e-12):
        raise ValueError(f"{path} contains a near-zero quaternion")
    quat_norm_max_error = float(np.max(np.abs(norms - 1.0)))
    q = q / norms[:, None]
    return PoseSeries(
        t=data[:, 0],
        p=data[:, 1:4],
        rot=Rotation.from_quat(q),
        quat_norm_max_error=quat_norm_max_error,
        duplicate_or_nonmonotonic_count=duplicate_count,
    )


def load_imu_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 7:
        raise ValueError(f"{path} must have timestamp plus six IMU columns")
    order = np.argsort(data[:, 0], kind="stable")
    data = data[order]
    keep = np.concatenate(([True], np.diff(data[:, 0]) > 0.0))
    data = data[keep]
    t = data[:, 0].astype(np.float64) * 1e-9
    u = data[:, 1:7].astype(np.float64)
    if len(t) < 2:
        raise ValueError(f"{path} has fewer than two IMU rows")
    return t, u


def make_query_grid(t_min: float, t_max: float, rate_hz: float) -> tuple[np.ndarray, np.ndarray, int]:
    if rate_hz <= 0.0:
        raise ValueError("--output-rate-hz must be positive")
    dt_ns_float = 1e9 / float(rate_hz)
    dt_ns = int(round(dt_ns_float))
    if abs(dt_ns - dt_ns_float) > 1e-3:
        raise ValueError(f"output rate {rate_hz} Hz does not map cleanly to integer nanoseconds")
    start_ns = int(math.ceil(t_min * 1e9 / dt_ns) * dt_ns)
    end_ns = int(math.floor(t_max * 1e9 / dt_ns) * dt_ns)
    if end_ns <= start_ns:
        raise ValueError("No fixed-rate timestamps inside GT/IMU overlap")
    query_ns = np.arange(start_ns, end_ns + 1, dt_ns, dtype=np.int64)
    query_t = query_ns.astype(np.float64) * 1e-9
    return query_t, query_ns, dt_ns


def interp_columns(t_src: np.ndarray, values: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    out = np.empty((len(t_query), values.shape[1]), dtype=np.float64)
    for axis in range(values.shape[1]):
        out[:, axis] = np.interp(t_query, t_src, values[:, axis])
    return out


def interpolate_gt(gt: PoseSeries, query_t: np.ndarray) -> PoseSeries:
    traj = estimate_tool.PoseTrajectory("corldata_Body_0", gt.t, gt.p, gt.rot)
    interp = estimate_tool.interpolate_trajectory(traj, query_t, name="corldata_Body_0_interp")
    return PoseSeries(
        t=interp.t,
        p=interp.p,
        rot=interp.rot,
        quat_norm_max_error=gt.quat_norm_max_error,
        duplicate_or_nonmonotonic_count=gt.duplicate_or_nonmonotonic_count,
    )


def so3_log_matrix(Rm: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(Rm).as_rotvec()


def angular_velocity_body(rot: Rotation, t: np.ndarray) -> np.ndarray:
    Rm = rot.as_matrix()
    dt = np.diff(t)
    w = np.zeros((len(t), 3), dtype=np.float64)
    for idx in range(len(t) - 1):
        if dt[idx] <= 0.0:
            continue
        w[idx] = so3_log_matrix(Rm[idx].T @ Rm[idx + 1]) / dt[idx]
    w[-1] = w[-2]
    return w


def smooth(data: np.ndarray, target_window: int, order: int) -> np.ndarray:
    if len(data) < order + 3:
        return data.copy()
    window = min(target_window, len(data) if len(data) % 2 else len(data) - 1)
    min_window = order + 2 if (order + 2) % 2 else order + 3
    window = max(window, min_window)
    if window >= len(data):
        window = len(data) if len(data) % 2 else len(data) - 1
    if window <= order:
        return data.copy()
    return savgol_filter(data, window, order, axis=0, mode="interp")


def estimate_body0_to_imu_rotation(
    w_body0: np.ndarray,
    gyro_imu: np.ndarray,
    min_rate: float,
) -> tuple[np.ndarray, dict[str, object]]:
    finite = np.all(np.isfinite(w_body0), axis=1) & np.all(np.isfinite(gyro_imu), axis=1)
    dynamic = np.linalg.norm(w_body0, axis=1) > min_rate
    mask = finite & dynamic
    if int(np.count_nonzero(mask)) < 20:
        return np.eye(3), {
            "method": "identity_fallback",
            "warning": "too few dynamic samples for gyro-based frame alignment",
            "sample_count": int(np.count_nonzero(mask)),
            "gyro_bias_rad_s": None,
            "rotation_rmse_rad_s": None,
        }

    w = w_body0[mask]
    g = gyro_imu[mask]
    g_center = g - np.median(g, axis=0, keepdims=True)
    w_center = w - np.median(w, axis=0, keepdims=True)
    center_mask = (np.linalg.norm(g_center, axis=1) > min_rate) & (np.linalg.norm(w_center, axis=1) > min_rate)
    if int(np.count_nonzero(center_mask)) >= 20:
        g_fit = g_center[center_mask]
        w_fit = w_center[center_mask]
    else:
        g_fit = g_center
        w_fit = w_center

    rot, _ = Rotation.align_vectors(w_fit, g_fit)
    R_m_i = rot.as_matrix()
    gyro_bias = np.zeros(3, dtype=np.float64)
    for _ in range(5):
        gyro_bias = np.median(g - (R_m_i.T @ w.T).T, axis=0)
        rot, _ = Rotation.align_vectors(w, g - gyro_bias.reshape(1, 3))
        R_m_i = rot.as_matrix()
    residual = (R_m_i @ (g - gyro_bias.reshape(1, 3)).T).T - w
    rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    return R_m_i, {
        "method": "gyro_wahba_with_constant_bias",
        "sample_count": int(len(w)),
        "gyro_bias_rad_s": gyro_bias.tolist(),
        "rotation_rmse_rad_s": rmse,
        "R_Body0_IMU": R_m_i.tolist(),
    }


def rotation_align_vector(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_norm = np.linalg.norm(source)
    target_norm = np.linalg.norm(target)
    if source_norm < 1e-9 or target_norm < 1e-9:
        return np.eye(3, dtype=np.float64)
    a = source / source_norm
    b = target / target_norm
    v = np.cross(a, b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if c > 1.0 - 1e-12:
        return np.eye(3, dtype=np.float64)
    if c < -1.0 + 1e-12:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.8:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = axis - np.dot(axis, a) * a
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(math.pi * axis).as_matrix()
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float64)
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def estimate_gravity_world(t: np.ndarray, p: np.ndarray, R_w_i: np.ndarray, acc_imu: np.ndarray) -> np.ndarray:
    p_smooth = smooth(p, min(401, len(p) if len(p) % 2 else len(p) - 1), 3)
    v = np.gradient(p_smooth, t, axis=0, edge_order=2)
    a_world = np.gradient(v, t, axis=0, edge_order=2)
    g_samples = a_world - np.einsum("nij,nj->ni", R_w_i, acc_imu)
    finite = np.all(np.isfinite(g_samples), axis=1)
    g_samples = g_samples[finite]
    if len(g_samples) < 20:
        return DEFAULT_GRAVITY.copy()
    norms = np.linalg.norm(g_samples, axis=1)
    lo, hi = np.percentile(norms, [5.0, 95.0])
    trimmed = g_samples[(norms >= lo) & (norms <= hi)]
    if len(trimmed) >= 20:
        g_samples = trimmed
    return np.median(g_samples, axis=0)


def apply_body_and_world_alignment(
    gt_interp: PoseSeries,
    imu_interp: np.ndarray,
    body_imu_alignment: str,
    min_align_rate: float,
    gravity_magnitude: float,
) -> tuple[np.ndarray, Rotation, dict[str, object], np.ndarray, np.ndarray]:
    R_w_m = gt_interp.rot.as_matrix()
    w_m = angular_velocity_body(gt_interp.rot, gt_interp.t)
    if body_imu_alignment == "gyro":
        R_m_i, body_record = estimate_body0_to_imu_rotation(w_m, imu_interp[:, :3], min_align_rate)
    else:
        R_m_i = np.eye(3, dtype=np.float64)
        body_record = {
            "method": "identity",
            "sample_count": 0,
            "gyro_bias_rad_s": None,
            "rotation_rmse_rad_s": None,
            "R_Body0_IMU": R_m_i.tolist(),
        }

    R_w_i = R_w_m @ R_m_i
    p_w_i = gt_interp.p.copy()
    g_source = estimate_gravity_world(gt_interp.t, p_w_i, R_w_i, imu_interp[:, 3:6])
    target_g = np.array([0.0, 0.0, -float(gravity_magnitude)], dtype=np.float64)
    R_g = rotation_align_vector(g_source, target_g)
    R_wg_i = R_g @ R_w_i
    p_wg_i = (R_g @ p_w_i.T).T
    p_wg_i = p_wg_i - p_wg_i[0]
    rot_wg_i = Rotation.from_matrix(R_wg_i)
    return p_wg_i, rot_wg_i, body_record, R_g, g_source


def mark_gap_samples(t_query: np.ndarray, t_source: np.ndarray, max_gap: float, pad: float) -> np.ndarray:
    mask = np.zeros(len(t_query), dtype=bool)
    gaps = np.diff(t_source)
    bad_indices = np.where(gaps > max_gap)[0]
    for idx in bad_indices:
        start = t_source[idx] - pad
        stop = t_source[idx + 1] + pad
        mask |= (t_query >= start) & (t_query <= stop)
    return mask


def compute_quality_windows(
    t: np.ndarray,
    p: np.ndarray,
    rot: Rotation,
    imu: np.ndarray,
    gt_raw_t: np.ndarray,
    imu_raw_t: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object], dict[str, np.ndarray]]:
    n = len(t)
    reasons: dict[str, np.ndarray] = {}
    q = rot.as_quat()
    q_norm = np.linalg.norm(q, axis=1)
    reasons["nonfinite"] = ~(
        np.all(np.isfinite(p), axis=1)
        & np.all(np.isfinite(q), axis=1)
        & np.all(np.isfinite(imu), axis=1)
        & np.isfinite(t)
    )
    reasons["quaternion_abnormal"] = np.abs(q_norm - 1.0) > 1e-5
    reasons["gt_timestamp_gap"] = mark_gap_samples(t, gt_raw_t, args.max_gt_gap_s, args.gap_pad_s)
    reasons["imu_timestamp_gap"] = mark_gap_samples(t, imu_raw_t, args.max_imu_gap_s, args.gap_pad_s)

    v = np.gradient(p, t, axis=0, edge_order=2)
    a = np.gradient(v, t, axis=0, edge_order=2)
    w = angular_velocity_body(rot, t)
    reasons["speed_abnormal"] = np.linalg.norm(v, axis=1) > args.max_speed_m_s
    reasons["accel_abnormal"] = np.linalg.norm(a, axis=1) > args.max_accel_m_s2
    reasons["angular_speed_abnormal"] = np.linalg.norm(w, axis=1) > args.max_angular_speed_rad_s

    raw_window_count = max(0, n - args.window_size + 1)
    valid_starts: list[int] = []
    drop_counts = {name: 0 for name in reasons}
    sample_bad = np.zeros(n, dtype=bool)
    for mask in reasons.values():
        sample_bad |= mask

    for start in range(raw_window_count):
        stop = start + args.window_size
        if not np.any(sample_bad[start:stop]):
            valid_starts.append(start)
            continue
        for name, mask in reasons.items():
            if np.any(mask[start:stop]):
                drop_counts[name] += 1

    valid = np.asarray(valid_starts, dtype=np.int64)
    stats = {
        "window_size": int(args.window_size),
        "raw_window_count": int(raw_window_count),
        "valid_window_count": int(len(valid)),
        "dropped_window_count": int(raw_window_count - len(valid)),
        "drop_reason_statistics": {name: int(count) for name, count in drop_counts.items() if count > 0},
        "sample_reason_counts": {name: int(np.count_nonzero(mask)) for name, mask in reasons.items()},
        "max_speed_m_s": float(np.max(np.linalg.norm(v, axis=1))) if len(v) else 0.0,
        "max_accel_m_s2": float(np.max(np.linalg.norm(a, axis=1))) if len(a) else 0.0,
        "max_angular_speed_rad_s": float(np.max(np.linalg.norm(w, axis=1))) if len(w) else 0.0,
    }
    return valid, stats, reasons


def write_imu_csv(path: Path, query_ns: np.ndarray, imu: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp",
                "angular_velocity_x",
                "angular_velocity_y",
                "angular_velocity_z",
                "linear_acceleration_x",
                "linear_acceleration_y",
                "linear_acceleration_z",
            ]
        )
        for stamp, row in zip(query_ns, imu):
            writer.writerow([str(int(stamp))] + [f"{value:.12f}" for value in row])


def write_mocap_csv(path: Path, query_ns: np.ndarray, p: np.ndarray, rot: Rotation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    q_xyzw = rot.as_quat()
    q_xyzw = q_xyzw / np.linalg.norm(q_xyzw, axis=1, keepdims=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp",
                "p_RS_R_x [m]",
                "p_RS_R_y [m]",
                "p_RS_R_z [m]",
                "q_RS_w []",
                "q_RS_x []",
                "q_RS_y []",
                "q_RS_z []",
            ]
        )
        for stamp, pos, quat in zip(query_ns, p, q_xyzw):
            writer.writerow(
                [
                    str(int(stamp)),
                    f"{pos[0]:.12f}",
                    f"{pos[1]:.12f}",
                    f"{pos[2]:.12f}",
                    f"{quat[3]:.12f}",
                    f"{quat[0]:.12f}",
                    f"{quat[1]:.12f}",
                    f"{quat[2]:.12f}",
                ]
            )


def write_tum(path: Path, t: np.ndarray, p: np.ndarray, rot: Rotation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    q = rot.as_quat()
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    with path.open("w", encoding="utf-8") as handle:
        for ti, pi, qi in zip(t, p, q):
            handle.write(
                f"{ti:.9f} {pi[0]:.9f} {pi[1]:.9f} {pi[2]:.9f} "
                f"{qi[0]:.9f} {qi[1]:.9f} {qi[2]:.9f} {qi[3]:.9f}\n"
            )


def write_sanity_plots(out_dir: Path, seq: str, t: np.ndarray, p: np.ndarray, reasons: dict[str, np.ndarray]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    bad = np.zeros(len(t), dtype=bool)
    for mask in reasons.values():
        bad |= mask
    rel_t = t - t[0]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(p[:, 0], p[:, 1], color="black", linewidth=1.0, label="valid")
    if np.any(bad):
        ax.scatter(p[bad, 0], p[bad, 1], s=2, color="tab:red", label="flagged samples")
    ax.set_title(f"{seq} GT IMU/body trajectory XY")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "sanity_xy.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    labels = ["x", "y", "z"]
    for axis, label in enumerate(labels):
        axes[axis].plot(rel_t, p[:, axis], color="black", linewidth=0.9)
        if np.any(bad):
            axes[axis].scatter(rel_t[bad], p[bad, axis], s=2, color="tab:red")
        axes[axis].set_ylabel(f"{label} [m]")
        axes[axis].grid(True, alpha=0.25)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"{seq} GT IMU/body position")
    fig.tight_layout()
    fig.savefig(out_dir / "sanity_xyz.png", dpi=160)
    plt.close(fig)


def export_sequence(args: argparse.Namespace, sequence: str, split: str) -> SequenceExport:
    gt_path = args.dataset_root / "0511_data" / "gt" / sequence / "Body_0.tum"
    imu_path = args.pseudo_root / sequence / "sequence" / "imu.txt"
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)
    if not imu_path.exists():
        raise FileNotFoundError(imu_path)

    gt = load_tum_xyzw(gt_path)
    imu_t, imu_u = load_imu_txt(imu_path)
    overlap_start = max(float(gt.t[0]), float(imu_t[0]))
    overlap_end = min(float(gt.t[-1]), float(imu_t[-1]))
    query_t, query_ns, dt_ns = make_query_grid(overlap_start, overlap_end, args.output_rate_hz)
    gt_interp = interpolate_gt(gt, query_t)
    imu_interp = interp_columns(imu_t, imu_u, query_t)
    p_body, rot_body, body_record, R_gravity, g_source = apply_body_and_world_alignment(
        gt_interp,
        imu_interp,
        args.body_imu_alignment,
        args.min_align_angular_rate_rad_s,
        args.gravity_magnitude,
    )

    valid_starts, quality_stats, reason_masks = compute_quality_windows(
        query_t,
        p_body,
        rot_body,
        imu_interp,
        gt.t,
        imu_t,
        args,
    )

    per_seq_dir = args.pseudo_root / sequence / f"debias_imu_body_gt_{int(round(args.output_rate_hz))}hz_gt"
    aggregate_mav0 = args.aggregate_root / sequence / "mav0"
    if args.overwrite:
        for path in (per_seq_dir, args.aggregate_root / sequence):
            if path.exists():
                shutil.rmtree(path)
    per_seq_dir.mkdir(parents=True, exist_ok=True)
    aggregate_mav0.mkdir(parents=True, exist_ok=True)

    for root in (aggregate_mav0, per_seq_dir / "debias_tumvi_source" / "mav0"):
        write_imu_csv(root / "imu0" / "data.csv", query_ns, imu_interp)
        write_mocap_csv(root / "mocap0" / "data.csv", query_ns, p_body, rot_body)
        quality_dir = root / "quality"
        quality_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(quality_dir / f"valid_window_starts_{args.window_size}.txt", valid_starts, fmt="%d")
        (quality_dir / f"window_stats_{args.window_size}.json").write_text(
            json.dumps(quality_stats, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    write_tum(per_seq_dir / "body_gt_world_imu_frame.tum", query_t, p_body, rot_body)
    write_sanity_plots(per_seq_dir, sequence, query_t, p_body, reason_masks)

    metadata = {
        "dataset": "corldata",
        "sequence": sequence,
        "split": split,
        "source_gt": str(gt_path),
        "source_imu": str(imu_path),
        "target_supervision": "T_W_I, IMU/body pose in gravity-aligned world",
        "raw_gt_frame": "T_W_Body0 from mocap Body_0.tum",
        "frame_alignment": body_record,
        "translation_alignment": "p_Body0_IMU assumed zero; no calibrated lever arm is available",
        "gravity_alignment": {
            "route": "rotate world so median(a_world - R_WI * acc_imu) aligns to default -Z gravity",
            "gravity_before_alignment_m_s2": g_source.tolist(),
            "gravity_target_world_m_s2": [0.0, 0.0, -float(args.gravity_magnitude)],
            "R_gravity_alignment": R_gravity.tolist(),
        },
        "timestamp_policy": {
            "output_rate_hz": float(args.output_rate_hz),
            "dt_ns": int(dt_ns),
            "no_extrapolation": True,
            "overlap_start_s": overlap_start,
            "overlap_end_s": overlap_end,
        },
        "interpolation": {
            "position": "linear",
            "rotation": "SLERP via tools/estimate_mocap_body_gt.interpolate_trajectory",
            "imu": "linear per channel onto the same fixed-rate timestamp grid",
        },
        "quality": quality_stats,
        "quat_norm_max_error_raw": gt.quat_norm_max_error,
        "duplicate_or_nonmonotonic_gt_rows_dropped": gt.duplicate_or_nonmonotonic_count,
    }
    (per_seq_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return SequenceExport(
        sequence=sequence,
        split=split,
        rows=int(len(query_t)),
        duration_s=float(query_t[-1] - query_t[0]),
        output_rate_hz=float(args.output_rate_hz),
        raw_window_count=int(quality_stats["raw_window_count"]),
        valid_window_count=int(quality_stats["valid_window_count"]),
        dropped_window_count=int(quality_stats["dropped_window_count"]),
        drop_reason_statistics=dict(quality_stats["drop_reason_statistics"]),
        body_imu_rotation_rmse_rad_s=body_record.get("rotation_rmse_rad_s"),
        body_imu_gyro_bias_rad_s=body_record.get("gyro_bias_rad_s"),
        gravity_before_alignment_m_s2=g_source.tolist(),
        output_dir=str(per_seq_dir),
    )


def write_split_file(root: Path, train_seqs: list[str], test_seqs: list[str]) -> None:
    payload = {
        "train": train_seqs,
        "val": "per training sequence, last percent_for_val split in Debias_IMU runner",
        "test": test_seqs,
        "note": "S13+ are visually degraded testing sequences and are not used for training.",
    }
    (root / "split_used.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_summary(root: Path, rows: list[SequenceExport]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "window_quality_summary.csv"
    fieldnames = [
        "sequence",
        "split",
        "rows",
        "duration_s",
        "output_rate_hz",
        "raw_window_count",
        "valid_window_count",
        "dropped_window_count",
        "drop_reason_statistics",
        "body_imu_rotation_rmse_rad_s",
        "body_imu_gyro_bias_rad_s",
        "gravity_before_alignment_m_s2",
        "output_dir",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = row.__dict__.copy()
            for key in ("drop_reason_statistics", "body_imu_gyro_bias_rad_s", "gravity_before_alignment_m_s2"):
                payload[key] = json.dumps(payload[key], sort_keys=True)
            writer.writerow(payload)

    lines = [
        "# corldata GT IMU/Body Supervision Export",
        "",
        f"Aggregate root: `{root}`",
        "",
        "| sequence | split | rows | duration s | raw windows | valid windows | dropped | gyro align RMSE rad/s | drop reasons |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        rmse = "n/a" if row.body_imu_rotation_rmse_rad_s is None else f"{row.body_imu_rotation_rmse_rad_s:.5f}"
        reasons = json.dumps(row.drop_reason_statistics, sort_keys=True)
        lines.append(
            f"| {row.sequence} | {row.split} | {row.rows} | {row.duration_s:.3f} | "
            f"{row.raw_window_count} | {row.valid_window_count} | {row.dropped_window_count} | {rmse} | `{reasons}` |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "summary.json").write_text(
        json.dumps([row.__dict__ for row in rows], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.pseudo_root = args.pseudo_root.resolve()
    args.aggregate_root = args.aggregate_root.resolve()
    args.aggregate_root.mkdir(parents=True, exist_ok=True)

    rows: list[SequenceExport] = []
    for sequence in args.train_seqs:
        print(f"[export] train {sequence}", flush=True)
        rows.append(export_sequence(args, sequence, "train"))
    for sequence in args.test_seqs:
        print(f"[export] test  {sequence}", flush=True)
        rows.append(export_sequence(args, sequence, "test"))
    write_split_file(args.aggregate_root, list(args.train_seqs), list(args.test_seqs))
    write_summary(args.aggregate_root, rows)
    print(json.dumps({"aggregate_root": str(args.aggregate_root), "sequences": [row.__dict__ for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
