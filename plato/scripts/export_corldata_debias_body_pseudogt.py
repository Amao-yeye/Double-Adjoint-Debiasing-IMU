#!/usr/bin/env python3
"""Export corldata ORB-SLAM3 stereo VO pseudo-GT in the IMU/body frame.

Inputs are the current corldata evo-aligned stereo visual-only trajectories:

    results/pseudo_gt/corldata/<seq>/evo_gt_aligned/estimate_evo_aligned.tum

The estimate is the left DAVIS/camera pose after evo's SE(3) no-scale world
alignment. This script right-multiplies the physical camera-IMU extrinsic,
optionally rotates the world so estimated gravity points to -Z, and writes a
small Debias_IMU-style tree. IMU timestamps are shifted from the raw IMU clock
to the image/VO clock using uw_robot.yaml's td.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


PLATO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PSEUDO_ROOT = PLATO_ROOT / "results" / "pseudo_gt" / "corldata"
DEFAULT_AGGREGATE_ROOT = PLATO_ROOT / "results" / "debias_imu" / "corldata_training_pseudogt"
DEFAULT_ORB_CONFIG = PLATO_ROOT / "plato" / "configs" / "orbslam3" / "corldata_davis_stereo_inertial.yaml"
DEFAULT_SEQUENCES = ["S00", "S02", "S03", "S07"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    parser.add_argument("--pseudo-root", type=Path, default=DEFAULT_PSEUDO_ROOT)
    parser.add_argument("--aggregate-root", type=Path, default=DEFAULT_AGGREGATE_ROOT)
    parser.add_argument("--orb-config", type=Path, default=DEFAULT_ORB_CONFIG)
    parser.add_argument("--imu-time-shift-s", type=float, default=0.00777437)
    parser.add_argument("--gravity-magnitude", type=float, default=9.7803)
    parser.add_argument("--no-gravity-align", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_matrix_from_yaml(path: Path, key: str) -> np.ndarray:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(path)
    node = fs.getNode(key)
    mat = node.mat()
    fs.release()
    if mat is None:
        raise KeyError(f"{key} not found in {path}")
    mat = np.asarray(mat, dtype=float)
    if mat.shape != (4, 4):
        raise ValueError(f"{path}:{key} has shape {mat.shape}, expected (4, 4)")
    return mat


def load_tum(path: Path) -> list[dict[str, np.ndarray | float]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 8:
                continue
            rows.append(
                {
                    "t": float(parts[0]),
                    "p": np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float),
                    "q": np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])], dtype=float),
                }
            )
    rows.sort(key=lambda item: float(item["t"]))
    deduped = []
    last_t = -math.inf
    for row in rows:
        timestamp = float(row["t"])
        if timestamp <= last_t:
            continue
        deduped.append(row)
        last_t = timestamp
    if len(deduped) < 2:
        raise ValueError(f"{path} has fewer than two strictly increasing poses")
    return deduped


def write_tum(path: Path, rows: list[dict[str, np.ndarray | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            p = np.asarray(row["p"], dtype=float)
            q = np.asarray(row["q"], dtype=float)
            handle.write(
                f"{float(row['t']):.9f} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f} "
                f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
            )


def pose_matrix(row: dict[str, np.ndarray | float]) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = Rotation.from_quat(np.asarray(row["q"], dtype=float)).as_matrix()
    pose[:3, 3] = np.asarray(row["p"], dtype=float)
    return pose


def matrix_to_row(t: float, pose: np.ndarray) -> dict[str, np.ndarray | float]:
    return {
        "t": float(t),
        "p": pose[:3, 3].copy(),
        "q": Rotation.from_matrix(pose[:3, :3]).as_quat(),
    }


def odd_window(n: int, target: int, order: int) -> int:
    if n <= order + 2:
        return max(order + 1 if (order + 1) % 2 else order + 2, 3)
    window = min(target, n if n % 2 else n - 1)
    min_window = order + 2 if (order + 2) % 2 else order + 3
    return max(window, min_window)


def smooth_positions(positions: np.ndarray) -> np.ndarray:
    if positions.shape[0] < 9:
        return positions.copy()
    window = odd_window(positions.shape[0], 21, 3)
    return savgol_filter(positions, window, 3, axis=0, mode="interp")


def load_imu_shifted(path: Path, shift_s: float) -> np.ndarray:
    rows = []
    shift_ns = int(round(shift_s * 1e9))
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split(",")
            if len(parts) < 7:
                continue
            rows.append([int(parts[0]) - shift_ns] + [float(value) for value in parts[1:7]])
    if not rows:
        raise ValueError(f"No IMU rows found in {path}")
    return np.asarray(rows, dtype=float)


def interp_columns(t_src: np.ndarray, values: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    out = np.empty((t_query.shape[0], values.shape[1]), dtype=float)
    for axis in range(values.shape[1]):
        out[:, axis] = np.interp(t_query, t_src, values[:, axis])
    return out


def estimate_gravity_world(body_rows: list[dict[str, np.ndarray | float]], imu_shifted: np.ndarray) -> dict[str, object]:
    t = np.array([float(row["t"]) for row in body_rows], dtype=float)
    p = np.array([row["p"] for row in body_rows], dtype=float)
    rotations = Rotation.from_quat(np.array([row["q"] for row in body_rows], dtype=float)).as_matrix()
    p_smooth = smooth_positions(p)
    v = np.gradient(p_smooth, t, axis=0, edge_order=2)
    a_world = np.gradient(v, t, axis=0, edge_order=2)
    imu_t = imu_shifted[:, 0] * 1e-9
    acc_body = interp_columns(imu_t, imu_shifted[:, 4:7], t)
    # Accelerometer model: a_meas = R_WB^T (a_world - g_world) + b_a.
    # Ignoring mean bias gives g_world ~= a_world - R_WB * a_meas.
    g_samples = a_world - np.einsum("nij,nj->ni", rotations, acc_body)
    g_samples = g_samples[np.all(np.isfinite(g_samples), axis=1)]
    if g_samples.shape[0] < 8:
        return {
            "gravity_median_world_m_s2": [0.0, 0.0, -9.7803],
            "gravity_mean_world_m_s2": [0.0, 0.0, -9.7803],
            "gravity_median_norm": 9.7803,
            "gravity_mean_norm": 9.7803,
            "warning": "gravity estimate had too few finite samples; default gravity direction used",
        }
    trim = max(2, int(0.05 * len(g_samples)))
    if 2 * trim < len(g_samples):
        g_samples = g_samples[trim:-trim]
    g_median = np.median(g_samples, axis=0)
    g_mean = np.mean(g_samples, axis=0)
    return {
        "gravity_median_world_m_s2": g_median.tolist(),
        "gravity_mean_world_m_s2": g_mean.tolist(),
        "gravity_median_norm": float(np.linalg.norm(g_median)),
        "gravity_mean_norm": float(np.linalg.norm(g_mean)),
    }


def rotation_align_vector(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_norm = np.linalg.norm(source)
    target_norm = np.linalg.norm(target)
    if source_norm < 1e-9 or target_norm < 1e-9:
        return np.eye(3, dtype=float)
    a = source / source_norm
    b = target / target_norm
    v = np.cross(a, b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if c > 1.0 - 1e-12:
        return np.eye(3, dtype=float)
    if c < -1.0 + 1e-12:
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.8:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - np.dot(axis, a) * a
        axis = axis / np.linalg.norm(axis)
        return Rotation.from_rotvec(math.pi * axis).as_matrix()
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=float)
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def write_imu_csv(path: Path, imu_shifted: np.ndarray, t_min: float, t_max: float) -> dict[str, object]:
    mask = (imu_shifted[:, 0] * 1e-9 >= t_min) & (imu_shifted[:, 0] * 1e-9 <= t_max)
    cropped = imu_shifted[mask]
    if cropped.shape[0] < 2:
        raise ValueError("IMU/VO overlap is empty after timestamp shift")
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
        for row in cropped:
            writer.writerow([str(int(round(row[0])))] + [f"{value:.12f}" for value in row[1:7]])
    dt = np.diff(cropped[:, 0] * 1e-9)
    return {
        "rows": int(cropped.shape[0]),
        "first_timestamp_ns": int(round(cropped[0, 0])),
        "last_timestamp_ns": int(round(cropped[-1, 0])),
        "median_hz": float(1.0 / np.median(dt)),
    }


def write_pose_csv(path: Path, body_rows: list[dict[str, np.ndarray | float]]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]", "q_RS_w []", "q_RS_x []", "q_RS_y []", "q_RS_z []"])
        for row in body_rows:
            t_ns = int(round(float(row["t"]) * 1e9))
            p = np.asarray(row["p"], dtype=float)
            q = np.asarray(row["q"], dtype=float)
            writer.writerow([str(t_ns), f"{p[0]:.12f}", f"{p[1]:.12f}", f"{p[2]:.12f}", f"{q[3]:.12f}", f"{q[0]:.12f}", f"{q[1]:.12f}", f"{q[2]:.12f}"])
    return {
        "rows": len(body_rows),
        "first_timestamp_ns": int(round(float(body_rows[0]["t"]) * 1e9)),
        "last_timestamp_ns": int(round(float(body_rows[-1]["t"]) * 1e9)),
    }


def export_sequence(args: argparse.Namespace, sequence: str, t_c_b: np.ndarray) -> dict[str, object]:
    seq_dir = args.pseudo_root / sequence
    estimate_camera = seq_dir / "evo_gt_aligned" / "estimate_evo_aligned.tum"
    imu_path = seq_dir / "sequence" / "imu.txt"
    if not estimate_camera.exists():
        raise FileNotFoundError(estimate_camera)
    if not imu_path.exists():
        raise FileNotFoundError(imu_path)

    out_dir = seq_dir / "debias_imu_body_gt_200hz"
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    camera_rows = load_tum(estimate_camera)
    body_unaligned = [matrix_to_row(float(row["t"]), pose_matrix(row) @ t_c_b) for row in camera_rows]
    write_tum(out_dir / "vo_imu_traj_mocap_world_unaligned.tum", body_unaligned)
    imu_shifted = load_imu_shifted(imu_path, args.imu_time_shift_s)
    gravity_record = estimate_gravity_world(body_unaligned, imu_shifted)
    g_source = np.asarray(gravity_record["gravity_median_world_m_s2"], dtype=float)
    target_g = np.array([0.0, 0.0, -float(args.gravity_magnitude)], dtype=float)
    r_align = np.eye(3, dtype=float) if args.no_gravity_align else rotation_align_vector(g_source, target_g)

    body_rows = []
    p0 = None
    for row in body_unaligned:
        pose = pose_matrix(row)
        pose[:3, :3] = r_align @ pose[:3, :3]
        pose[:3, 3] = r_align @ pose[:3, 3]
        if p0 is None:
            p0 = pose[:3, 3].copy()
        pose[:3, 3] -= p0
        body_rows.append(matrix_to_row(float(row["t"]), pose))
    assert p0 is not None
    write_tum(out_dir / "vo_imu_traj.tum", body_rows)

    aggregate_seq = args.aggregate_root / sequence / "mav0"
    per_seq = out_dir / "debias_tumvi_source" / "mav0"
    t_min = float(body_rows[0]["t"])
    t_max = float(body_rows[-1]["t"])
    imu_outputs = {}
    pose_outputs = {}
    for root in [aggregate_seq, per_seq]:
        imu_outputs[str(root)] = write_imu_csv(root / "imu0" / "data.csv", imu_shifted, t_min, t_max)
        pose_outputs[str(root)] = write_pose_csv(root / "mocap0" / "data.csv", body_rows)

    metadata = {
        "dataset": "corldata",
        "sequence": sequence,
        "source_camera_trajectory": str(estimate_camera),
        "source_imu": str(imu_path),
        "vo_imu_traj": str(out_dir / "vo_imu_traj.tum"),
        "frame_convention": "T_W_B = T_W_C_evo_aligned * T_C_B, with T_C_B = inv(IMU.T_b_c1)",
        "world_convention": "gravity-aligned world, initial position subtracted",
        "imu_time_shift_s": args.imu_time_shift_s,
        "imu_time_shift_formula": "t_imu_for_training = t_imu_raw - imu_time_shift_s",
        "T_C_B_camera_to_imu_body": t_c_b.tolist(),
        "R_gravity_alignment": r_align.tolist(),
        "gravity_target_world_m_s2": target_g.tolist(),
        "gravity_estimate_before_alignment": gravity_record,
        "body_pose_count": len(body_rows),
        "imu_outputs": imu_outputs,
        "pose_outputs": pose_outputs,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "sequence": sequence,
        "body_pose_count": len(body_rows),
        "duration_s": t_max - t_min,
        "gravity_est_norm": gravity_record["gravity_median_norm"],
        "aggregate_sequence_dir": str(args.aggregate_root / sequence),
        "body_gt_dir": str(out_dir),
        "imu_rows": imu_outputs[str(aggregate_seq)]["rows"],
    }


def write_summary(root: Path, rows: list[dict[str, object]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "selected_sequences.txt").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row['sequence']}\n")
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# corldata Debias_IMU Body Pseudo-GT Export",
        "",
        f"Aggregate root: `{root}`",
        "",
        "| sequence | poses | duration s | shifted IMU rows | gravity norm before align |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['sequence']} | {row['body_pose_count']} | {float(row['duration_s']):.3f} | "
            f"{row['imu_rows']} | {float(row['gravity_est_norm']):.4f} |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.pseudo_root = args.pseudo_root.resolve()
    args.aggregate_root = args.aggregate_root.resolve()
    args.orb_config = args.orb_config.resolve()
    t_b_c = load_matrix_from_yaml(args.orb_config, "IMU.T_b_c1")
    t_c_b = np.linalg.inv(t_b_c)
    rows = []
    for sequence in args.sequences:
        print(f"[export] {sequence}", flush=True)
        rows.append(export_sequence(args, sequence, t_c_b))
    write_summary(args.aggregate_root, rows)
    print(json.dumps({"aggregate_root": str(args.aggregate_root), "rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
