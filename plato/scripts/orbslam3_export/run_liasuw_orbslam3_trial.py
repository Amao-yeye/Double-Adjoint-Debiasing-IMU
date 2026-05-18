#!/usr/bin/env python3
"""Run one LIASuw ORB-SLAM3 stereo or stereo-inertial trial and evaluate it.

This script is PLATO glue code. It keeps ORB-SLAM3 outputs under PLATO
results instead of writing experiment artifacts inside the external repo.
"""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


DEFAULT_PLATO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ORB_ROOT = DEFAULT_PLATO_ROOT / "external" / "ORB_SLAM3"
DEFAULT_DATASET = "LIASuw"
DEFAULT_SEQUENCE = "0912s15"
DEFAULT_BAG = Path("/home/amao/datasets/LIASuw/LIASuw_Training/0912event/data/S15.bag")
DEFAULT_GT = Path("/home/amao/datasets/LIASuw/LIASuw_Training/0912event/data/GT/15/mc.tum")
DEFAULT_TIME_OFFSET = -1.665
DEFAULT_MAX_DIFF = 0.03


def mode_terms(mode):
    if mode == "stereo-inertial":
        return {
            "title": "ORB-SLAM3 Stereo-Inertial Trial",
            "mode_label": "stereo-inertial odometry",
            "metrics_mode": "ORB-SLAM3 stereo-inertial odometry",
            "sensor_frame": "IMU/body",
            "sensor_frame_lower": "IMU/body",
            "rotation_key": "R_body_to_gt_frame",
            "rotation_model": "R_gt ~= R_world_align * R_orb_imu_body * R_body_to_gt_frame",
            "transform_model": "T_gt ~= T_world_align * T_orb_imu_body * T_body_to_gt_frame",
            "translation_key": "t_body_to_gt_frame",
            "trajectory_note": (
                "ORB-SLAM3 inertial `SaveTrajectoryEuRoC` writes `Twb`, the IMU/body pose in the ORB world"
            ),
        }
    return {
        "title": "ORB-SLAM3 Stereo Pseudo-GT Trial",
        "mode_label": "stereo visual odometry",
        "metrics_mode": "ORB-SLAM3 stereo visual odometry",
        "sensor_frame": "camera",
        "sensor_frame_lower": "left-camera",
        "rotation_key": "R_camera_to_gt_frame",
        "rotation_model": "R_gt ~= R_world_align * R_orb_left_camera * R_camera_to_gt_frame",
        "transform_model": "T_gt ~= T_world_align * T_orb_left_camera * T_camera_to_gt_frame",
        "translation_key": "t_camera_to_gt_frame",
        "trajectory_note": "ORB-SLAM3 stereo `SaveTrajectoryEuRoC` writes `Twc`, the left camera pose in the ORB world",
    }


def run_command(cmd, cwd=None, log_path=None, env=None):
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    result = subprocess.run(
        [str(item) for item in cmd],
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=process_env,
    )
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code {}:\n{}\nLog: {}".format(
                result.returncode, " ".join(str(item) for item in cmd), log_path
            )
        )
    return result.stdout


def clean_result_paths(paths, reuse_raw):
    dirs_to_clean = ["aligned", "plots", "evo"]
    files_to_clean = ["metrics", "report", "tuning_log", "frame_alignment"]
    if not reuse_raw:
        dirs_to_clean.extend(["sequence", "raw"])
    for key in dirs_to_clean:
        path = paths[key]
        if path.exists():
            shutil.rmtree(path)
    for key in files_to_clean:
        path = paths[key]
        if path.exists():
            path.unlink()


def orbslam_runtime_env(orb_root):
    lib_dirs = [
        orb_root / "lib",
        orb_root / "Thirdparty" / "DBoW2" / "lib",
        orb_root / "Thirdparty" / "g2o" / "lib",
        orb_root / "Thirdparty" / "PangolinInstall" / "lib",
    ]
    existing = [str(path) for path in lib_dirs if path.exists()]
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if current:
        existing.append(current)
    return {"LD_LIBRARY_PATH": ":".join(existing)}


def truncate_sequence_by_duration(sequence_dir, duration_s):
    if not duration_s or duration_s <= 0:
        return None
    timestamps_path = sequence_dir / "timestamps.txt"
    imu_path = sequence_dir / "imu.txt"
    left_dir = sequence_dir / "left"
    right_dir = sequence_dir / "right"
    timestamps = [
        int(line.strip())
        for line in timestamps_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not timestamps:
        raise ValueError("No timestamps found in {}".format(timestamps_path))

    first_ts = timestamps[0]
    max_ts = first_ts + int(duration_s * 1e9)
    kept = [timestamp for timestamp in timestamps if timestamp <= max_ts]
    if len(kept) < 3:
        raise ValueError("Duration {} s keeps only {} frames".format(duration_s, len(kept)))
    keep_names = {"{}.png".format(timestamp) for timestamp in kept}

    for image_dir in [left_dir, right_dir]:
        for image_path in image_dir.glob("*.png"):
            if image_path.name not in keep_names:
                image_path.unlink()

    timestamps_path.write_text("".join("{}\n".format(timestamp) for timestamp in kept), encoding="utf-8")

    if imu_path.exists():
        imu_rows = []
        with imu_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                timestamp = int(stripped.split(",", 1)[0])
                if first_ts <= timestamp <= kept[-1]:
                    imu_rows.append(stripped)
        imu_path.write_text("".join("{}\n".format(row) for row in imu_rows), encoding="utf-8")
    else:
        imu_rows = []

    return {
        "duration_s": float(duration_s),
        "original_stereo_pairs": len(timestamps),
        "kept_stereo_pairs": len(kept),
        "start_timestamp_ns": first_ts,
        "end_timestamp_ns": kept[-1],
        "actual_duration_s": (kept[-1] - first_ts) * 1e-9,
        "kept_imu_measurements": len(imu_rows),
    }


def load_truncation_info(log_path):
    if not log_path.exists():
        return None
    marker = "PLATO truncation:\n"
    text = log_path.read_text(encoding="utf-8")
    if marker not in text:
        return None
    payload = text.split(marker, 1)[1].strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def load_tum(path, timestamp_scale="auto"):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            if len(parts) < 8:
                continue
            timestamp = float(parts[0])
            if timestamp_scale == "auto" and timestamp > 1e12:
                timestamp *= 1e-9
            elif timestamp_scale == "ns":
                timestamp *= 1e-9
            rows.append(
                {
                    "t": timestamp,
                    "p": np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float),
                    "q": normalize_quat(
                        np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])], dtype=float)
                    ),
                }
            )
    rows.sort(key=lambda row: row["t"])
    if not rows:
        raise ValueError("No trajectory rows found in {}".format(path))
    return rows


def write_tum(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            p = row["p"]
            q = row["q"]
            handle.write(
                "{:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f}\n".format(
                    row["t"], p[0], p[1], p[2], q[0], q[1], q[2], q[3]
                )
            )


def normalize_quat(q):
    norm = np.linalg.norm(q)
    if norm <= 0:
        raise ValueError("Invalid zero quaternion")
    return q / norm


def quat_xyzw_to_rot(q):
    x, y, z, w = normalize_quat(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rot_to_quat_xyzw(rot):
    trace = float(np.trace(rot))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s
    return normalize_quat(np.array([qx, qy, qz, qw], dtype=float))


def rot_to_rpy_zyx(rot):
    """Return roll, pitch, yaw in radians using ZYX yaw-pitch-roll convention."""
    sy = math.sqrt(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0])
    singular = sy < 1e-8
    if not singular:
        roll = math.atan2(rot[2, 1], rot[2, 2])
        pitch = math.atan2(-rot[2, 0], sy)
        yaw = math.atan2(rot[1, 0], rot[0, 0])
    else:
        roll = math.atan2(-rot[1, 2], rot[1, 1])
        pitch = math.atan2(-rot[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=float)


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def associate_by_time(gt_rows, est_rows, time_offset, max_diff):
    gt_times = np.array([row["t"] for row in gt_rows], dtype=float)
    pairs = []
    for est_idx, est in enumerate(est_rows):
        shifted_t = est["t"] + time_offset
        insert = int(np.searchsorted(gt_times, shifted_t))
        candidates = []
        if insert < len(gt_times):
            candidates.append(insert)
        if insert > 0:
            candidates.append(insert - 1)
        if not candidates:
            continue
        gt_idx = min(candidates, key=lambda idx: abs(gt_times[idx] - shifted_t))
        diff = abs(gt_times[gt_idx] - shifted_t)
        if diff <= max_diff:
            pairs.append((gt_idx, est_idx, diff, shifted_t))
    if len(pairs) < 3:
        raise ValueError("Only {} timestamp associations found".format(len(pairs)))
    return pairs


def umeyama_se3(src, dst):
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = src_centered.T @ dst_centered / src.shape[0]
    u_mat, _, vh_mat = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(vh_mat.T @ u_mat.T) < 0:
        correction[2, 2] = -1.0
    rot = vh_mat.T @ correction @ u_mat.T
    trans = dst_mean - rot @ src_mean
    return rot, trans


def diagnostic_sim3_scale(src, dst):
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = src_centered.T @ dst_centered / src.shape[0]
    u_mat, singular_values, vh_mat = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(vh_mat.T @ u_mat.T) < 0:
        correction[2, 2] = -1.0
    variance = np.sum(src_centered * src_centered) / src.shape[0]
    return float(np.sum(singular_values * np.diag(correction)) / variance)


def solve_world_and_sensor_translations(est_positions, gt_positions, est_rotations, world_rot):
    """Solve p_gt ~= Rw * (p_est + R_est * t_sensor) + t_world."""
    rows = est_positions.shape[0]
    design = np.zeros((rows * 3, 6), dtype=float)
    rhs = np.zeros(rows * 3, dtype=float)
    for idx in range(rows):
        slc = slice(idx * 3, idx * 3 + 3)
        design[slc, 0:3] = np.eye(3)
        design[slc, 3:6] = world_rot @ est_rotations[idx]
        rhs[slc] = gt_positions[idx] - world_rot @ est_positions[idx]
    solution, _, rank, singular_values = np.linalg.lstsq(design, rhs, rcond=None)
    if len(singular_values) >= 2 and singular_values[-1] > 0:
        condition = float(singular_values[0] / singular_values[-1])
    else:
        condition = math.inf
    return solution[0:3], solution[3:6], int(rank), condition


def fit_world_sensor_se3(gt_positions, est_positions, gt_rotations, est_rotations, iterations=6):
    """Fit T_gt ~= T_world_align * T_est * T_sensor_to_gt with no scale."""
    world_rot, _ = umeyama_se3(est_positions, gt_positions)
    sensor_trans = np.zeros(3, dtype=float)
    world_trans = np.zeros(3, dtype=float)
    rank = 0
    condition = math.inf
    for _ in range(iterations):
        world_trans, sensor_trans, rank, condition = solve_world_and_sensor_translations(
            est_positions, gt_positions, est_rotations, world_rot
        )
        lever_arm_positions = est_positions + np.einsum("nij,j->ni", est_rotations, sensor_trans)
        world_rot, _ = umeyama_se3(lever_arm_positions, gt_positions)
    world_trans, sensor_trans, rank, condition = solve_world_and_sensor_translations(
        est_positions, gt_positions, est_rotations, world_rot
    )

    relative_sensor_rotations = []
    raw_orientation_errors_deg = []
    for gt_rot, est_rot in zip(gt_rotations, est_rotations):
        sensor_rot_i = (world_rot @ est_rot).T @ gt_rot
        relative_sensor_rotations.append(sensor_rot_i)
        raw_orientation_errors_deg.append(rotation_angle_deg(sensor_rot_i))
    sensor_rot = mean_rotation(relative_sensor_rotations)
    sensor_frame_residuals_deg = [
        rotation_angle_deg(sensor_rot.T @ relative_rot) for relative_rot in relative_sensor_rotations
    ]
    return {
        "world_rot": world_rot,
        "world_trans": world_trans,
        "sensor_rot": sensor_rot,
        "sensor_trans": sensor_trans,
        "translation_lstsq_rank": rank,
        "translation_lstsq_condition": condition,
        "raw_orientation_errors_deg": raw_orientation_errors_deg,
        "sensor_frame_residuals_deg": sensor_frame_residuals_deg,
    }


def mean_rotation(rotations):
    if not rotations:
        return np.eye(3)
    matrix = np.sum(rotations, axis=0) / float(len(rotations))
    u_mat, _, vh_mat = np.linalg.svd(matrix)
    rot = u_mat @ vh_mat
    if np.linalg.det(rot) < 0:
        u_mat[:, -1] *= -1.0
        rot = u_mat @ vh_mat
    return rot


def rotation_angle_deg(rot):
    return math.degrees(math.acos(float(np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0))))


def evaluate_and_plot(args, paths):
    gt_rows = load_tum(args.groundtruth, timestamp_scale="sec")
    est_rows = load_tum(paths["raw_estimate"], timestamp_scale="auto")
    write_tum(paths["raw_estimate_seconds"], est_rows)
    shutil.copy2(args.groundtruth, paths["raw_groundtruth"])

    pairs = associate_by_time(gt_rows, est_rows, args.time_offset, args.max_time_diff)
    gt_positions = np.array([gt_rows[gt_idx]["p"] for gt_idx, _, _, _ in pairs], dtype=float)
    est_positions = np.array([est_rows[est_idx]["p"] for _, est_idx, _, _ in pairs], dtype=float)
    sim3_scale = diagnostic_sim3_scale(est_positions, gt_positions)
    gt_rotations = np.array([quat_xyzw_to_rot(gt_rows[gt_idx]["q"]) for gt_idx, _, _, _ in pairs], dtype=float)
    est_rotations = np.array([quat_xyzw_to_rot(est_rows[est_idx]["q"]) for _, est_idx, _, _ in pairs], dtype=float)
    pose_alignment = fit_world_sensor_se3(gt_positions, est_positions, gt_rotations, est_rotations)
    align_rot = pose_alignment["world_rot"]
    align_trans = pose_alignment["world_trans"]
    sensor_frame_rot = pose_alignment["sensor_rot"]
    sensor_frame_trans = pose_alignment["sensor_trans"]
    raw_orientation_errors_deg = pose_alignment["raw_orientation_errors_deg"]
    sensor_frame_residuals_deg = pose_alignment["sensor_frame_residuals_deg"]

    csv_rows = []
    aligned_est_tum = []
    associated_gt_tum = []
    position_errors = []
    orientation_errors_deg = []
    rpy_errors_deg = []
    times = []
    for gt_idx, est_idx, time_diff, shifted_t in pairs:
        gt = gt_rows[gt_idx]
        est = est_rows[est_idx]
        est_rot = quat_xyzw_to_rot(est["q"])
        gt_rot = quat_xyzw_to_rot(gt["q"])
        camera_rot_after_world_alignment = align_rot @ est_rot
        aligned_rot = camera_rot_after_world_alignment @ sensor_frame_rot
        aligned_pos = align_rot @ (est["p"] + est_rot @ sensor_frame_trans) + align_trans
        aligned_quat = rot_to_quat_xyzw(aligned_rot)

        gt_rpy = rot_to_rpy_zyx(gt_rot)
        est_rpy = rot_to_rpy_zyx(aligned_rot)
        rpy_error = np.array([wrap_angle(est_rpy[i] - gt_rpy[i]) for i in range(3)], dtype=float)
        rot_error = aligned_rot.T @ gt_rot
        angle = math.acos(float(np.clip((np.trace(rot_error) - 1.0) * 0.5, -1.0, 1.0)))
        pos_error = float(np.linalg.norm(aligned_pos - gt["p"]))

        timestamp = gt["t"]
        times.append(timestamp)
        position_errors.append(pos_error)
        orientation_errors_deg.append(math.degrees(angle))
        rpy_errors_deg.append(np.degrees(rpy_error))
        csv_rows.append(
            {
                "timestamp": timestamp,
                "est_x": aligned_pos[0],
                "est_y": aligned_pos[1],
                "est_z": aligned_pos[2],
                "est_roll": est_rpy[0],
                "est_pitch": est_rpy[1],
                "est_yaw": est_rpy[2],
                "gt_x": gt["p"][0],
                "gt_y": gt["p"][1],
                "gt_z": gt["p"][2],
                "gt_roll": gt_rpy[0],
                "gt_pitch": gt_rpy[1],
                "gt_yaw": gt_rpy[2],
                "position_error_m": pos_error,
                "orientation_error_deg": math.degrees(angle),
                "time_difference_s": time_diff,
            }
        )
        aligned_est_tum.append({"t": timestamp, "p": aligned_pos, "q": aligned_quat})
        associated_gt_tum.append({"t": timestamp, "p": gt["p"], "q": gt["q"]})

    write_csv(paths["aligned_csv"], csv_rows)
    write_tum(paths["aligned_estimate_tum"], aligned_est_tum)
    write_tum(paths["associated_groundtruth_tum"], associated_gt_tum)

    terms = mode_terms(args.mode)
    frame_alignment = {
        "description": "Constant right-multiplied SE(3) mapping ORB-SLAM3 {} pose into the GT pose frame after no-scale world alignment.".format(
            terms["sensor_frame_lower"]
        ),
        "model": terms["transform_model"],
        terms["rotation_key"]: sensor_frame_rot.tolist(),
        terms["translation_key"]: sensor_frame_trans.tolist(),
        "angle_deg": rotation_angle_deg(sensor_frame_rot),
        "translation_norm_m": float(np.linalg.norm(sensor_frame_trans)),
        "world_alignment_translation": align_trans.tolist(),
        "translation_lstsq_rank": pose_alignment["translation_lstsq_rank"],
        "translation_lstsq_condition": pose_alignment["translation_lstsq_condition"],
        "residual_mean_deg": float(np.mean(sensor_frame_residuals_deg)),
        "residual_rmse_deg": rmse(sensor_frame_residuals_deg),
        "residual_max_deg": float(np.max(sensor_frame_residuals_deg)),
        "estimated_from_pose_pairs": len(pairs),
    }
    paths["frame_alignment"].write_text(json.dumps(frame_alignment, indent=2, sort_keys=True), encoding="utf-8")
    metrics = build_metrics(
        args,
        pairs,
        est_rows,
        position_errors,
        orientation_errors_deg,
        rpy_errors_deg,
        sim3_scale,
        raw_orientation_errors_deg,
        frame_alignment,
    )
    paths["metrics"].write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    make_plots(csv_rows, paths["plots"])
    write_report(args, paths, metrics)
    write_tuning_log(args, paths, metrics)
    return metrics


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write to {}".format(path))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_metrics(
    args,
    pairs,
    est_rows,
    position_errors,
    orientation_errors_deg,
    rpy_errors_deg,
    sim3_scale,
    raw_orientation_errors_deg,
    frame_alignment,
):
    position_errors = np.array(position_errors, dtype=float)
    orientation_errors_deg = np.array(orientation_errors_deg, dtype=float)
    rpy_errors_deg = np.array(rpy_errors_deg, dtype=float)
    raw_orientation_errors_deg = np.array(raw_orientation_errors_deg, dtype=float)
    terms = mode_terms(args.mode)
    return {
        "dataset": args.dataset,
        "sequence": args.sequence,
        "mode": terms["metrics_mode"],
        "alignment": "SE(3) no-scale world alignment plus right-multiplied constant {}-to-GT-frame SE(3) compensation".format(
            terms["sensor_frame"]
        ),
        "time_offset_s": args.time_offset,
        "max_time_difference_s": args.max_time_diff,
        "associated_pose_count": len(pairs),
        "estimated_pose_count": len(est_rows),
        "tracking_drop_count_proxy": max(0, count_timestamps(args.sequence_dir / "timestamps.txt") - len(est_rows)),
        "position_rmse_m": rmse(position_errors),
        "position_mean_m": float(np.mean(position_errors)),
        "position_median_m": float(np.median(position_errors)),
        "position_max_m": float(np.max(position_errors)),
        "orientation_rmse_deg": rmse(orientation_errors_deg),
        "orientation_mean_deg": float(np.mean(orientation_errors_deg)),
        "orientation_max_deg": float(np.max(orientation_errors_deg)),
        "orientation_without_sensor_frame_alignment_rmse_deg": rmse(raw_orientation_errors_deg),
        "orientation_without_sensor_frame_alignment_mean_deg": float(np.mean(raw_orientation_errors_deg)),
        "orientation_frame_alignment": frame_alignment,
        "rpy_rmse_deg": {
            "roll": rmse(rpy_errors_deg[:, 0]),
            "pitch": rmse(rpy_errors_deg[:, 1]),
            "yaw": rmse(rpy_errors_deg[:, 2]),
        },
        "sim3_scale_diagnostic": sim3_scale,
    }


def count_timestamps(path):
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip() and not line.startswith("#"))


def rmse(values):
    values = np.array(values, dtype=float)
    return float(math.sqrt(np.mean(values * values)))


def make_plots(rows, plot_dir):
    plot_dir.mkdir(parents=True, exist_ok=True)
    t0 = rows[0]["timestamp"]
    time = np.array([row["timestamp"] - t0 for row in rows], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for axis, key, label in zip(axes, ["x", "y", "z"], ["x [m]", "y [m]", "z [m]"]):
        axis.plot(time, [row["est_" + key] for row in rows], label="ORB-SLAM3 aligned", linewidth=1.2)
        axis.plot(time, [row["gt_" + key] for row in rows], label="Groundtruth", linewidth=1.0, linestyle="--")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_dir / "position_xyz_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for axis, key, label in zip(axes, ["roll", "pitch", "yaw"], ["roll [deg]", "pitch [deg]", "yaw [deg]"]):
        axis.plot(time, np.degrees([row["est_" + key] for row in rows]), label="ORB-SLAM3 aligned", linewidth=1.2)
        axis.plot(time, np.degrees([row["gt_" + key] for row in rows]), label="Groundtruth", linewidth=1.0, linestyle="--")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_dir / "orientation_rpy_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(6, 1, figsize=(11, 13), sharex=True)
    plot_specs = [
        ("x", "x [m]", False),
        ("y", "y [m]", False),
        ("z", "z [m]", False),
        ("roll", "roll [deg]", True),
        ("pitch", "pitch [deg]", True),
        ("yaw", "yaw [deg]", True),
    ]
    for axis, (key, label, degrees) in zip(axes, plot_specs):
        est = np.array([row["est_" + key] for row in rows], dtype=float)
        gt = np.array([row["gt_" + key] for row in rows], dtype=float)
        if degrees:
            est = np.degrees(est)
            gt = np.degrees(gt)
        axis.plot(time, est, label="ORB-SLAM3 aligned", linewidth=1.1)
        axis.plot(time, gt, label="Groundtruth", linewidth=1.0, linestyle="--")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_dir / "six_dof_comparison.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 7))
    axis.plot([row["gt_x"] for row in rows], [row["gt_y"] for row in rows], label="Groundtruth", linestyle="--")
    axis.plot([row["est_x"] for row in rows], [row["est_y"] for row in rows], label="ORB-SLAM3 aligned")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.axis("equal")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_dir / "trajectory_xy_comparison.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 7))
    axis = fig.add_subplot(111, projection="3d")
    axis.plot([row["gt_x"] for row in rows], [row["gt_y"] for row in rows], [row["gt_z"] for row in rows], label="Groundtruth")
    axis.plot(
        [row["est_x"] for row in rows],
        [row["est_y"] for row in rows],
        [row["est_z"] for row in rows],
        label="ORB-SLAM3 aligned",
    )
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_dir / "trajectory_3d_comparison.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(time, [row["position_error_m"] for row in rows], linewidth=1.2)
    axis.set_xlabel("time [s]")
    axis.set_ylabel("position error [m]")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "position_error.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(time, [row["orientation_error_deg"] for row in rows], linewidth=1.2)
    axis.set_xlabel("time [s]")
    axis.set_ylabel("orientation error [deg]")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "orientation_error.png", dpi=180)
    plt.close(fig)


def run_evo(args, paths):
    evo_executable = Path(args.evo_ape)
    if not evo_executable.exists():
        return {"available": False, "reason": "evo_ape executable not found: {}".format(evo_executable)}
    env = {"MPLBACKEND": "Agg"}
    common = [
        evo_executable,
        "tum",
        paths["raw_groundtruth"],
        paths["raw_estimate_seconds"],
        "-a",
        "--t_offset",
        str(args.time_offset),
        "--t_max_diff",
        str(args.max_time_diff),
        "--no_warnings",
        "--plot_mode",
        "xyz",
    ]
    aligned_common = [
        evo_executable,
        "tum",
        paths["associated_groundtruth_tum"],
        paths["aligned_estimate_tum"],
        "--t_max_diff",
        "0.001",
        "--no_warnings",
        "--plot_mode",
        "xyz",
    ]
    outputs = {}
    commands = [
        (
            "ape_trans_raw_se3",
            common
            + [
                "-r",
                "trans_part",
                "--save_results",
                paths["evo"] / "evo_ape_trans_se3.zip",
                "--save_plot",
                paths["evo"] / "evo_ape_trans_se3.pdf",
                "-v",
            ],
            paths["evo"] / "evo_ape_trans_se3.txt",
        ),
        (
            "ape_rot_deg_raw_se3",
            common
            + [
                "-r",
                "angle_deg",
                "--save_results",
                paths["evo"] / "evo_ape_rot_deg_se3.zip",
                "--save_plot",
                paths["evo"] / "evo_ape_rot_deg_se3.pdf",
                "-v",
            ],
            paths["evo"] / "evo_ape_rot_deg_se3.txt",
        ),
        (
            "ape_trans_frame_aligned",
            aligned_common
            + [
                "-r",
                "trans_part",
                "--save_results",
                paths["evo"] / "evo_ape_trans_frame_aligned.zip",
                "--save_plot",
                paths["evo"] / "evo_ape_trans_frame_aligned.pdf",
                "-v",
            ],
            paths["evo"] / "evo_ape_trans_frame_aligned.txt",
        ),
        (
            "ape_rot_deg_frame_aligned",
            aligned_common
            + [
                "-r",
                "angle_deg",
                "--save_results",
                paths["evo"] / "evo_ape_rot_deg_frame_aligned.zip",
                "--save_plot",
                paths["evo"] / "evo_ape_rot_deg_frame_aligned.pdf",
                "-v",
            ],
            paths["evo"] / "evo_ape_rot_deg_frame_aligned.txt",
        ),
    ]
    for name, command, log_path in commands:
        run_command(command, log_path=log_path, env=env)
        outputs[name] = str(log_path)
    return {"available": True, "outputs": outputs}


def write_report(args, paths, metrics):
    terms = mode_terms(args.mode)
    metric_values = dict(metrics)
    metric_values.pop("dataset", None)
    metric_values.pop("sequence", None)
    metric_values.pop("mode", None)
    metric_values.pop("alignment", None)
    frame_alignment = metric_values.pop("orientation_frame_alignment")
    truncation_note = "No sequence truncation was requested."
    if args.truncation_info:
        truncation_note = (
            "Sequence was truncated to the first {duration_s:.3f} s after export, keeping "
            "{kept_stereo_pairs} of {original_stereo_pairs} stereo pairs and {kept_imu_measurements} "
            "IMU measurements. Actual image timestamp span is {actual_duration_s:.6f} s."
        ).format(**args.truncation_info)
    report = """# {title}: {dataset} {sequence}

## Dataset

- Dataset: `{dataset}`
- Sequence: `{sequence}`
- Rosbag: `{bag}`
- Groundtruth: `{groundtruth}`
- Truncation: {truncation_note}

## ORB-SLAM3

- Mode: {mode_label}
- Executable: `{executable}`
- Config: `{config}`
- Vocabulary: `{vocabulary}`
- Time offset applied for association: `{time_offset}` s
- Maximum timestamp association difference: `{max_time_diff}` s

## Outputs

- Raw full-frame estimate: `{raw_estimate}`
- Raw keyframe estimate: `{raw_keyframes}`
- Raw estimate converted to seconds: `{raw_estimate_seconds}`
- Associated aligned CSV: `{aligned_csv}`
- Aligned estimate TUM: `{aligned_estimate_tum}`
- Associated groundtruth TUM: `{associated_groundtruth_tum}`
- Estimated {sensor_frame}-to-GT-frame SE(3): `{frame_alignment_path}`
- Plots: `{plots}`
- Evo results: `{evo}`
- Metrics JSON: `{metrics_path}`

## Metrics

- Associated poses: `{associated_pose_count}`
- Estimated poses: `{estimated_pose_count}`
- Tracking drop count proxy: `{tracking_drop_count_proxy}`
- Position RMSE: `{position_rmse_m:.6f}` m
- Position mean: `{position_mean_m:.6f}` m
- Position max: `{position_max_m:.6f}` m
- Orientation RMSE after {sensor_frame}-to-GT-frame SE(3): `{orientation_rmse_deg:.6f}` deg
- Orientation mean after {sensor_frame}-to-GT-frame SE(3): `{orientation_mean_deg:.6f}` deg
- Orientation max after {sensor_frame}-to-GT-frame SE(3): `{orientation_max_deg:.6f}` deg
- Orientation RMSE before {sensor_frame}-to-GT-frame rotation: `{orientation_without_sensor_frame_alignment_rmse_deg:.6f}` deg
- Estimated fixed {sensor_frame}-to-GT-frame rotation angle: `{frame_alignment_angle_deg:.6f}` deg
- Estimated fixed {sensor_frame}-to-GT-frame translation norm: `{frame_alignment_translation_norm_m:.6f}` m
- Fixed-SE(3) orientation residual RMSE: `{frame_alignment_residual_rmse_deg:.6f}` deg
- Sim(3) diagnostic scale, not applied: `{sim3_scale_diagnostic:.6f}`

## Conclusion

This trial uses a `{time_offset}` s timestamp offset for association, SE(3) world alignment without scale correction, and a right-multiplied constant SE(3) transform that maps ORB-SLAM3 {sensor_frame_lower} poses into the GT pose frame. {trajectory_note}; the GT pose file may use a different body or mocap rigid-body pose frame from the ORB-SLAM3 sensor frame. The estimated fixed transform resolves the large pose-frame offset for this sequence, but it should be replaced by a calibrated {sensor_frame}-to-GT-body transform when that calibration is identified.
""".format(
        title=terms["title"],
        dataset=args.dataset,
        sequence=args.sequence,
        bag=args.bag,
        groundtruth=args.groundtruth,
        truncation_note=truncation_note,
        executable=args.executable,
        config=args.config,
        vocabulary=args.vocabulary,
        mode_label=terms["mode_label"],
        sensor_frame=terms["sensor_frame"],
        sensor_frame_lower=terms["sensor_frame_lower"],
        trajectory_note=terms["trajectory_note"],
        time_offset=args.time_offset,
        max_time_diff=args.max_time_diff,
        raw_estimate=paths["raw_estimate"],
        raw_keyframes=paths["raw_keyframes"],
        raw_estimate_seconds=paths["raw_estimate_seconds"],
        aligned_csv=paths["aligned_csv"],
        aligned_estimate_tum=paths["aligned_estimate_tum"],
        associated_groundtruth_tum=paths["associated_groundtruth_tum"],
        frame_alignment_path=paths["frame_alignment"],
        plots=paths["plots"],
        evo=paths["evo"],
        metrics_path=paths["metrics"],
        frame_alignment_angle_deg=frame_alignment["angle_deg"],
        frame_alignment_translation_norm_m=frame_alignment["translation_norm_m"],
        frame_alignment_residual_rmse_deg=frame_alignment["residual_rmse_deg"],
        **metric_values
    )
    paths["report"].write_text(report, encoding="utf-8")


def write_tuning_log(args, paths, metrics):
    terms = mode_terms(args.mode)
    metric_values = dict(metrics)
    metric_values.pop("dataset", None)
    metric_values.pop("sequence", None)
    metric_values.pop("mode", None)
    metric_values.pop("alignment", None)
    truncation_note = "No sequence truncation was requested."
    if args.truncation_info:
        truncation_note = (
            "Sequence was truncated to the first {duration_s:.3f} s after export, keeping "
            "{kept_stereo_pairs} of {original_stereo_pairs} stereo pairs and {kept_imu_measurements} "
            "IMU measurements. Actual image timestamp span is {actual_duration_s:.6f} s."
        ).format(**args.truncation_info)
    text = """# Tuning Log: {dataset} {sequence}

## Config Version

- Config used: `{config}`
- Mode: `{mode_label}`

## Sequence Truncation

{truncation_note}

## Attempt 1

- Controlled setting: run `{mode_label}` with the config above.
- Reason: evaluate this single ORB-SLAM3 configuration with the same association and alignment procedure.
- Tracking behavior proxy: {estimated_pose_count} estimated poses, {tracking_drop_count_proxy} dropped-frame proxy count.
- Position RMSE after SE(3) alignment: {position_rmse_m:.6f} m.
- Orientation RMSE after SE(3) alignment: {orientation_rmse_deg:.6f} deg.

## Final Selected Setting For This Trial

Use `{config}` for this candidate unless later comparisons show a clear improvement from controlled changes to ORB feature count, FAST thresholds, stereo depth threshold, or far point filtering.
""".format(
        dataset=args.dataset,
        sequence=args.sequence,
        config=args.config,
        mode_label=terms["mode_label"],
        truncation_note=truncation_note,
        **metric_values
    )
    paths["tuning_log"].write_text(text, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plato-root", type=Path, default=DEFAULT_PLATO_ROOT)
    parser.add_argument("--orb-root", type=Path, default=DEFAULT_ORB_ROOT)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    parser.add_argument("--bag", type=Path, default=DEFAULT_BAG)
    parser.add_argument("--groundtruth", type=Path, default=DEFAULT_GT)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="ORB-SLAM3 YAML config; defaults to external/ORB_SLAM3/Examples/Stereo/LIASuw_image_robust.yaml",
    )
    parser.add_argument(
        "--input-sequence-dir",
        type=Path,
        default=None,
        help="reuse an already exported image sequence directory for ORB-SLAM3, but regenerate raw/eval outputs",
    )
    parser.add_argument(
        "--mode",
        choices=["stereo", "stereo-inertial"],
        default="stereo",
        help="ORB-SLAM3 example mode to run. stereo-inertial expects an imu.txt in the sequence directory.",
    )
    parser.add_argument("--time-offset", type=float, default=DEFAULT_TIME_OFFSET)
    parser.add_argument("--max-time-diff", type=float, default=DEFAULT_MAX_DIFF)
    parser.add_argument("--run-name", default="plato_liasuw_0912s15_stereo")
    parser.add_argument("--evo-ape", default="/home/amao/anaconda3/bin/evo_ape")
    parser.add_argument("--force", action="store_true", help="overwrite existing result files for this sequence")
    parser.add_argument("--reuse-raw", action="store_true", help="reuse existing raw estimate files and skip export/ORB-SLAM3")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="truncate exported sequence to the first N seconds before running ORB-SLAM3",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.plato_root = args.plato_root.resolve()
    args.orb_root = args.orb_root.resolve()
    args.bag = args.bag.resolve()
    args.groundtruth = args.groundtruth.resolve()
    if args.config:
        args.config = args.config.resolve()
    if args.input_sequence_dir:
        args.input_sequence_dir = args.input_sequence_dir.resolve()
    if args.mode == "stereo-inertial":
        args.executable = args.orb_root / "Examples" / "Stereo-Inertial" / "stereo_inertial_tum_vi"
    else:
        args.executable = args.orb_root / "Examples" / "Stereo" / "stereo_tum_vi_batch"
    args.vocabulary = args.orb_root / "Vocabulary" / "ORBvoc.txt"
    if not args.config:
        args.config = args.orb_root / "Examples" / "Stereo" / "LIASuw_image_robust.yaml"

    for path in [args.bag, args.groundtruth, args.executable, args.vocabulary, args.config]:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.input_sequence_dir:
        for path in [
            args.input_sequence_dir / "left",
            args.input_sequence_dir / "right",
            args.input_sequence_dir / "timestamps.txt",
        ]:
            if not path.exists():
                raise FileNotFoundError(path)
        if args.mode == "stereo-inertial" and not (args.input_sequence_dir / "imu.txt").exists():
            raise FileNotFoundError(args.input_sequence_dir / "imu.txt")

    result_dir = args.plato_root / "results" / "pseudo_gt" / args.dataset / args.sequence
    paths = {
        "result": result_dir,
        "sequence": result_dir / "sequence",
        "raw": result_dir / "raw",
        "aligned": result_dir / "aligned",
        "plots": result_dir / "plots",
        "evo": result_dir / "evo",
        "metrics": result_dir / "metrics.json",
        "report": result_dir / "report.md",
        "tuning_log": result_dir / "tuning_log.md",
    }
    paths["raw_estimate"] = paths["raw"] / "estimate_f.txt"
    paths["raw_keyframes"] = paths["raw"] / "estimate_kf.txt"
    paths["raw_estimate_seconds"] = paths["raw"] / "estimate_f_seconds.tum"
    paths["raw_groundtruth"] = paths["raw"] / "groundtruth.tum"
    paths["aligned_csv"] = paths["aligned"] / "aligned_poses.csv"
    paths["aligned_estimate_tum"] = paths["aligned"] / "estimate_aligned_se3.tum"
    paths["associated_groundtruth_tum"] = paths["aligned"] / "groundtruth_associated.tum"
    paths["frame_alignment"] = paths["aligned"] / "camera_to_gt_frame_rotation.json"

    if result_dir.exists() and any(result_dir.iterdir()):
        if not args.force:
            raise RuntimeError("Result directory already exists and is not empty: {}. Use --force.".format(result_dir))
        clean_result_paths(paths, args.reuse_raw)
    for key in ["sequence", "raw", "aligned", "plots", "evo"]:
        paths[key].mkdir(parents=True, exist_ok=True)

    args.sequence_dir = paths["sequence"]
    args.truncation_info = None
    if args.reuse_raw:
        if args.input_sequence_dir:
            raise ValueError("--reuse-raw cannot be combined with --input-sequence-dir")
        if args.duration:
            raise ValueError("--duration cannot be used with --reuse-raw because raw ORB output would not match")
        for path in [paths["raw_estimate"], paths["raw_keyframes"], paths["sequence"] / "timestamps.txt"]:
            if not path.exists():
                raise FileNotFoundError("Cannot reuse raw output; missing {}".format(path))
        args.truncation_info = load_truncation_info(paths["raw"] / "export.log")
    else:
        export_log = paths["raw"] / "export.log"
        if args.input_sequence_dir:
            args.truncation_info = load_truncation_info(args.input_sequence_dir.parent / "raw" / "export.log")
            shutil.copy2(args.input_sequence_dir / "timestamps.txt", paths["sequence"] / "timestamps.txt")
            imu_path = args.input_sequence_dir / "imu.txt"
            if imu_path.exists():
                shutil.copy2(imu_path, paths["sequence"] / "imu.txt")
            export_note = {
                "input_sequence_dir": str(args.input_sequence_dir),
                "note": "Reused an already exported image sequence; left/right PNGs were not copied.",
                "truncation_info": args.truncation_info,
            }
            export_log.write_text(json.dumps(export_note, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            run_command(
                [args.orb_root / "tools" / "export_liasuw_bag.py", args.bag, paths["sequence"]],
                log_path=export_log,
            )
            args.truncation_info = truncate_sequence_by_duration(paths["sequence"], args.duration)
            if args.truncation_info:
                with export_log.open("a", encoding="utf-8") as handle:
                    handle.write("\nPLATO truncation:\n")
                    handle.write(json.dumps(args.truncation_info, indent=2, sort_keys=True))
                    handle.write("\n")

        orb_log = paths["raw"] / "orbslam3.log"
        generated_f = args.orb_root / ("f_" + args.run_name + ".txt")
        generated_kf = args.orb_root / ("kf_" + args.run_name + ".txt")
        orb_sequence_dir = args.input_sequence_dir if args.input_sequence_dir else paths["sequence"]
        for generated in [generated_f, generated_kf]:
            if generated.exists():
                generated.unlink()
        run_command(
            [
                args.executable,
                args.vocabulary,
                args.config,
                orb_sequence_dir / "left",
                orb_sequence_dir / "right",
                orb_sequence_dir / "timestamps.txt",
            ]
            + ([orb_sequence_dir / "imu.txt"] if args.mode == "stereo-inertial" else [])
            + [args.run_name],
            cwd=args.orb_root,
            log_path=orb_log,
            env=orbslam_runtime_env(args.orb_root),
        )
        if not generated_f.exists() or not generated_kf.exists():
            raise FileNotFoundError("ORB-SLAM3 did not produce expected f_/kf_ trajectory files")
        shutil.move(str(generated_f), paths["raw_estimate"])
        shutil.move(str(generated_kf), paths["raw_keyframes"])

    metrics = evaluate_and_plot(args, paths)
    evo_status = run_evo(args, paths)
    (paths["evo"] / "evo_status.json").write_text(json.dumps(evo_status, indent=2), encoding="utf-8")
    print(json.dumps({"result_dir": str(result_dir), "metrics": metrics, "evo": evo_status}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)
