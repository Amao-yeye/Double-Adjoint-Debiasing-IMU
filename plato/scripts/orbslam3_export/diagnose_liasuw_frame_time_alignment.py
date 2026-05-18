#!/usr/bin/env python3
"""Diagnose LIASuw ORB-SLAM3/GT time and frame alignment.

This script reuses an existing ORB-SLAM3 trajectory and mocap TUM file. It does
not run ORB-SLAM3. The goal is to separate timestamp offset, GT quaternion
direction, and camera/body frame effects before treating a trajectory as
pseudo-groundtruth.
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SEQUENCE_DIR = DEFAULT_ROOT / "results" / "pseudo_gt" / "LIASuw" / "0912s15"
DEFAULT_ESTIMATE = DEFAULT_SEQUENCE_DIR / "raw" / "estimate_f.txt"
DEFAULT_GT = DEFAULT_SEQUENCE_DIR / "raw" / "groundtruth.tum"
DEFAULT_CAM_IMU = Path(
    "/home/amao/datasets/LIASuw/LIASuw_Training/0912event/Calibration files/cam-imu.yaml"
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


def invert_quat_xyzw(q):
    q = normalize_quat(q)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)


def rotation_angle_deg(rot):
    value = float(np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(value))


def load_tum(path, timestamp_scale="auto"):
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
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
        raise ValueError("No TUM rows found in {}".format(path))
    return rows


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
            pairs.append((gt_idx, est_idx, diff))
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


def mean_rotation(rotations):
    matrix = np.sum(rotations, axis=0) / float(len(rotations))
    u_mat, _, vh_mat = np.linalg.svd(matrix)
    rot = u_mat @ vh_mat
    if np.linalg.det(rot) < 0:
        u_mat[:, -1] *= -1.0
        rot = u_mat @ vh_mat
    return rot


def rmse(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values * values)))


def load_rotation_from_yaml_matrix(path, key):
    text = Path(path).read_text(encoding="utf-8")
    match = re.search(r"{}\s*:.*?\n\s*\[\[(.*?)\]\]".format(re.escape(key)), text, flags=re.S)
    if not match:
        raise ValueError("Could not find matrix {} in {}".format(key, path))
    numbers = [float(item) for item in re.findall(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", match.group(1))]
    if len(numbers) < 16:
        raise ValueError("Matrix {} in {} has only {} numeric values".format(key, path, len(numbers)))
    return np.array(numbers[:16], dtype=float).reshape(4, 4)[:3, :3]


def evaluate_case(gt_rows, est_rows, pairs, gt_quat_mode, est_frame_mode, cam0_to_imu_rot, fit_constant_frame):
    gt_positions = np.array([gt_rows[gt_idx]["p"] for gt_idx, _, _ in pairs], dtype=float)
    est_positions = np.array([est_rows[est_idx]["p"] for _, est_idx, _ in pairs], dtype=float)
    world_rot, world_trans = umeyama_se3(est_positions, gt_positions)
    del world_trans

    relative_rotations = []
    raw_errors = []
    for gt_idx, est_idx, _ in pairs:
        gt_q = gt_rows[gt_idx]["q"]
        if gt_quat_mode == "inverse":
            gt_q = invert_quat_xyzw(gt_q)
        gt_rot = quat_xyzw_to_rot(gt_q)
        est_rot = quat_xyzw_to_rot(est_rows[est_idx]["q"])
        if est_frame_mode == "imu":
            est_rot = est_rot @ cam0_to_imu_rot
        est_rot = world_rot @ est_rot
        relative_rot = est_rot.T @ gt_rot
        relative_rotations.append(relative_rot)
        raw_errors.append(rotation_angle_deg(relative_rot))

    if fit_constant_frame:
        frame_rot = mean_rotation(relative_rotations)
        errors = [rotation_angle_deg(frame_rot.T @ relative_rot) for relative_rot in relative_rotations]
        frame_angle = rotation_angle_deg(frame_rot)
    else:
        errors = raw_errors
        frame_angle = 0.0

    return {
        "count": len(pairs),
        "orientation_rmse_deg": rmse(errors),
        "orientation_mean_deg": float(np.mean(errors)),
        "orientation_max_deg": float(np.max(errors)),
        "raw_orientation_rmse_deg": rmse(raw_errors),
        "raw_orientation_mean_deg": float(np.mean(raw_errors)),
        "fitted_constant_frame_angle_deg": frame_angle,
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path, args, best, rows, cam0_to_imu_rot):
    lines = [
        "# LIASuw frame/time alignment diagnosis",
        "",
        "This diagnostic reuses an existing ORB-SLAM3 stereo trajectory and does not rerun ORB-SLAM3.",
        "",
        "## Inputs",
        "",
        "- Estimate: `{}`".format(args.estimate),
        "- Groundtruth: `{}`".format(args.groundtruth),
        "- Camera-IMU calibration: `{}`".format(args.cam_imu_yaml),
        "- Maximum association difference: `{:.6f}` s".format(args.max_diff),
        "",
        "## Best case",
        "",
        "- Time offset: `{:.6f}` s".format(best["time_offset_s"]),
        "- GT quaternion mode: `{}`".format(best["gt_quaternion_mode"]),
        "- ORB-SLAM3 pose frame mode: `{}`".format(best["est_frame_mode"]),
        "- Constant frame rotation fitted: `{}`".format(best["fit_constant_frame_rotation"]),
        "- Associated poses: `{}`".format(best["associated_pose_count"]),
        "- Orientation RMSE: `{:.6f}` deg".format(best["orientation_rmse_deg"]),
        "- Orientation mean: `{:.6f}` deg".format(best["orientation_mean_deg"]),
        "- Orientation max: `{:.6f}` deg".format(best["orientation_max_deg"]),
        "- Raw orientation RMSE before fitting constant frame rotation: `{:.6f}` deg".format(
            best["raw_orientation_rmse_deg"]
        ),
        "- Fitted constant frame rotation angle: `{:.6f}` deg".format(
            best["fitted_constant_frame_angle_deg"]
        ),
        "",
        "## Interpretation",
        "",
        "- Standard TUM columns are assumed: `timestamp tx ty tz qx qy qz qw`.",
        "- `gt_quaternion_mode=inverse` tests whether the mocap quaternion needs to be inverted.",
        "- `est_frame_mode=imu` applies the `cam0 to imu0` rotation from `cam-imu.yaml` to ORB-SLAM3 left-camera orientation.",
        "- `fit_constant_frame_rotation=true` estimates the remaining fixed rotation between the ORB-SLAM3 pose frame and the mocap `Body_0` frame. This is diagnostic, not a replacement for a real camera/body/mocap extrinsic calibration.",
        "",
        "## Camera to IMU rotation used",
        "",
        "```text",
        np.array2string(cam0_to_imu_rot, precision=8, suppress_small=False),
        "```",
        "",
        "## Top cases",
        "",
    ]
    for row in rows[:12]:
        lines.append(
            "- offset={time_offset_s:.6f}, gt={gt_quaternion_mode}, est={est_frame_mode}, "
            "fit={fit_constant_frame_rotation}, rmse={orientation_rmse_deg:.6f} deg, "
            "raw_rmse={raw_orientation_rmse_deg:.6f} deg, count={associated_pose_count}".format(**row)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate", type=Path, default=DEFAULT_ESTIMATE)
    parser.add_argument("--groundtruth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--cam-imu-yaml", type=Path, default=DEFAULT_CAM_IMU)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SEQUENCE_DIR / "diagnostics" / "frame_time")
    parser.add_argument("--min-offset", type=float, default=-1.7)
    parser.add_argument("--max-offset", type=float, default=-1.3)
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument("--max-diff", type=float, default=0.03)
    return parser.parse_args()


def main():
    args = parse_args()
    gt_rows = load_tum(args.groundtruth, timestamp_scale="sec")
    est_rows = load_tum(args.estimate, timestamp_scale="auto")
    cam0_to_imu_rot = load_rotation_from_yaml_matrix(args.cam_imu_yaml, "T_ic")

    rows = []
    offset_count = int(round((args.max_offset - args.min_offset) / args.step)) + 1
    offsets = [args.min_offset + args.step * idx for idx in range(offset_count)]
    for offset in offsets:
        pairs = associate_by_time(gt_rows, est_rows, offset, args.max_diff)
        if len(pairs) < 3:
            continue
        for gt_quat_mode in ["as_is", "inverse"]:
            for est_frame_mode in ["camera", "imu"]:
                for fit_constant_frame in [False, True]:
                    metrics = evaluate_case(
                        gt_rows,
                        est_rows,
                        pairs,
                        gt_quat_mode,
                        est_frame_mode,
                        cam0_to_imu_rot,
                        fit_constant_frame,
                    )
                    rows.append(
                        {
                            "time_offset_s": float(offset),
                            "gt_quaternion_mode": gt_quat_mode,
                            "est_frame_mode": est_frame_mode,
                            "fit_constant_frame_rotation": fit_constant_frame,
                            "associated_pose_count": metrics["count"],
                            "orientation_rmse_deg": metrics["orientation_rmse_deg"],
                            "orientation_mean_deg": metrics["orientation_mean_deg"],
                            "orientation_max_deg": metrics["orientation_max_deg"],
                            "raw_orientation_rmse_deg": metrics["raw_orientation_rmse_deg"],
                            "raw_orientation_mean_deg": metrics["raw_orientation_mean_deg"],
                            "fitted_constant_frame_angle_deg": metrics["fitted_constant_frame_angle_deg"],
                        }
                    )

    if not rows:
        raise RuntimeError("No timestamp associations were found in the requested offset range")
    rows.sort(key=lambda row: (row["orientation_rmse_deg"], -row["associated_pose_count"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "alignment_diagnostic_sweep.csv", rows)
    (args.output_dir / "best_alignment_case.json").write_text(
        json.dumps(rows[0], indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report(args.output_dir / "README.md", args, rows[0], rows, cam0_to_imu_rot)
    print(json.dumps(rows[0], indent=2, sort_keys=True))
    print("Wrote diagnostics to {}".format(args.output_dir))


if __name__ == "__main__":
    main()
