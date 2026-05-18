#!/usr/bin/env python3
"""Generate corldata right-multiplied SE(3) fit diagnostic trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import run_liasuw_orbslam3_trial as trial


DEFAULT_PLATO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = Path("/home/amao/datasets/corldata/0511_data")
DEFAULT_SEQUENCES = ["S00", "S01", "S02", "S03", "S04", "S05"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plato-root", type=Path, default=DEFAULT_PLATO_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    parser.add_argument("--output-name", default="se3_fit")
    return parser.parse_args()


def result_root(plato_root: Path, sequence: str) -> Path:
    return plato_root / "results" / "pseudo_gt" / "corldata" / sequence


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def array_rmse(values: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean(values * values))))


def rotation_angle_deg(rot: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0))))


def evaluate_sequence(args: argparse.Namespace, sequence: str) -> dict[str, object]:
    seq_root = result_root(args.plato_root, sequence)
    out_dir = seq_root / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    original_metrics = json.loads((seq_root / "metrics.json").read_text(encoding="utf-8"))
    time_offset = float(original_metrics["time_offset_s"])
    max_time_diff = float(original_metrics["max_time_difference_s"])

    gt_rows = trial.load_tum(args.data_root / "gt" / sequence / "Body_0.tum", timestamp_scale="sec")
    est_rows = trial.load_tum(seq_root / "raw" / "estimate_f.txt", timestamp_scale="auto")
    pairs = trial.associate_by_time(gt_rows, est_rows, time_offset, max_time_diff)

    gt_positions = np.array([gt_rows[gt_idx]["p"] for gt_idx, _, _, _ in pairs], dtype=float)
    est_positions = np.array([est_rows[est_idx]["p"] for _, est_idx, _, _ in pairs], dtype=float)
    gt_rotations = np.array([trial.quat_xyzw_to_rot(gt_rows[gt_idx]["q"]) for gt_idx, _, _, _ in pairs], dtype=float)
    est_rotations = np.array([trial.quat_xyzw_to_rot(est_rows[est_idx]["q"]) for _, est_idx, _, _ in pairs], dtype=float)

    fit = trial.fit_world_sensor_se3(gt_positions, est_positions, gt_rotations, est_rotations)
    world_rot = fit["world_rot"]
    world_trans = fit["world_trans"]
    sensor_rot = fit["sensor_rot"]
    sensor_trans = fit["sensor_trans"]

    aligned_rows: list[dict[str, object]] = []
    aligned_est_tum: list[dict[str, object]] = []
    associated_gt_tum: list[dict[str, object]] = []
    associated_est_raw_tum: list[dict[str, object]] = []
    position_errors: list[float] = []
    orientation_errors: list[float] = []
    rpy_errors: list[np.ndarray] = []

    for gt_idx, est_idx, time_diff, _ in pairs:
        gt = gt_rows[gt_idx]
        est = est_rows[est_idx]
        gt_rot = trial.quat_xyzw_to_rot(gt["q"])
        est_rot = trial.quat_xyzw_to_rot(est["q"])
        aligned_pos = world_rot @ (est["p"] + est_rot @ sensor_trans) + world_trans
        aligned_rot = (world_rot @ est_rot) @ sensor_rot
        aligned_quat = trial.rot_to_quat_xyzw(aligned_rot)

        gt_rpy = trial.rot_to_rpy_zyx(gt_rot)
        est_rpy = trial.rot_to_rpy_zyx(aligned_rot)
        rpy_error = np.array([trial.wrap_angle(est_rpy[idx] - gt_rpy[idx]) for idx in range(3)], dtype=float)
        rot_error = aligned_rot.T @ gt_rot
        orientation_error = rotation_angle_deg(rot_error)
        position_error = float(np.linalg.norm(aligned_pos - gt["p"]))

        timestamp = float(gt["t"])
        position_errors.append(position_error)
        orientation_errors.append(orientation_error)
        rpy_errors.append(np.degrees(rpy_error))
        aligned_rows.append(
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
                "position_error_m": position_error,
                "orientation_error_deg": orientation_error,
                "time_difference_s": time_diff,
            }
        )
        aligned_est_tum.append({"t": timestamp, "p": aligned_pos, "q": aligned_quat})
        associated_gt_tum.append({"t": timestamp, "p": gt["p"], "q": gt["q"]})
        associated_est_raw_tum.append({"t": timestamp, "p": est["p"], "q": est["q"]})

    position_errors_array = np.array(position_errors, dtype=float)
    orientation_errors_array = np.array(orientation_errors, dtype=float)
    rpy_errors_array = np.array(rpy_errors, dtype=float)
    raw_orientation_errors_array = np.array(fit["raw_orientation_errors_deg"], dtype=float)

    frame_alignment = {
        "description": "Diagnostic per-sequence right-multiplied SE(3) mapping ORB-SLAM3 left-camera pose into mocap Body_0 pose after no-scale world alignment. This is fitted from the same sequence and is not a calibrated external transform.",
        "model": "T_gt_body ~= T_world_align * T_orb_left_camera * T_camera_to_body_fit",
        "R_camera_to_body_fit": sensor_rot.tolist(),
        "t_camera_to_body_fit": sensor_trans.tolist(),
        "angle_deg": rotation_angle_deg(sensor_rot),
        "translation_norm_m": float(np.linalg.norm(sensor_trans)),
        "world_alignment_rotation": world_rot.tolist(),
        "world_alignment_translation": world_trans.tolist(),
        "translation_lstsq_rank": fit["translation_lstsq_rank"],
        "translation_lstsq_condition": fit["translation_lstsq_condition"],
        "estimated_from_pose_pairs": len(pairs),
    }
    metrics = {
        "dataset": "corldata",
        "sequence": sequence,
        "mode": "ORB-SLAM3 stereo visual-only, per-sequence fitted right SE(3) diagnostic",
        "alignment": "SE(3) no-scale world alignment plus per-sequence fitted right-multiplied camera-to-Body_0 SE(3); diagnostic only, not a calibrated extrinsic",
        "time_offset_s": time_offset,
        "time_offset_source": "existing original automatic time-offset search; not re-optimized for this diagnostic",
        "max_time_difference_s": max_time_diff,
        "associated_pose_count": len(pairs),
        "estimated_pose_count": len(est_rows),
        "position_rmse_m": array_rmse(position_errors_array),
        "position_mean_m": float(np.mean(position_errors_array)),
        "position_median_m": float(np.median(position_errors_array)),
        "position_p90_m": float(np.percentile(position_errors_array, 90)),
        "position_p99_m": float(np.percentile(position_errors_array, 99)),
        "position_max_m": float(np.max(position_errors_array)),
        "orientation_rmse_deg": array_rmse(orientation_errors_array),
        "orientation_mean_deg": float(np.mean(orientation_errors_array)),
        "orientation_max_deg": float(np.max(orientation_errors_array)),
        "orientation_without_sensor_frame_alignment_rmse_deg": array_rmse(raw_orientation_errors_array),
        "orientation_without_sensor_frame_alignment_mean_deg": float(np.mean(raw_orientation_errors_array)),
        "rpy_rmse_deg": {
            "roll": array_rmse(rpy_errors_array[:, 0]),
            "pitch": array_rmse(rpy_errors_array[:, 1]),
            "yaw": array_rmse(rpy_errors_array[:, 2]),
        },
        "sim3_scale_diagnostic": trial.diagnostic_sim3_scale(est_positions, gt_positions),
        "right_se3_frame_alignment": frame_alignment,
        "original_metrics": {
            "position_rmse_m": original_metrics["position_rmse_m"],
            "position_max_m": original_metrics["position_max_m"],
            "orientation_rmse_deg": original_metrics["orientation_rmse_deg"],
            "orientation_max_deg": original_metrics["orientation_max_deg"],
            "associated_pose_count": original_metrics["associated_pose_count"],
        },
    }

    write_csv(out_dir / "aligned_poses.csv", aligned_rows)
    trial.write_tum(out_dir / "estimate_aligned_se3.tum", aligned_est_tum)
    trial.write_tum(out_dir / "groundtruth_associated.tum", associated_gt_tum)
    trial.write_tum(out_dir / "estimate_raw_associated.tum", associated_est_raw_tum)
    (out_dir / "camera_to_body_fit_se3.json").write_text(
        json.dumps(frame_alignment, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def write_aggregate(args: argparse.Namespace, rows: list[dict[str, object]]) -> dict[str, str]:
    root = args.plato_root / "results" / "pseudo_gt" / "corldata"
    prefix = f"S00_S05_{args.output_name}"
    csv_path = root / f"{prefix}_summary.csv"
    json_path = root / f"{prefix}_summary.json"
    report_path = root / f"{prefix}_report.md"

    flat_rows = []
    for row in rows:
        original = row["original_metrics"]
        frame = row["right_se3_frame_alignment"]
        flat_rows.append(
            {
                "sequence": row["sequence"],
                "time_offset_s": row["time_offset_s"],
                "associated_pose_count": row["associated_pose_count"],
                "position_rmse_m_original": original["position_rmse_m"],
                "position_rmse_m_se3_fit": row["position_rmse_m"],
                "position_p90_m_se3_fit": row["position_p90_m"],
                "position_p99_m_se3_fit": row["position_p99_m"],
                "position_max_m_original": original["position_max_m"],
                "position_max_m_se3_fit": row["position_max_m"],
                "orientation_rmse_deg_original": original["orientation_rmse_deg"],
                "orientation_rmse_deg_se3_fit": row["orientation_rmse_deg"],
                "orientation_max_deg_se3_fit": row["orientation_max_deg"],
                "right_se3_translation_norm_m": frame["translation_norm_m"],
                "right_se3_rotation_angle_deg": frame["angle_deg"],
                "right_se3_lstsq_condition": frame["translation_lstsq_condition"],
                "sim3_scale_diagnostic": row["sim3_scale_diagnostic"],
            }
        )
    write_csv(csv_path, flat_rows)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# corldata S00-S05 Per-Sequence Right-SE(3) Fit Diagnostic",
        "",
        "This is a visualization/evaluation diagnostic. It fits a separate right-multiplied `T_camera_to_body_fit` for each sequence using the same sequence's GT, then fits a no-scale world alignment. It is not a calibrated global camera-to-`Body_0` extrinsic.",
        "",
        "| seq | assoc | pos RMSE orig -> SE3 fit m | pos p90/p99/max SE3 fit m | angle RMSE orig -> SE3 fit deg | right t norm m | right R angle deg |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in flat_rows:
        lines.append(
            "| {sequence} | {assoc} | {pos0:.5f} -> {pos1:.5f} | {p90:.5f} / {p99:.5f} / {posmax:.5f} | {ori0:.3f} -> {ori1:.3f} | {tnorm:.4f} | {rangle:.3f} |".format(
                sequence=row["sequence"],
                assoc=row["associated_pose_count"],
                pos0=row["position_rmse_m_original"],
                pos1=row["position_rmse_m_se3_fit"],
                p90=row["position_p90_m_se3_fit"],
                p99=row["position_p99_m_se3_fit"],
                posmax=row["position_max_m_se3_fit"],
                ori0=row["orientation_rmse_deg_original"],
                ori1=row["orientation_rmse_deg_se3_fit"],
                tnorm=row["right_se3_translation_norm_m"],
                rangle=row["right_se3_rotation_angle_deg"],
            )
        )
    lines.extend(
        [
            "",
            "Per-sequence outputs:",
            "",
            f"- `results/pseudo_gt/corldata/<seq>/{args.output_name}/aligned_poses.csv`",
            f"- `results/pseudo_gt/corldata/<seq>/{args.output_name}/metrics.json`",
            f"- `results/pseudo_gt/corldata/<seq>/{args.output_name}/camera_to_body_fit_se3.json`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path), "report": str(report_path)}


def main() -> None:
    args = parse_args()
    args.plato_root = args.plato_root.resolve()
    args.data_root = args.data_root.resolve()

    rows = [evaluate_sequence(args, sequence) for sequence in args.sequences]
    outputs = write_aggregate(args, rows)
    outputs["sequence_count"] = len(rows)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
