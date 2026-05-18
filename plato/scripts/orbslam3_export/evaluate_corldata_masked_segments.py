#!/usr/bin/env python3
"""Evaluate corldata after masking suspicious GT jump/problem segments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_liasuw_orbslam3_trial as trial


DEFAULT_PLATO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = Path("/home/amao/datasets/corldata/0511_data")
DEFAULT_SOURCES = [
    "gt_small_dt",
    "gt_large_dt",
    "gt_position_step",
    "gt_speed_spike",
    "gt_angle_step",
    "gt_angular_speed",
    "gt_accel_spike",
    "residual_position",
    "residual_orientation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plato-root", type=Path, default=DEFAULT_PLATO_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sequences", nargs="+", default=["S00", "S01", "S02", "S03", "S04", "S05"])
    parser.add_argument(
        "--segments-csv",
        type=Path,
        default=DEFAULT_PLATO_ROOT
        / "results"
        / "pseudo_gt"
        / "corldata"
        / "gt_quality"
        / "suspicious_segments.csv",
    )
    parser.add_argument(
        "--mask-sources",
        nargs="+",
        default=DEFAULT_SOURCES,
        help="Suspicious segment sources to mask. Defaults to GT anomalies plus residual spikes.",
    )
    parser.add_argument("--padding-s", type=float, default=0.20)
    parser.add_argument("--output-name", default="masked_gt_jumps")
    return parser.parse_args()


def result_root(plato_root: Path, sequence: str) -> Path:
    return plato_root / "results" / "pseudo_gt" / "corldata" / sequence


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted((min(a, b), max(a, b)) for a, b in intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def load_mask_intervals(path: Path, sequence: str, sources: set[str], padding_s: float) -> tuple[list[tuple[float, float]], list[dict[str, object]]]:
    intervals: list[tuple[float, float]] = []
    selected_rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["sequence"] != sequence or row["source"] not in sources:
                continue
            start = float(row["start_time_s"]) - padding_s
            end = float(row["end_time_s"]) + padding_s
            intervals.append((start, end))
            selected_rows.append(row)
    return merge_intervals(intervals), selected_rows


def is_masked(timestamp: float, intervals: list[tuple[float, float]]) -> bool:
    # Intervals are few; linear scan keeps the logic explicit.
    for start, end in intervals:
        if start <= timestamp <= end:
            return True
    return False


def rotation_angle_deg(rot: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0))))


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


def evaluate_sequence(args: argparse.Namespace, sequence: str) -> dict[str, object]:
    seq_root = result_root(args.plato_root, sequence)
    out_dir = seq_root / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    original_metrics = json.loads((seq_root / "metrics.json").read_text(encoding="utf-8"))
    time_offset = float(original_metrics["time_offset_s"])
    max_time_diff = float(original_metrics["max_time_difference_s"])

    intervals, selected_segments = load_mask_intervals(
        args.segments_csv,
        sequence,
        set(args.mask_sources),
        args.padding_s,
    )
    interval_rows = [
        {
            "sequence": sequence,
            "start_time_s": start,
            "end_time_s": end,
            "duration_s": max(0.0, end - start),
        }
        for start, end in intervals
    ]
    write_csv(out_dir / "mask_intervals.csv", interval_rows)
    write_csv(out_dir / "mask_source_segments.csv", selected_segments)

    gt_path = args.data_root / "gt" / sequence / "Body_0.tum"
    est_path = seq_root / "raw" / "estimate_f.txt"
    gt_rows = trial.load_tum(gt_path, timestamp_scale="sec")
    est_rows = trial.load_tum(est_path, timestamp_scale="auto")
    pairs_all = trial.associate_by_time(gt_rows, est_rows, time_offset, max_time_diff)
    pairs = [
        pair
        for pair in pairs_all
        if not is_masked(float(gt_rows[pair[0]]["t"]), intervals)
    ]
    if len(pairs) < 10:
        raise RuntimeError(f"{sequence}: mask leaves too few associated poses ({len(pairs)})")

    gt_positions = np.array([gt_rows[gt_idx]["p"] for gt_idx, _, _, _ in pairs], dtype=float)
    est_positions = np.array([est_rows[est_idx]["p"] for _, est_idx, _, _ in pairs], dtype=float)
    sim3_scale = trial.diagnostic_sim3_scale(est_positions, gt_positions)
    gt_rotations = np.array([trial.quat_xyzw_to_rot(gt_rows[gt_idx]["q"]) for gt_idx, _, _, _ in pairs], dtype=float)
    est_rotations = np.array([trial.quat_xyzw_to_rot(est_rows[est_idx]["q"]) for _, est_idx, _, _ in pairs], dtype=float)
    pose_alignment = trial.fit_world_sensor_se3(gt_positions, est_positions, gt_rotations, est_rotations)
    align_rot = pose_alignment["world_rot"]
    align_trans = pose_alignment["world_trans"]
    sensor_frame_rot = pose_alignment["sensor_rot"]
    sensor_frame_trans = pose_alignment["sensor_trans"]
    raw_orientation_errors_deg = pose_alignment["raw_orientation_errors_deg"]

    aligned_rows: list[dict[str, object]] = []
    aligned_est_tum: list[dict[str, object]] = []
    associated_gt_tum: list[dict[str, object]] = []
    associated_est_raw_tum: list[dict[str, object]] = []
    position_errors: list[float] = []
    orientation_errors: list[float] = []
    rpy_errors: list[np.ndarray] = []

    for gt_idx, est_idx, time_diff, shifted_t in pairs:
        gt = gt_rows[gt_idx]
        est = est_rows[est_idx]
        gt_rot = trial.quat_xyzw_to_rot(gt["q"])
        est_rot = trial.quat_xyzw_to_rot(est["q"])
        aligned_pos = align_rot @ (est["p"] + est_rot @ sensor_frame_trans) + align_trans
        aligned_rot = (align_rot @ est_rot) @ sensor_frame_rot
        aligned_quat = trial.rot_to_quat_xyzw(aligned_rot)

        gt_rpy = trial.rot_to_rpy_zyx(gt_rot)
        est_rpy = trial.rot_to_rpy_zyx(aligned_rot)
        rpy_error = np.array([trial.wrap_angle(est_rpy[i] - gt_rpy[i]) for i in range(3)], dtype=float)
        rot_error = aligned_rot.T @ gt_rot
        orientation_error = rotation_angle_deg(rot_error)
        position_error = float(np.linalg.norm(aligned_pos - gt["p"]))

        position_errors.append(position_error)
        orientation_errors.append(orientation_error)
        rpy_errors.append(np.degrees(rpy_error))
        timestamp = float(gt["t"])
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

    rpy_errors_array = np.array(rpy_errors, dtype=float)
    position_errors_array = np.array(position_errors, dtype=float)
    orientation_errors_array = np.array(orientation_errors, dtype=float)
    raw_orientation_errors_array = np.array(raw_orientation_errors_deg, dtype=float)

    frame_alignment = {
        "description": "Constant right-multiplied SE(3) mapping ORB-SLAM3 left-camera pose into the GT pose frame after no-scale world alignment, estimated after masking suspicious segments.",
        "model": "T_gt ~= T_world_align * T_orb_left_camera * T_camera_to_gt_frame",
        "R_camera_to_gt_frame": sensor_frame_rot.tolist(),
        "t_camera_to_gt_frame": sensor_frame_trans.tolist(),
        "angle_deg": rotation_angle_deg(sensor_frame_rot),
        "translation_norm_m": float(np.linalg.norm(sensor_frame_trans)),
        "world_alignment_translation": align_trans.tolist(),
        "translation_lstsq_rank": pose_alignment["translation_lstsq_rank"],
        "translation_lstsq_condition": pose_alignment["translation_lstsq_condition"],
        "residual_mean_deg": float(np.mean(orientation_errors_array)),
        "residual_rmse_deg": trial.rmse(orientation_errors_array),
        "residual_max_deg": float(np.max(orientation_errors_array)),
        "estimated_from_pose_pairs": len(pairs),
    }
    metrics = {
        "dataset": "corldata",
        "sequence": sequence,
        "mode": "ORB-SLAM3 stereo visual odometry, suspicious segments masked",
        "time_offset_s": time_offset,
        "time_offset_source": "original unmasked auto-search result; not re-optimized after masking",
        "max_time_difference_s": max_time_diff,
        "mask_padding_s": args.padding_s,
        "mask_sources": args.mask_sources,
        "mask_interval_count": len(intervals),
        "mask_source_segment_count": len(selected_segments),
        "mask_total_duration_s": sum(end - start for start, end in intervals),
        "associated_pose_count_original": len(pairs_all),
        "associated_pose_count_after_mask": len(pairs),
        "masked_associated_pose_count": len(pairs_all) - len(pairs),
        "estimated_pose_count": len(est_rows),
        "alignment": "SE(3) no-scale world alignment plus right-multiplied constant camera-to-GT-frame SE(3) compensation; suspicious intervals removed before fitting alignment",
        "position_rmse_m": trial.rmse(position_errors_array),
        "position_mean_m": float(np.mean(position_errors_array)),
        "position_median_m": float(np.median(position_errors_array)),
        "position_max_m": float(np.max(position_errors_array)),
        "orientation_rmse_deg": trial.rmse(orientation_errors_array),
        "orientation_mean_deg": float(np.mean(orientation_errors_array)),
        "orientation_max_deg": float(np.max(orientation_errors_array)),
        "orientation_without_sensor_frame_alignment_rmse_deg": trial.rmse(raw_orientation_errors_array),
        "orientation_without_sensor_frame_alignment_mean_deg": float(np.mean(raw_orientation_errors_array)),
        "rpy_rmse_deg": {
            "roll": trial.rmse(rpy_errors_array[:, 0]),
            "pitch": trial.rmse(rpy_errors_array[:, 1]),
            "yaw": trial.rmse(rpy_errors_array[:, 2]),
        },
        "sim3_scale_diagnostic": sim3_scale,
        "orientation_frame_alignment": frame_alignment,
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
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    make_plots(sequence, out_dir, aligned_rows, intervals, gt_rows[0]["t"])
    return metrics


def make_plots(sequence: str, out_dir: Path, rows: list[dict[str, object]], intervals: list[tuple[float, float]], first_gt_time: float) -> None:
    times = np.array([float(row["timestamp"]) - first_gt_time for row in rows], dtype=float)
    pos = np.array([float(row["position_error_m"]) for row in rows], dtype=float)
    ori = np.array([float(row["orientation_error_deg"]) for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(times, pos, linewidth=1.0)
    axes[0].set_ylabel("position error [m]")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(times, ori, linewidth=1.0)
    axes[1].set_ylabel("orientation error [deg]")
    axes[1].set_xlabel("relative time [s]")
    axes[1].grid(True, alpha=0.25)
    for start, end in intervals:
        rel_start = start - first_gt_time
        rel_end = end - first_gt_time
        for axis in axes:
            axis.axvspan(rel_start, rel_end, color="tab:red", alpha=0.08)
    fig.suptitle(f"{sequence} masked suspicious segments evaluation")
    fig.tight_layout()
    fig.savefig(out_dir / "masked_error_timeseries.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 6))
    axis = fig.add_subplot(111, projection="3d")
    est = np.array([[row["est_x"], row["est_y"], row["est_z"]] for row in rows], dtype=float)
    gt = np.array([[row["gt_x"], row["gt_y"], row["gt_z"]] for row in rows], dtype=float)
    axis.plot(gt[:, 0], gt[:, 1], gt[:, 2], "--", color="black", linewidth=1.2, label="GT masked")
    axis.plot(est[:, 0], est[:, 1], est[:, 2], color="tab:blue", linewidth=1.0, label="ORB aligned")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "masked_trajectory_3d.png", dpi=180)
    plt.close(fig)


def write_aggregate(args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    root = args.plato_root / "results" / "pseudo_gt" / "corldata"
    output_prefix = f"S00_S05_{args.output_name}"
    csv_path = root / f"{output_prefix}_summary.csv"
    json_path = root / f"{output_prefix}_summary.json"
    report_path = root / f"{output_prefix}_report.md"

    flat_rows = []
    for row in rows:
        original = row["original_metrics"]
        flat_rows.append(
            {
                "sequence": row["sequence"],
                "time_offset_s": row["time_offset_s"],
                "associated_after_mask": row["associated_pose_count_after_mask"],
                "associated_original": row["associated_pose_count_original"],
                "masked_associated_pose_count": row["masked_associated_pose_count"],
                "mask_interval_count": row["mask_interval_count"],
                "mask_total_duration_s": row["mask_total_duration_s"],
                "position_rmse_m_masked": row["position_rmse_m"],
                "position_rmse_m_original": original["position_rmse_m"],
                "position_max_m_masked": row["position_max_m"],
                "position_max_m_original": original["position_max_m"],
                "orientation_rmse_deg_masked": row["orientation_rmse_deg"],
                "orientation_rmse_deg_original": original["orientation_rmse_deg"],
                "orientation_max_deg_masked": row["orientation_max_deg"],
                "orientation_max_deg_original": original["orientation_max_deg"],
                "sim3_scale_diagnostic_masked": row["sim3_scale_diagnostic"],
            }
        )
    write_csv(csv_path, flat_rows)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        f"# corldata S00-S05 Evaluation After Masking Suspicious Segments ({args.output_name})",
        "",
        "Original results are unchanged. This supplementary evaluation removes intervals listed in `results/pseudo_gt/corldata/gt_quality/suspicious_segments.csv` using the configured mask sources and padding, then recomputes no-scale world alignment plus a right-multiplied fixed sensor-to-`Body_0` SE(3) compensation.",
        "",
        f"- Mask padding: `{args.padding_s}` s on both sides of each selected segment.",
        f"- Mask sources: `{', '.join(args.mask_sources)}`.",
        "- Time offsets are inherited from the original automatic search and are not re-optimized after masking.",
        "",
        "| seq | kept/orig assoc | masked poses | pos RMSE orig -> masked m | pos max orig -> masked m | angle RMSE orig -> masked deg | angle max orig -> masked deg | masked duration s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in flat_rows:
        lines.append(
            "| {sequence} | {kept}/{orig} | {masked} | {pos0:.5f} -> {pos1:.5f} | {posmax0:.5f} -> {posmax1:.5f} | {ori0:.3f} -> {ori1:.3f} | {orimax0:.3f} -> {orimax1:.3f} | {dur:.3f} |".format(
                sequence=row["sequence"],
                kept=row["associated_after_mask"],
                orig=row["associated_original"],
                masked=row["masked_associated_pose_count"],
                pos0=row["position_rmse_m_original"],
                pos1=row["position_rmse_m_masked"],
                posmax0=row["position_max_m_original"],
                posmax1=row["position_max_m_masked"],
                ori0=row["orientation_rmse_deg_original"],
                ori1=row["orientation_rmse_deg_masked"],
                orimax0=row["orientation_max_deg_original"],
                orimax1=row["orientation_max_deg_masked"],
                dur=row["mask_total_duration_s"],
            )
        )
    lines.extend(
        [
            "",
            "Per-sequence outputs:",
            "",
            f"- `results/pseudo_gt/corldata/<seq>/{args.output_name}/metrics.json`",
            f"- `results/pseudo_gt/corldata/<seq>/{args.output_name}/aligned_poses.csv`",
            f"- `results/pseudo_gt/corldata/<seq>/{args.output_name}/mask_intervals.csv`",
            f"- `results/pseudo_gt/corldata/<seq>/{args.output_name}/masked_error_timeseries.png`",
            f"- `results/pseudo_gt/corldata/<seq>/{args.output_name}/masked_trajectory_3d.png`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.aggregate_report_path = report_path


def main() -> None:
    args = parse_args()
    args.plato_root = args.plato_root.resolve()
    args.data_root = args.data_root.resolve()
    args.segments_csv = args.segments_csv.resolve()
    if not args.segments_csv.exists():
        raise FileNotFoundError(args.segments_csv)

    rows = [evaluate_sequence(args, sequence) for sequence in args.sequences]
    write_aggregate(args, rows)
    print(
        json.dumps(
            {
                "report": str(args.aggregate_report_path),
                "sequences": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
