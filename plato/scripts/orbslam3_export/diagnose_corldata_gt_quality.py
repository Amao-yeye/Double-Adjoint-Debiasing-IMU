#!/usr/bin/env python3
"""Diagnose corldata mocap GT jumps and local VO-GT residual spikes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_PLATO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = Path("/home/amao/datasets/corldata/0511_data")


@dataclass
class Series:
    t_abs: np.ndarray
    t_rel: np.ndarray
    values: np.ndarray
    threshold: float
    label: str
    units: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plato-root", type=Path, default=DEFAULT_PLATO_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sequences", nargs="+", default=["S00", "S01", "S02", "S03", "S04", "S05"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PLATO_ROOT / "results" / "pseudo_gt" / "corldata" / "gt_quality",
    )
    parser.add_argument("--segment-gap-s", type=float, default=0.15)
    parser.add_argument("--residual-gap-s", type=float, default=0.20)
    return parser.parse_args()


def load_tum(path: Path) -> tuple[np.ndarray, np.ndarray, Rotation]:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 8:
        raise ValueError(f"Unexpected TUM pose file shape for {path}: {data.shape}")
    timestamps = data[:, 0].astype(float)
    positions = data[:, 1:4].astype(float)
    quats = data[:, 4:8].astype(float)
    norms = np.linalg.norm(quats, axis=1)
    if np.any(norms < 1e-12):
        raise ValueError(f"Near-zero quaternion in {path}")
    quats = quats / norms[:, None]
    for idx in range(1, len(quats)):
        if np.dot(quats[idx - 1], quats[idx]) < 0:
            quats[idx] *= -1.0
    return timestamps, positions, Rotation.from_quat(quats)


def robust_mad(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, 1.4826 * mad


def robust_threshold(values: np.ndarray, multiplier: float, floor: float) -> float:
    median, sigma = robust_mad(values)
    if sigma <= 1e-12:
        return max(floor, float(np.percentile(values, 99.5)))
    return max(floor, median + multiplier * sigma)


def read_aligned_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out: dict[str, list[float]] = {"timestamp": [], "position_error_m": [], "orientation_error_deg": []}
    for row in rows:
        for key in out:
            out[key].append(float(row[key]))
    return {key: np.array(values, dtype=float) for key, values in out.items()}


def segment_mask(
    sequence: str,
    source: str,
    series: Series,
    mask: np.ndarray,
    max_gap_s: float,
) -> list[dict[str, object]]:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return []
    segments: list[dict[str, object]] = []
    start = int(indices[0])
    previous = int(indices[0])
    for index in map(int, indices[1:]):
        if series.t_rel[index] - series.t_rel[previous] > max_gap_s:
            segments.append(make_segment(sequence, source, series, start, previous))
            start = index
        previous = index
    segments.append(make_segment(sequence, source, series, start, previous))
    return segments


def make_segment(sequence: str, source: str, series: Series, start: int, end: int) -> dict[str, object]:
    segment_values = series.values[start : end + 1]
    max_local = int(np.argmax(segment_values)) + start
    return {
        "sequence": sequence,
        "source": source,
        "label": series.label,
        "units": series.units,
        "threshold": series.threshold,
        "start_time_s": float(series.t_abs[start]),
        "end_time_s": float(series.t_abs[end]),
        "start_rel_s": float(series.t_rel[start]),
        "end_rel_s": float(series.t_rel[end]),
        "duration_s": float(max(0.0, series.t_rel[end] - series.t_rel[start])),
        "sample_count": int(end - start + 1),
        "max_value": float(series.values[max_local]),
        "mean_value": float(np.mean(segment_values)),
        "max_time_s": float(series.t_abs[max_local]),
        "max_rel_s": float(series.t_rel[max_local]),
    }


def summarize_gt(sequence: str, gt_path: Path, segment_gap_s: float) -> tuple[dict[str, object], list[dict[str, object]], dict[str, Series]]:
    t, p, r = load_tum(gt_path)
    t_rel = t - t[0]
    dt = np.diff(t)
    dp = np.linalg.norm(np.diff(p, axis=0), axis=1)
    velocity = np.diff(p, axis=0) / dt[:, None]
    speed = np.linalg.norm(velocity, axis=1)
    accel = np.linalg.norm(np.diff(velocity, axis=0) / ((dt[:-1] + dt[1:])[:, None] * 0.5), axis=1)
    angle_step_deg = np.degrees((r[:-1].inv() * r[1:]).magnitude())
    angular_speed = angle_step_deg / dt

    median_dt = float(np.median(dt))
    small_dt_threshold = min(0.001, 0.25 * median_dt)
    large_dt_threshold = max(0.02, 2.5 * median_dt)
    position_step_threshold = robust_threshold(dp, 20.0, 0.02)
    angular_step_threshold = robust_threshold(angle_step_deg, 20.0, 1.0)
    speed_threshold = robust_threshold(speed, 20.0, 2.0)
    accel_threshold = robust_threshold(accel, 20.0, 100.0)

    series = {
        "gt_small_dt": Series(t[:-1], t_rel[:-1], dt, small_dt_threshold, "GT small timestamp step", "s"),
        "gt_large_dt": Series(t[:-1], t_rel[:-1], dt, large_dt_threshold, "GT large timestamp gap", "s"),
        "gt_position_step": Series(t[:-1], t_rel[:-1], dp, position_step_threshold, "GT position step", "m"),
        "gt_speed_spike": Series(t[:-1], t_rel[:-1], speed, speed_threshold, "GT speed spike", "m/s"),
        "gt_accel_spike": Series(t[1:-1], t_rel[1:-1], accel, accel_threshold, "GT acceleration spike", "m/s^2"),
        "gt_angle_step": Series(t[:-1], t_rel[:-1], angle_step_deg, angular_step_threshold, "GT angular step", "deg"),
        "gt_angular_speed": Series(t[:-1], t_rel[:-1], angular_speed, robust_threshold(angular_speed, 20.0, 120.0), "GT angular speed spike", "deg/s"),
    }
    masks = {
        "gt_small_dt": dt < small_dt_threshold,
        "gt_large_dt": dt > large_dt_threshold,
        "gt_position_step": dp > position_step_threshold,
        "gt_speed_spike": speed > speed_threshold,
        "gt_accel_spike": accel > accel_threshold,
        "gt_angle_step": angle_step_deg > angular_step_threshold,
        "gt_angular_speed": angular_speed > series["gt_angular_speed"].threshold,
    }
    segments: list[dict[str, object]] = []
    for source, mask in masks.items():
        segments.extend(segment_mask(sequence, source, series[source], mask, segment_gap_s))

    stats = {
        "sequence": sequence,
        "gt_pose_count": int(len(t)),
        "gt_duration_s": float(t[-1] - t[0]),
        "gt_median_dt_s": median_dt,
        "gt_min_dt_s": float(np.min(dt)),
        "gt_max_dt_s": float(np.max(dt)),
        "gt_small_dt_count": int(np.sum(masks["gt_small_dt"])),
        "gt_large_dt_count": int(np.sum(masks["gt_large_dt"])),
        "gt_max_position_step_m": float(np.max(dp)),
        "gt_max_speed_mps": float(np.max(speed)),
        "gt_max_accel_mps2": float(np.max(accel)),
        "gt_max_angle_step_deg": float(np.max(angle_step_deg)),
        "gt_max_angular_speed_degps": float(np.max(angular_speed)),
    }
    return stats, segments, series


def summarize_residuals(
    sequence: str,
    aligned_csv: Path,
    residual_gap_s: float,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, Series]]:
    aligned = read_aligned_csv(aligned_csv)
    t = aligned["timestamp"]
    t_rel = t - t[0]
    pos = aligned["position_error_m"]
    ori = aligned["orientation_error_deg"]
    pos_threshold = robust_threshold(pos, 6.0, 0.10)
    ori_threshold = robust_threshold(ori, 6.0, 5.0)
    series = {
        "residual_position": Series(t, t_rel, pos, pos_threshold, "VO-GT position residual", "m"),
        "residual_orientation": Series(t, t_rel, ori, ori_threshold, "VO-GT orientation residual", "deg"),
    }
    segments = []
    segments.extend(segment_mask(sequence, "residual_position", series["residual_position"], pos > pos_threshold, residual_gap_s))
    segments.extend(segment_mask(sequence, "residual_orientation", series["residual_orientation"], ori > ori_threshold, residual_gap_s))
    stats = {
        "sequence": sequence,
        "associated_pose_count": int(len(t)),
        "residual_position_threshold_m": pos_threshold,
        "residual_orientation_threshold_deg": ori_threshold,
        "residual_position_max_m": float(np.max(pos)),
        "residual_position_p99_m": float(np.percentile(pos, 99)),
        "residual_orientation_max_deg": float(np.max(ori)),
        "residual_orientation_p99_deg": float(np.percentile(ori, 99)),
    }
    return stats, segments, series


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


def plot_sequence(
    sequence: str,
    output_dir: Path,
    gt_series: dict[str, Series],
    residual_series: dict[str, Series],
    segments: list[dict[str, object]],
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=False)
    fig.suptitle(f"{sequence} GT quality and VO-GT residual diagnostics")

    axes[0].plot(gt_series["gt_small_dt"].t_rel, gt_series["gt_small_dt"].values * 1000.0, linewidth=0.8)
    axes[0].axhline(gt_series["gt_small_dt"].threshold * 1000.0, color="r", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("dt [ms]")

    axes[1].plot(gt_series["gt_position_step"].t_rel, gt_series["gt_position_step"].values, linewidth=0.8)
    axes[1].axhline(gt_series["gt_position_step"].threshold, color="r", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("GT step [m]")

    axes[2].plot(gt_series["gt_angle_step"].t_rel, gt_series["gt_angle_step"].values, linewidth=0.8)
    axes[2].axhline(gt_series["gt_angle_step"].threshold, color="r", linestyle="--", linewidth=0.8)
    axes[2].set_ylabel("GT angle step [deg]")

    axes[3].plot(residual_series["residual_position"].t_rel, residual_series["residual_position"].values, label="position [m]", linewidth=0.9)
    axes[3].plot(
        residual_series["residual_orientation"].t_rel,
        residual_series["residual_orientation"].values / 10.0,
        label="orientation / 10 [deg]",
        linewidth=0.9,
    )
    axes[3].axhline(residual_series["residual_position"].threshold, color="r", linestyle="--", linewidth=0.8)
    axes[3].set_ylabel("VO-GT residual")
    axes[3].set_xlabel("relative time [s]")
    axes[3].legend(loc="upper right")

    for axis in axes:
        axis.grid(True, alpha=0.25)
    for segment in segments:
        if segment["sequence"] != sequence:
            continue
        start = float(segment["start_rel_s"])
        end = float(segment["end_rel_s"])
        if end <= start:
            end = start + 0.02
        color = "tab:red" if str(segment["source"]).startswith("gt_") else "tab:orange"
        for axis in axes:
            axis.axvspan(start, end, color=color, alpha=0.08)

    fig.tight_layout()
    fig.savefig(output_dir / f"{sequence}_gt_quality.png", dpi=180)
    plt.close(fig)


def severity_key(segment: dict[str, object]) -> float:
    threshold = float(segment["threshold"])
    if threshold <= 0:
        return float(segment["max_value"])
    return float(segment["max_value"]) / threshold


def fmt(value: object, precision: int = 3) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{precision}f}"
    return str(value)


def write_report(
    path: Path,
    gt_stats: list[dict[str, object]],
    residual_stats: list[dict[str, object]],
    segments: list[dict[str, object]],
) -> None:
    residual_by_seq = {row["sequence"]: row for row in residual_stats}
    lines = [
        "# corldata GT Quality Diagnostics",
        "",
        "This report checks the mocap `Body_0.tum` files for local timestamp, position, and orientation discontinuities, then cross-checks them against the already aligned ORB-SLAM3 VO residuals.",
        "",
        "A segment in this report is a diagnostic candidate, not an automatic rejection. Small timestamp steps can inflate velocity/acceleration without a large position jump; large VO-GT residual spikes can come from either bad GT or ORB-SLAM3 tracking.",
        "",
        "## Sequence Summary",
        "",
        "| seq | GT min dt ms | small dt | large dt | max GT step m | max GT angle step deg | pos residual max m | ori residual max deg | candidate segments |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in gt_stats:
        seq = row["sequence"]
        res = residual_by_seq[seq]
        seq_segments = [seg for seg in segments if seg["sequence"] == seq]
        lines.append(
            "| {seq} | {min_dt} | {small_dt} | {large_dt} | {step} | {angle} | {pos_res} | {ori_res} | {count} |".format(
                seq=seq,
                min_dt=fmt(float(row["gt_min_dt_s"]) * 1000.0, 4),
                small_dt=row["gt_small_dt_count"],
                large_dt=row["gt_large_dt_count"],
                step=fmt(row["gt_max_position_step_m"], 4),
                angle=fmt(row["gt_max_angle_step_deg"], 3),
                pos_res=fmt(res["residual_position_max_m"], 4),
                ori_res=fmt(res["residual_orientation_max_deg"], 3),
                count=len(seq_segments),
            )
        )

    lines.extend(["", "## Top Candidate Bad Segments", ""])
    for seq in [row["sequence"] for row in gt_stats]:
        seq_segments = sorted(
            [seg for seg in segments if seg["sequence"] == seq],
            key=severity_key,
            reverse=True,
        )[:12]
        lines.extend(
            [
                f"### {seq}",
                "",
                "| source | rel start s | rel end s | max rel s | max value | threshold | samples |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        if not seq_segments:
            lines.append("| none | | | | | | |")
        for seg in seq_segments:
            lines.append(
                "| {source} | {start} | {end} | {max_t} | {max_v} {units} | {threshold} {units} | {samples} |".format(
                    source=seg["source"],
                    start=fmt(seg["start_rel_s"], 3),
                    end=fmt(seg["end_rel_s"], 3),
                    max_t=fmt(seg["max_rel_s"], 3),
                    max_v=fmt(seg["max_value"], 4),
                    threshold=fmt(seg["threshold"], 4),
                    units=seg["units"],
                    samples=seg["sample_count"],
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Artifacts",
            "",
            "- Full segment CSV: `results/pseudo_gt/corldata/gt_quality/suspicious_segments.csv`",
            "- Summary CSV: `results/pseudo_gt/corldata/gt_quality/gt_quality_summary.csv`",
            "- Per-sequence plots: `results/pseudo_gt/corldata/gt_quality/Sxx_gt_quality.png`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.plato_root = args.plato_root.resolve()
    args.data_root = args.data_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gt_stats: list[dict[str, object]] = []
    residual_stats: list[dict[str, object]] = []
    all_segments: list[dict[str, object]] = []

    for sequence in args.sequences:
        gt_path = args.data_root / "gt" / sequence / "Body_0.tum"
        aligned_csv = args.plato_root / "results" / "pseudo_gt" / "corldata" / sequence / "aligned" / "aligned_poses.csv"
        if not gt_path.exists():
            raise FileNotFoundError(gt_path)
        if not aligned_csv.exists():
            raise FileNotFoundError(aligned_csv)

        gt_row, gt_segments, gt_series = summarize_gt(sequence, gt_path, args.segment_gap_s)
        res_row, res_segments, residual_series = summarize_residuals(sequence, aligned_csv, args.residual_gap_s)
        segments = gt_segments + res_segments
        gt_stats.append(gt_row)
        residual_stats.append(res_row)
        all_segments.extend(segments)
        plot_sequence(sequence, args.output_dir, gt_series, residual_series, segments)

    summary_rows: list[dict[str, object]] = []
    residual_by_seq = {row["sequence"]: row for row in residual_stats}
    for row in gt_stats:
        seq = row["sequence"]
        merged = dict(row)
        merged.update(residual_by_seq[seq])
        merged["candidate_segment_count"] = sum(1 for seg in all_segments if seg["sequence"] == seq)
        summary_rows.append(merged)

    write_csv(args.output_dir / "gt_quality_summary.csv", summary_rows)
    write_csv(args.output_dir / "suspicious_segments.csv", sorted(all_segments, key=lambda item: (item["sequence"], item["start_rel_s"], item["source"])))
    (args.output_dir / "gt_quality_summary.json").write_text(json.dumps(summary_rows, indent=2, sort_keys=True), encoding="utf-8")
    (args.output_dir / "suspicious_segments.json").write_text(json.dumps(all_segments, indent=2, sort_keys=True), encoding="utf-8")
    write_report(args.output_dir / "gt_quality_report.md", gt_stats, residual_stats, all_segments)
    print(json.dumps({"output_dir": str(args.output_dir), "segments": len(all_segments)}, indent=2))


if __name__ == "__main__":
    main()
