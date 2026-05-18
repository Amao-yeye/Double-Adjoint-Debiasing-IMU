#!/usr/bin/env python3
"""Detect corldata lights-on/off intervals from DAVIS images and plot GT XY."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from PIL import Image


DEFAULT_PLATO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SEQUENCES = ["S13", "S15", "S20", "S21", "S22", "S23"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plato-root", type=Path, default=DEFAULT_PLATO_ROOT)
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PLATO_ROOT / "results" / "pseudo_gt" / "corldata" / "light_segments",
    )
    parser.add_argument("--camera", choices=["left", "right"], default="left")
    parser.add_argument("--min-off-duration-s", type=float, default=0.8)
    parser.add_argument("--fill-gap-s", type=float, default=0.6)
    parser.add_argument("--smooth-window-s", type=float, default=0.45)
    return parser.parse_args()


def seq_root(plato_root: Path, sequence: str) -> Path:
    return plato_root / "results" / "pseudo_gt" / "corldata" / sequence


def read_timestamp_ns(path: Path) -> np.ndarray:
    values = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) < 2:
        raise RuntimeError(f"{path} has too few timestamps")
    return np.array(values, dtype=np.int64)


def timestamp_seconds(timestamp_ns: np.ndarray) -> np.ndarray:
    return timestamp_ns.astype(np.float64) * 1e-9


def load_metrics(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_tum_positions(path: Path) -> dict[str, np.ndarray]:
    data = np.loadtxt(path, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] < 2 or data.shape[1] < 4:
        raise RuntimeError(f"{path} has too few trajectory rows")
    timestamps = data[:, 0]
    order = np.argsort(timestamps)
    data = data[order]
    timestamps = data[:, 0]
    _, unique_indices = np.unique(timestamps, return_index=True)
    data = data[np.sort(unique_indices)]
    return {
        "timestamp": data[:, 0],
        "x": data[:, 1],
        "y": data[:, 2],
        "z": data[:, 3],
    }


def otsu_threshold(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2 or float(np.ptp(values)) < 1e-9:
        return float(np.median(values))
    hist, edges = np.histogram(values, bins=256)
    centers = (edges[:-1] + edges[1:]) * 0.5
    weight1 = np.cumsum(hist).astype(float)
    weight2 = np.cumsum(hist[::-1]).astype(float)[::-1]
    mean1 = np.cumsum(hist * centers) / np.maximum(weight1, 1.0)
    mean2 = (np.cumsum((hist * centers)[::-1]) / np.maximum(weight2[::-1], 1.0))[::-1]
    variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    if not np.any(np.isfinite(variance)):
        return float(np.median(values))
    return float(centers[:-1][int(np.nanargmax(variance))])


def median_dt(times: np.ndarray) -> float:
    dt = np.diff(times)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return 1.0 / 30.0
    return float(np.median(dt))


def smooth_bool(mask: np.ndarray, window_frames: int) -> np.ndarray:
    if window_frames <= 1:
        return mask.copy()
    if window_frames % 2 == 0:
        window_frames += 1
    kernel = np.ones(window_frames, dtype=int)
    votes = np.convolve(mask.astype(int), kernel, mode="same")
    return votes >= (window_frames // 2 + 1)


def runs(mask: np.ndarray) -> list[tuple[bool, int, int]]:
    if len(mask) == 0:
        return []
    out: list[tuple[bool, int, int]] = []
    start = 0
    current = bool(mask[0])
    for idx in range(1, len(mask)):
        value = bool(mask[idx])
        if value != current:
            out.append((current, start, idx - 1))
            start = idx
            current = value
    out.append((current, start, len(mask) - 1))
    return out


def close_short_gaps(mask: np.ndarray, times: np.ndarray, max_gap_s: float) -> np.ndarray:
    closed = mask.copy()
    run_list = runs(closed)
    for idx, (value, start, end) in enumerate(run_list):
        if value or idx == 0 or idx == len(run_list) - 1:
            continue
        before = run_list[idx - 1][0]
        after = run_list[idx + 1][0]
        duration = float(times[end] - times[start])
        if before and after and duration <= max_gap_s:
            closed[start : end + 1] = True
    return closed


def remove_short_true_runs(mask: np.ndarray, times: np.ndarray, min_duration_s: float) -> np.ndarray:
    filtered = mask.copy()
    for value, start, end in runs(filtered):
        if not value:
            continue
        duration = float(times[end] - times[start])
        if duration < min_duration_s:
            filtered[start : end + 1] = False
    return filtered


def image_metrics(sequence_dir: Path, timestamp_ns: np.ndarray, camera: str) -> dict[str, np.ndarray]:
    image_dir = sequence_dir / camera
    means: list[float] = []
    medians: list[float] = []
    p10s: list[float] = []
    p90s: list[float] = []
    stds: list[float] = []
    dark_ratios: list[float] = []
    grad_energy: list[float] = []
    for stamp_ns in timestamp_ns:
        path = image_dir / f"{stamp_ns}.png"
        if not path.exists():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            values = np.asarray(image.convert("L"), dtype=np.float32)
        means.append(float(np.mean(values)))
        medians.append(float(np.median(values)))
        p10s.append(float(np.percentile(values, 10)))
        p90s.append(float(np.percentile(values, 90)))
        stds.append(float(np.std(values)))
        dark_ratios.append(float(np.mean(values < 8.0)))
        gy, gx = np.gradient(values)
        grad_energy.append(float(np.mean(np.sqrt(gx * gx + gy * gy))))
    return {
        "mean": np.array(means, dtype=float),
        "median": np.array(medians, dtype=float),
        "p10": np.array(p10s, dtype=float),
        "p90": np.array(p90s, dtype=float),
        "std": np.array(stds, dtype=float),
        "dark_ratio": np.array(dark_ratios, dtype=float),
        "gradient": np.array(grad_energy, dtype=float),
    }


def classify_light(times: np.ndarray, metrics: dict[str, np.ndarray], args: argparse.Namespace) -> tuple[np.ndarray, dict[str, float]]:
    score = metrics["mean"].copy()
    threshold = otsu_threshold(score)
    off = score <= threshold
    dt = median_dt(times)
    smooth_frames = max(1, int(round(args.smooth_window_s / dt)))
    gap_frames = max(1, int(round(args.fill_gap_s / dt)))
    off = smooth_bool(off, smooth_frames)
    off = close_short_gaps(off, times, gap_frames * dt)
    off = remove_short_true_runs(off, times, args.min_off_duration_s)
    return off, {
        "threshold_mean_intensity": threshold,
        "median_dt_s": dt,
        "smooth_window_frames": float(smooth_frames),
        "fill_gap_frames": float(gap_frames),
        "off_frame_ratio": float(np.mean(off)),
    }


def state_intervals(sequence: str, times: np.ndarray, off: np.ndarray, metrics: dict[str, np.ndarray], threshold: float, offset: float) -> list[dict[str, object]]:
    intervals: list[dict[str, object]] = []
    rel = times - times[0]
    for value, start, end in runs(off):
        state = "light_off" if value else "light_on"
        frame_slice = slice(start, end + 1)
        intervals.append(
            {
                "sequence": sequence,
                "state": state,
                "start_image_time_s": float(times[start]),
                "end_image_time_s": float(times[end]),
                "start_gt_time_s": float(times[start] + offset),
                "end_gt_time_s": float(times[end] + offset),
                "start_rel_s": float(rel[start]),
                "end_rel_s": float(rel[end]),
                "duration_s": float(times[end] - times[start]),
                "frame_count": int(end - start + 1),
                "mean_intensity": float(np.mean(metrics["mean"][frame_slice])),
                "median_intensity": float(np.mean(metrics["median"][frame_slice])),
                "p90_intensity": float(np.mean(metrics["p90"][frame_slice])),
                "std_intensity": float(np.mean(metrics["std"][frame_slice])),
                "dark_ratio": float(np.mean(metrics["dark_ratio"][frame_slice])),
                "gradient_energy": float(np.mean(metrics["gradient"][frame_slice])),
                "threshold_mean_intensity": threshold,
            }
        )
    return intervals


def state_for_gt(gt_times: np.ndarray, image_times: np.ndarray, off: np.ndarray, offset: float) -> np.ndarray:
    query_image_times = gt_times - offset
    indices = np.searchsorted(image_times, query_image_times, side="left")
    indices = np.clip(indices, 0, len(image_times) - 1)
    previous = np.clip(indices - 1, 0, len(image_times) - 1)
    choose_previous = np.abs(image_times[previous] - query_image_times) < np.abs(image_times[indices] - query_image_times)
    nearest = np.where(choose_previous, previous, indices)
    return off[nearest]


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


def plot_brightness(sequence: str, output_dir: Path, times: np.ndarray, off: np.ndarray, metrics: dict[str, np.ndarray], threshold: float, camera: str) -> None:
    rel = times - times[0]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"{sequence} DAVIS {camera} image light-state detection")
    axes[0].plot(rel, metrics["mean"], color="black", linewidth=1.0, label="mean")
    axes[0].plot(rel, metrics["p90"], color="tab:blue", linewidth=0.8, label="p90")
    axes[0].axhline(threshold, color="tab:red", linestyle="--", linewidth=1.1, label="Otsu threshold")
    axes[0].set_ylabel("intensity")
    axes[0].legend(loc="upper right")
    axes[1].plot(rel, metrics["dark_ratio"], color="tab:purple", linewidth=1.0)
    axes[1].set_ylabel("dark pixel ratio")
    axes[2].plot(rel, metrics["gradient"], color="tab:green", linewidth=1.0)
    axes[2].set_ylabel("gradient energy")
    axes[2].set_xlabel("image relative time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        for value, start, end in runs(off):
            if value:
                ax.axvspan(rel[start], rel[end], color="0.75", alpha=0.45)
    fig.tight_layout()
    fig.savefig(output_dir / f"{sequence}_brightness_diagnostic.png", dpi=180)
    plt.close(fig)


def plot_gt_xy(sequence: str, output_dir: Path, gt: dict[str, np.ndarray], gt_off: np.ndarray, intervals: list[dict[str, object]]) -> None:
    times = gt["timestamp"]
    x = gt["x"]
    y = gt["y"]
    xyz = np.column_stack([x, y, gt["z"]])
    dt = np.diff(times)
    dpos = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    median_step = float(np.median(dpos[dpos > 0])) if np.any(dpos > 0) else 0.0
    p99_step = float(np.quantile(dpos, 0.99)) if len(dpos) else 0.0
    max_dt = max(0.1, 10.0 * median_dt(times))
    max_step = max(0.05, 20.0 * max(p99_step, median_step))
    continuous = (dt > 0.0) & (dt <= max_dt) & (dpos <= max_step)

    points = np.column_stack([x, y])
    segments = np.stack([points[:-1], points[1:]], axis=1)[continuous]
    segment_off = (gt_off[:-1] | gt_off[1:])[continuous]
    colors = np.where(segment_off, "#bdbdbd", "#000000")
    widths = np.where(segment_off, 3.6, 2.6)

    fig, ax = plt.subplots(figsize=(8, 8))
    collection = LineCollection(segments, colors=colors, linewidths=widths, capstyle="round", joinstyle="round")
    ax.add_collection(collection)
    ax.scatter([x[0]], [y[0]], s=42, color="#1f77b4", edgecolor="white", linewidth=0.8, zorder=3, label="start")
    ax.scatter([x[-1]], [y[-1]], s=42, color="#d62728", edgecolor="white", linewidth=0.8, zorder=3, label="end")
    ax.plot([], [], color="#000000", linewidth=3.0, label="lights on")
    ax.plot([], [], color="#bdbdbd", linewidth=4.0, label="lights off")
    ax.set_title(f"{sequence} full GT XY trajectory by image light state")
    ax.set_xlabel("GT X [m]")
    ax.set_ylabel("GT Y [m]")
    ax.axis("equal")
    pad = max(0.05, 0.08 * max(float(np.ptp(x)), float(np.ptp(y)), 1e-6))
    ax.set_xlim(float(np.min(x) - pad), float(np.max(x) + pad))
    ax.set_ylim(float(np.min(y) - pad), float(np.max(y) + pad))
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=True, framealpha=0.95)

    off_intervals = [item for item in intervals if item["state"] == "light_off"]
    text = "\n".join(
        f"off {item['start_rel_s']:.1f}-{item['end_rel_s']:.1f}s"
        for item in off_intervals[:6]
    )
    if text:
        ax.text(
            0.02,
            0.02,
            text,
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.92},
        )
    fig.tight_layout()
    fig.savefig(output_dir / f"{sequence}_gt_xy_light_segments.png", dpi=220)
    plt.close(fig)


def process_sequence(args: argparse.Namespace, sequence: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    root = seq_root(args.plato_root, sequence)
    timestamp_ns = read_timestamp_ns(root / "sequence" / "timestamps.txt")
    timestamps = timestamp_seconds(timestamp_ns)
    metrics = image_metrics(root / "sequence", timestamp_ns, args.camera)
    off, params = classify_light(timestamps, metrics, args)
    eval_metrics = load_metrics(root / "evo_gt_aligned" / "metrics.json")
    offset = float(eval_metrics["time_offset_s"])
    intervals = state_intervals(sequence, timestamps, off, metrics, params["threshold_mean_intensity"], offset)
    full_gt = read_tum_positions(root / "gt_aligned" / "groundtruth_camera_frame.tum")
    gt_off = state_for_gt(full_gt["timestamp"], timestamps, off, offset)

    plot_brightness(sequence, args.output_dir, timestamps, off, metrics, params["threshold_mean_intensity"], args.camera)
    plot_gt_xy(sequence, args.output_dir, full_gt, gt_off, intervals)

    summary = {
        "sequence": sequence,
        "camera": args.camera,
        "image_count": int(len(timestamps)),
        "time_offset_s": offset,
        "threshold_mean_intensity": params["threshold_mean_intensity"],
        "off_frame_ratio": params["off_frame_ratio"],
        "off_duration_s": float(sum(item["duration_s"] for item in intervals if item["state"] == "light_off")),
        "on_duration_s": float(sum(item["duration_s"] for item in intervals if item["state"] == "light_on")),
        "off_interval_count": int(sum(1 for item in intervals if item["state"] == "light_off")),
        "xy_png": str(args.output_dir / f"{sequence}_gt_xy_light_segments.png"),
        "brightness_png": str(args.output_dir / f"{sequence}_brightness_diagnostic.png"),
    }
    return intervals, summary


def write_report(path: Path, summaries: list[dict[str, object]], intervals: list[dict[str, object]]) -> None:
    intervals_by_seq: dict[str, list[dict[str, object]]] = {}
    for item in intervals:
        intervals_by_seq.setdefault(str(item["sequence"]), []).append(item)
    lines = [
        "# corldata Test-Set Light Segment Detection",
        "",
        "Light state is detected from exported DAVIS grayscale images, not from trajectory error. The GT XY plots color the full converted ground-truth trajectory by the nearest image light state after applying each sequence's fitted time offset.",
        "",
        "Color convention in XY PNGs:",
        "",
        "- black: lights on",
        "- gray: lights off",
        "",
        "| seq | camera | images | off intervals | off duration s | off frame ratio | threshold mean | XY PNG |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summaries:
        lines.append(
            "| {seq} | {camera} | {count} | {off_count} | {off_dur:.2f} | {ratio:.3f} | {threshold:.2f} | `{png}` |".format(
                seq=row["sequence"],
                camera=row["camera"],
                count=row["image_count"],
                off_count=row["off_interval_count"],
                off_dur=row["off_duration_s"],
                ratio=row["off_frame_ratio"],
                threshold=row["threshold_mean_intensity"],
                png=Path(str(row["xy_png"])).name,
            )
        )
    lines.extend(["", "## Detected Intervals", ""])
    for row in summaries:
        seq = str(row["sequence"])
        lines.extend(
            [
                f"### {seq}",
                "",
                "| state | image rel start s | image rel end s | duration s | mean intensity | dark ratio |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in intervals_by_seq.get(seq, []):
            lines.append(
                "| {state} | {start:.2f} | {end:.2f} | {duration:.2f} | {mean:.2f} | {dark:.3f} |".format(
                    state=item["state"],
                    start=item["start_rel_s"],
                    end=item["end_rel_s"],
                    duration=item["duration_s"],
                    mean=item["mean_intensity"],
                    dark=item["dark_ratio"],
                )
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.plato_root = args.plato_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_intervals: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for sequence in args.sequences:
        intervals, summary = process_sequence(args, sequence)
        all_intervals.extend(intervals)
        summaries.append(summary)

    write_csv(args.output_dir / "light_segments.csv", all_intervals)
    write_csv(args.output_dir / "light_segments_summary.csv", summaries)
    (args.output_dir / "light_segments.json").write_text(json.dumps(all_intervals, indent=2, sort_keys=True), encoding="utf-8")
    (args.output_dir / "light_segments_summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    write_report(args.output_dir / "light_segments_summary.md", summaries, all_intervals)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "sequence_count": len(summaries),
                "interval_count": len(all_intervals),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
