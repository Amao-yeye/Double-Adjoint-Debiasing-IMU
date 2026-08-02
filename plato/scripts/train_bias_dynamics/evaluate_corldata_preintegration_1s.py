#!/usr/bin/env python3
"""Evaluate short-window corldata IMU preintegration drift against GT.

For each selected sequence this script chooses high-motion windows, starts
all methods from the GT state at the first sample, and compares:

* raw IMU with no bias compensation;
* GT-differenced constant bias fixed at the window start;
* learned bias from the configured Debias_IMU FullTra_result.p.

The main drift criterion is the first time that position error reaches the
configured threshold, default 1 m.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.signal import savgol_filter
except ImportError:  # pragma: no cover - scipy is available in the project env.
    savgol_filter = None


PLATO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BEST_RESULTS = (
    PLATO_ROOT
    / "results"
    / "debias_imu"
    / "corldata_sparse_endpoint_no_udot_w128_sg101_initmedian5_e301_s00s07_alltest"
)
DEFAULT_TRAIN_S00_RESULTS = (
    PLATO_ROOT / "new_results" / "corldata1" / "preinteg1s" / "ours_train_s00_w128_e301_onlytest"
)
DEFAULT_OUTPUT_DIR = PLATO_ROOT / "new_results" / "corldata1" / "preinteg1s"
DEFAULT_SEQUENCE_RESULTS = {
    "S00": DEFAULT_TRAIN_S00_RESULTS,
    "S13": DEFAULT_BEST_RESULTS,
    "S15": DEFAULT_BEST_RESULTS,
}
DEFAULT_SEQUENCE_ROLES = {
    "S00": "train",
    "S01": "train",
    "S02": "train",
    "S03": "train",
    "S13": "test",
    "S15": "test",
    "S20": "test",
    "S21": "test",
    "S22": "test",
    "S23": "test",
}
METHOD_LABELS = {
    "gt": "groundtruth",
    "raw": "raw no bias",
    "gt_const_start": "GT-diff const bias",
    "ours": "ours learned bias",
}
METHOD_COLORS = {
    "gt": "#111827",
    "raw": "#d97706",
    "gt_const_start": "#2563eb",
    "ours": "#059669",
}
PLOT_METHODS = ["gt", "raw", "ours"]


@dataclass(frozen=True)
class SequenceData:
    sequence: str
    result_dir: Path
    result_path: Path
    t: np.ndarray
    imu_raw: np.ndarray
    learned_bias: np.ndarray
    gt_bias: np.ndarray
    R_gt: np.ndarray
    v_gt: np.ndarray
    p_gt: np.ndarray


@dataclass(frozen=True)
class Window:
    segment_id: int
    start: int
    stop: int
    gt_path_m: float
    gt_displacement_m: float


@dataclass
class MethodResult:
    method: str
    R: np.ndarray
    v: np.ndarray
    p: np.ndarray
    error_m: np.ndarray
    metrics: dict[str, float | None]


@dataclass
class SegmentResult:
    sequence: str
    role: str
    result_dir: Path
    window: Window
    start_time_s: float
    t_rel: np.ndarray
    p_gt: np.ndarray
    method_results: dict[str, MethodResult]


def hat(w: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]],
        dtype=np.float64,
    )


def so3_exp(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    W = hat(w)
    if theta < 1.0e-12:
        return np.eye(3) + W
    return np.eye(3) + math.sin(theta) / theta * W + (1.0 - math.cos(theta)) / (theta * theta) * (W @ W)


def so3_log(R: np.ndarray) -> np.ndarray:
    cos_theta = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cos_theta)
    skew = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
    if theta < 1.0e-12:
        return 0.5 * skew
    return theta / (2.0 * math.sin(theta)) * skew


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"You are using `torch\.load` with `weights_only=False`.*",
                category=FutureWarning,
            )
            return pickle.load(f)


def make_odd_window(window: int, n: int, order: int) -> int:
    window = int(window)
    if window % 2 == 0:
        window += 1
    if window > n:
        window = n if n % 2 == 1 else n - 1
    if window <= order:
        return 0
    return max(window, 0)


def smooth_signal(values: np.ndarray, window: int, order: int) -> np.ndarray:
    if savgol_filter is None:
        return values
    win = make_odd_window(window, values.shape[0], order)
    if win < 3:
        return values
    return savgol_filter(values, window_length=win, polyorder=order, axis=0, mode="interp")


def compute_gt_differenced_bias(
    t: np.ndarray,
    imu_raw: np.ndarray,
    R_gt: np.ndarray,
    v_gt: np.ndarray,
    gravity: np.ndarray,
    gyro_window: int,
    accel_window: int,
    gyro_order: int,
    accel_order: int,
) -> np.ndarray:
    if t.shape[0] < 3:
        raise ValueError("Need at least three samples to compute GT-differenced bias.")

    omega_body = np.empty((t.shape[0], 3), dtype=np.float64)
    for i in range(t.shape[0] - 1):
        dt = float(t[i + 1] - t[i])
        if dt <= 0.0:
            omega_body[i] = omega_body[i - 1] if i > 0 else 0.0
            continue
        omega_body[i] = so3_log(R_gt[i].T @ R_gt[i + 1]) / dt
    omega_body[-1] = omega_body[-2]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        acc_world = np.gradient(v_gt, t, axis=0, edge_order=2)
    specific_force_body = np.einsum("nij,nj->ni", np.swapaxes(R_gt, 1, 2), acc_world - gravity)

    gyro_bias = imu_raw[:, :3] - omega_body
    accel_bias = imu_raw[:, 3:6] - specific_force_body
    gyro_bias = smooth_signal(gyro_bias, gyro_window, gyro_order)
    accel_bias = smooth_signal(accel_bias, accel_window, accel_order)
    return np.column_stack([gyro_bias, accel_bias])


def load_sequence(sequence: str, result_dir: Path, args: argparse.Namespace) -> SequenceData:
    result_path = result_dir / "Sequnces-results" / sequence / "FullTra_result.p"
    if not result_path.exists():
        raise FileNotFoundError(result_path)

    data = load_pickle(result_path)
    sol_pred = as_numpy(data["sol_pred"]).astype(np.float64)
    X_gt = as_numpy(data["X_gt"]).astype(np.float64)
    t = as_numpy(data["t_gt"]).astype(np.float64)
    imu_raw = as_numpy(data["imu_raw"]).astype(np.float64)
    if sol_pred.shape[0] != t.shape[0] or imu_raw.shape[0] != t.shape[0] or X_gt.shape[0] != t.shape[0]:
        raise ValueError(
            f"{sequence} length mismatch: sol={sol_pred.shape[0]}, imu={imu_raw.shape[0]}, "
            f"X_gt={X_gt.shape[0]}, t={t.shape[0]}"
        )

    R_gt = X_gt[:, :3, :3]
    v_gt = X_gt[:, :3, 3]
    p_gt = X_gt[:, :3, 4]
    learned_bias = np.column_stack([sol_pred[:, 4:7], sol_pred[:, 13:16]])
    gt_bias = compute_gt_differenced_bias(
        t,
        imu_raw,
        R_gt,
        v_gt,
        np.asarray(args.gravity, dtype=np.float64),
        args.gt_bias_sg_window_bg,
        args.gt_bias_sg_window_ba,
        args.gt_bias_sg_order_bg,
        args.gt_bias_sg_order_ba,
    )
    return SequenceData(sequence, result_dir, result_path, t, imu_raw, learned_bias, gt_bias, R_gt, v_gt, p_gt)


def cumulative_path(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def window_label(args: argparse.Namespace) -> str:
    if getattr(args, "rollout_steps", 0) > 0:
        return f"{args.rollout_steps}-step"
    return f"{args.segment_duration_s:g} s"


def choose_windows(data: SequenceData, args: argparse.Namespace) -> list[Window]:
    t0 = float(data.t[0] + args.edge_margin_s)
    if args.rollout_steps > 0:
        if data.t.shape[0] <= args.rollout_steps + 1:
            raise ValueError(f"{data.sequence} is too short for {args.rollout_steps} rollout steps.")
        latest_start_index = data.t.shape[0] - args.rollout_steps - 1
        t1 = float(data.t[latest_start_index] - args.edge_margin_s)
    else:
        t1 = float(data.t[-1] - args.segment_duration_s - args.edge_margin_s)
    if t1 <= t0:
        raise ValueError(f"{data.sequence} is too short for {window_label(args)} windows and edge margin.")

    candidates: list[Window] = []
    candidate_id = 0
    start_time = t0
    while start_time <= t1:
        start = int(np.searchsorted(data.t, start_time, side="left"))
        if args.rollout_steps > 0:
            stop = start + args.rollout_steps + 1
        else:
            stop = int(np.searchsorted(data.t, data.t[start] + args.segment_duration_s, side="right"))
        if stop - start >= 3:
            actual_duration = float(data.t[stop - 1] - data.t[start])
            if args.rollout_steps > 0 or actual_duration >= args.segment_duration_s * args.min_duration_ratio:
                p = data.p_gt[start:stop]
                path_m = float(cumulative_path(p)[-1])
                disp_m = float(np.linalg.norm(p[-1] - p[0]))
                candidates.append(Window(candidate_id, start, stop, path_m, disp_m))
                candidate_id += 1
        start_time += args.candidate_stride_s

    if not candidates:
        raise ValueError(f"No valid {window_label(args)} candidate windows for {data.sequence}.")

    candidates.sort(key=lambda w: (w.gt_path_m, w.gt_displacement_m), reverse=True)
    selected: list[Window] = []
    for window in candidates:
        start_t = float(data.t[window.start])
        if all(abs(start_t - float(data.t[prev.start])) >= args.nonoverlap_s for prev in selected):
            selected.append(Window(len(selected), window.start, window.stop, window.gt_path_m, window.gt_displacement_m))
        if len(selected) >= args.segments_per_seq:
            break
    return selected


def propagate(
    t: np.ndarray,
    imu_raw: np.ndarray,
    bias: np.ndarray,
    R0: np.ndarray,
    v0: np.ndarray,
    p0: np.ndarray,
    gravity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = np.empty((t.shape[0], 3, 3), dtype=np.float64)
    v = np.empty((t.shape[0], 3), dtype=np.float64)
    p = np.empty((t.shape[0], 3), dtype=np.float64)
    R[0] = R0
    v[0] = v0
    p[0] = p0
    for i in range(t.shape[0] - 1):
        dt = float(t[i + 1] - t[i])
        omega = imu_raw[i, :3] - bias[i, :3]
        acc_body = imu_raw[i, 3:6] - bias[i, 3:6]
        acc_world = R[i] @ acc_body + gravity
        p[i + 1] = p[i] + v[i] * dt
        v[i + 1] = v[i] + acc_world * dt
        R[i + 1] = R[i] @ so3_exp(omega * dt)
    return R, v, p


def bias_for_method(method: str, data: SequenceData, start: int, stop: int) -> np.ndarray:
    n = stop - start
    if method == "raw":
        return np.zeros((n, 6), dtype=np.float64)
    if method == "gt_const_start":
        return np.repeat(data.gt_bias[start : start + 1], n, axis=0)
    if method == "ours":
        return data.learned_bias[start:stop].copy()
    raise ValueError(f"Unknown method: {method}")


def drift_metrics(
    t_rel: np.ndarray,
    p_gt: np.ndarray,
    p_est: np.ndarray,
    threshold_m: float,
) -> tuple[np.ndarray, dict[str, float | None]]:
    error = np.linalg.norm(p_est - p_gt, axis=1)
    gt_path = cumulative_path(p_gt)
    crossed = np.flatnonzero(error >= threshold_m)
    if crossed.shape[0] == 0:
        diverge_time_s = None
        diverge_gt_path_m = None
        accepted_duration_s = float(t_rel[-1])
        accepted_gt_path_m = float(gt_path[-1])
    else:
        idx = int(crossed[0])
        diverge_time_s = float(t_rel[idx])
        diverge_gt_path_m = float(gt_path[idx])
        accepted_duration_s = float(t_rel[idx])
        accepted_gt_path_m = float(gt_path[idx])
    metrics: dict[str, float | None] = {
        "pos_rmse_m": float(np.sqrt(np.mean(error * error))),
        "pos_mean_m": float(np.mean(error)),
        "pos_max_m": float(np.max(error)),
        "pos_end_m": float(error[-1]),
        "diverge_time_s": diverge_time_s,
        "diverge_gt_path_m": diverge_gt_path_m,
        "accepted_duration_s": accepted_duration_s,
        "accepted_gt_path_m": accepted_gt_path_m,
    }
    return error, metrics


def evaluate_window(data: SequenceData, window: Window, role: str, args: argparse.Namespace) -> SegmentResult:
    start, stop = window.start, window.stop
    t = data.t[start:stop]
    t_rel = t - t[0]
    gravity = np.asarray(args.gravity, dtype=np.float64)
    p_gt = data.p_gt[start:stop]
    method_results: dict[str, MethodResult] = {}

    gt_error = np.zeros((stop - start,), dtype=np.float64)
    method_results["gt"] = MethodResult(
        method="gt",
        R=data.R_gt[start:stop],
        v=data.v_gt[start:stop],
        p=p_gt,
        error_m=gt_error,
        metrics={
            "pos_rmse_m": 0.0,
            "pos_mean_m": 0.0,
            "pos_max_m": 0.0,
            "pos_end_m": 0.0,
            "diverge_time_s": None,
            "diverge_gt_path_m": None,
            "accepted_duration_s": float(t_rel[-1]),
            "accepted_gt_path_m": float(cumulative_path(p_gt)[-1]),
        },
    )

    for method in args.methods:
        bias = bias_for_method(method, data, start, stop)
        R_est, v_est, p_est = propagate(
            t,
            data.imu_raw[start:stop],
            bias,
            data.R_gt[start],
            data.v_gt[start],
            data.p_gt[start],
            gravity,
        )
        error, metrics = drift_metrics(t_rel, p_gt, p_est, args.diverge_threshold_m)
        method_results[method] = MethodResult(method, R_est, v_est, p_est, error, metrics)

    return SegmentResult(data.sequence, role, data.result_dir, window, float(t[0]), t_rel, p_gt, method_results)


def parse_mapping(items: list[str] | None, defaults: dict[str, Path | str]) -> dict[str, Path | str]:
    mapping: dict[str, Path | str] = dict(defaults)
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected SEQ=value mapping, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty sequence key in mapping {item!r}")
        default_value = next(iter(defaults.values()), "")
        if isinstance(default_value, Path):
            mapping[key] = Path(value)
        else:
            mapping[key] = value
    return mapping


def fmt_metric(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "not reached"
    return f"{value:.{digits}f}"


def csv_value(value: float | None) -> str | float:
    return "" if value is None else value


def make_summary_rows(segments: list[SegmentResult], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in segments:
        w = segment.window
        for method in args.methods:
            result = segment.method_results[method]
            row: dict[str, Any] = {
                "sequence": segment.sequence,
                "role": segment.role,
                "segment_id": w.segment_id,
                "result_dir": str(segment.result_dir),
                "start_time_s": segment.start_time_s,
                "duration_s": float(segment.t_rel[-1]),
                "samples": int(segment.t_rel.shape[0]),
                "gt_path_m": w.gt_path_m,
                "gt_displacement_m": w.gt_displacement_m,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "diverge_threshold_m": args.diverge_threshold_m,
            }
            row.update({key: csv_value(value) for key, value in result.metrics.items()})
            rows.append(row)
    return rows


def write_summary_files(rows: list[dict[str, Any]], output_dir: Path) -> None:
    fields = list(rows[0].keys())
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def write_trajectory_csv(segment: SegmentResult, out_dir: Path) -> Path:
    path = out_dir / f"{segment.sequence}_seg{segment.window.segment_id:02d}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        fields = ["sequence", "role", "segment_id", "method", "rel_time_s", "x_m", "y_m", "z_m", "position_error_m"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        origin = segment.p_gt[0]
        for method, result in segment.method_results.items():
            points = result.p - origin
            for i in range(points.shape[0]):
                writer.writerow(
                    {
                        "sequence": segment.sequence,
                        "role": segment.role,
                        "segment_id": segment.window.segment_id,
                        "method": method,
                        "rel_time_s": float(segment.t_rel[i]),
                        "x_m": float(points[i, 0]),
                        "y_m": float(points[i, 1]),
                        "z_m": float(points[i, 2]),
                        "position_error_m": float(result.error_m[i]),
                    }
                )
    return path


def project_points(points: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
    return points[:, axes]


def svg_polyline(points_2d: np.ndarray, bounds: tuple[float, float, float, float], width: int, height: int, pad: int) -> str:
    xmin, xmax, ymin, ymax = bounds
    span_x = xmax - xmin
    span_y = ymax - ymin
    if span_x <= 1.0e-12:
        span_x = 1.0
        xmin -= 0.5
    if span_y <= 1.0e-12:
        span_y = 1.0
        ymin -= 0.5
    x = pad + (points_2d[:, 0] - xmin) / span_x * (width - 2 * pad)
    y = height - pad - (points_2d[:, 1] - ymin) / span_y * (height - 2 * pad)
    return " ".join(f"{float(xi):.1f},{float(yi):.1f}" for xi, yi in zip(x, y))


def bounds_for_paths(paths: list[np.ndarray], axes: tuple[int, int]) -> tuple[float, float, float, float]:
    projected = np.vstack([project_points(path, axes) for path in paths])
    xmin, ymin = np.min(projected, axis=0)
    xmax, ymax = np.max(projected, axis=0)
    margin_x = max((xmax - xmin) * 0.08, 0.05)
    margin_y = max((ymax - ymin) * 0.08, 0.05)
    return float(xmin - margin_x), float(xmax + margin_x), float(ymin - margin_y), float(ymax + margin_y)


def render_projection(segment: SegmentResult, axes: tuple[int, int], title: str) -> str:
    width, height, pad = 520, 360, 34
    origin = segment.p_gt[0]
    rel_paths = {method: result.p - origin for method, result in segment.method_results.items()}
    bounds = bounds_for_paths(list(rel_paths.values()), axes)
    lines = [
        f"<figure><figcaption>{html.escape(title)}</figcaption>",
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#d1d5db"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#d1d5db"/>',
    ]
    for method in PLOT_METHODS:
        points = svg_polyline(project_points(rel_paths[method], axes), bounds, width, height, pad)
        width_px = 3 if method in {"gt", "ours"} else 2
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{METHOD_COLORS[method]}" '
            f'stroke-width="{width_px}" stroke-linejoin="round" stroke-linecap="round"/>'
        )
    lines.append("</svg></figure>")
    return "\n".join(lines)


def render_error_plot(segment: SegmentResult, threshold_m: float) -> str:
    width, height, pad = 520, 300, 34
    max_error = max(float(np.max(result.error_m)) for result in segment.method_results.values())
    ymax = max(max_error, threshold_m, 0.1) * 1.12
    xmax = float(segment.t_rel[-1]) if segment.t_rel[-1] > 0 else 1.0

    def line_points(result: MethodResult) -> str:
        x = pad + segment.t_rel / xmax * (width - 2 * pad)
        y = height - pad - result.error_m / ymax * (height - 2 * pad)
        return " ".join(f"{float(xi):.1f},{float(yi):.1f}" for xi, yi in zip(x, y))

    threshold_y = height - pad - threshold_m / ymax * (height - 2 * pad)
    lines = [
        "<figure><figcaption>position error vs GT</figcaption>",
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="position error">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#d1d5db"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#d1d5db"/>',
        f'<line x1="{pad}" y1="{threshold_y:.1f}" x2="{width - pad}" y2="{threshold_y:.1f}" '
        'stroke="#dc2626" stroke-dasharray="5 4"/>',
    ]
    for method in ["raw", "ours"]:
        result = segment.method_results[method]
        lines.append(
            f'<polyline points="{line_points(result)}" fill="none" stroke="{METHOD_COLORS[method]}" '
            f'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
        )
    lines.append("</svg></figure>")
    return "\n".join(lines)


def render_legend() -> str:
    items = []
    for method in PLOT_METHODS:
        items.append(
            f'<span class="legend-item"><span class="swatch" style="background:{METHOD_COLORS[method]}"></span>'
            f"{html.escape(METHOD_LABELS[method])}</span>"
        )
    return '<div class="legend">' + "\n".join(items) + "</div>"


def render_segment_metrics(segment: SegmentResult) -> str:
    rows = []
    for method in ["raw", "ours"]:
        result = segment.method_results[method]
        metrics = result.metrics
        rows.append(
            "<tr>"
            f"<td>{html.escape(METHOD_LABELS[method])}</td>"
            f"<td>{fmt_metric(metrics['pos_end_m'])}</td>"
            f"<td>{fmt_metric(metrics['pos_max_m'])}</td>"
            f"<td>{fmt_metric(metrics['diverge_time_s'])}</td>"
            f"<td>{fmt_metric(metrics['accepted_gt_path_m'])}</td>"
            "</tr>"
        )
    return (
        '<table class="metrics"><thead><tr>'
        "<th>method</th><th>end err m</th><th>max err m</th><th>diverge time s</th><th>accepted GT path m</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )


def render_segment_html(segment: SegmentResult, threshold_m: float, standalone: bool) -> str:
    title = f"{segment.sequence} {segment.role} seg{segment.window.segment_id:02d}"
    body = [
        f"<section class='segment' id='{html.escape(segment.sequence)}-seg{segment.window.segment_id:02d}'>",
        f"<h2>{html.escape(title)}</h2>",
        (
            f"<p>duration {segment.t_rel[-1]:.3f} s, samples {segment.t_rel.shape[0]}, "
            f"GT path {segment.window.gt_path_m:.3f} m, GT displacement {segment.window.gt_displacement_m:.3f} m, "
            f"diverge threshold {threshold_m:.3f} m.</p>"
        ),
        render_legend(),
        render_segment_metrics(segment),
        "<div class='plots'>",
        render_projection(segment, (0, 1), "XY trajectory, common origin"),
        render_projection(segment, (0, 2), "XZ trajectory, common origin"),
        render_projection(segment, (1, 2), "YZ trajectory, common origin"),
        render_error_plot(segment, threshold_m),
        "</div>",
        "</section>",
    ]
    if not standalone:
        return "\n".join(body)
    return html_page(title, "\n".join(body))


def html_page(title: str, body: str) -> str:
    css = """
body { font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #111827; background: #f9fafb; }
h1 { font-size: 24px; margin: 0 0 12px; }
h2 { font-size: 18px; margin: 0 0 8px; }
p { color: #4b5563; margin: 4px 0 14px; }
a { color: #2563eb; }
.segment { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin: 0 0 18px; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 10px 0 14px; font-size: 13px; }
.legend-item { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 22px; height: 3px; display: inline-block; border-radius: 2px; }
.metrics { border-collapse: collapse; width: 100%; max-width: 860px; margin: 10px 0 14px; font-size: 13px; }
.metrics th, .metrics td { border: 1px solid #e5e7eb; padding: 6px 8px; text-align: right; }
.metrics th:first-child, .metrics td:first-child { text-align: left; }
.metrics th { background: #f3f4f6; font-weight: 600; }
.plots { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 14px; }
figure { margin: 0; border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; background: #fff; }
figcaption { padding: 8px 10px; border-bottom: 1px solid #e5e7eb; color: #374151; font-size: 13px; font-weight: 600; }
svg { display: block; width: 100%; height: auto; }
.summary-table { border-collapse: collapse; width: 100%; background: #fff; font-size: 13px; margin: 12px 0 20px; }
.summary-table th, .summary-table td { border: 1px solid #e5e7eb; padding: 6px 8px; text-align: right; }
.summary-table th:first-child, .summary-table td:first-child, .summary-table th:nth-child(2), .summary-table td:nth-child(2) { text-align: left; }
.summary-table th { background: #f3f4f6; font-weight: 600; }
"""
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>{css}</style></head><body>"
        f"{body}</body></html>\n"
    )


def write_segment_html(segment: SegmentResult, html_dir: Path, threshold_m: float) -> Path:
    path = html_dir / f"{segment.sequence}_seg{segment.window.segment_id:02d}.html"
    path.write_text(render_segment_html(segment, threshold_m, standalone=True), encoding="utf-8")
    return path


def render_summary_table(rows: list[dict[str, Any]]) -> str:
    table_rows = []
    for row in rows:
        if row["method"] not in {"raw", "ours"}:
            continue
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['sequence']))}</td>"
            f"<td>{html.escape(str(row['role']))}</td>"
            f"<td>{int(row['segment_id'])}</td>"
            f"<td>{html.escape(str(row['method_label']))}</td>"
            f"<td>{float(row['gt_path_m']):.3f}</td>"
            f"<td>{float(row['pos_end_m']):.3f}</td>"
            f"<td>{float(row['pos_max_m']):.3f}</td>"
            f"<td>{fmt_metric(None if row['diverge_time_s'] == '' else float(row['diverge_time_s']))}</td>"
            f"<td>{float(row['accepted_gt_path_m']):.3f}</td>"
            "</tr>"
        )
    return (
        '<table class="summary-table"><thead><tr>'
        "<th>sequence</th><th>role</th><th>seg</th><th>method</th><th>GT path m</th>"
        "<th>end err m</th><th>max err m</th><th>diverge time s</th><th>accepted GT path m</th>"
        "</tr></thead><tbody>"
        + "\n".join(table_rows)
        + "</tbody></table>"
    )


def write_index_html(segments: list[SegmentResult], rows: list[dict[str, Any]], html_dir: Path, args: argparse.Namespace) -> Path:
    title = f"{args.dataset_label} {window_label(args)} Preintegration Drift"
    body = [
        f"<h1>{html.escape(title)}</h1>",
        (
            f"<p>Each selected window starts from its GT pose, velocity, and position. "
            f"Divergence is position error reaching {args.diverge_threshold_m:.3f} m.</p>"
        ),
        render_summary_table(rows),
    ]
    for segment in segments:
        body.append(render_segment_html(segment, args.diverge_threshold_m, standalone=False))
    path = html_dir / "index.html"
    path.write_text(html_page(title, "\n".join(body)), encoding="utf-8")
    return path


def write_markdown_summary(rows: list[dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> Path:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["role"]), str(row["method"])), []).append(row)

    lines = [
        f"# {args.dataset_label} {window_label(args)} Preintegration Drift",
        "",
        f"Divergence threshold: {args.diverge_threshold_m:.3f} m position error to GT at current time.",
        "",
        "| role | method | segments | mean end err m | max err m | best accepted GT path m | mean accepted GT path m |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (role, method), vals in sorted(groups.items()):
        end_err = np.array([float(v["pos_end_m"]) for v in vals], dtype=np.float64)
        max_err = np.array([float(v["pos_max_m"]) for v in vals], dtype=np.float64)
        accepted = np.array([float(v["accepted_gt_path_m"]) for v in vals], dtype=np.float64)
        lines.append(
            f"| {role} | {METHOD_LABELS.get(method, method)} | {len(vals)} | {end_err.mean():.3f} | "
            f"{max_err.max():.3f} | {accepted.max():.3f} | {accepted.mean():.3f} |"
        )

    lines.extend(
        [
            "",
            "## Per Segment",
            "",
            "| sequence | role | seg | method | GT path m | end err m | max err m | diverge time s | accepted GT path m |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        diverge = row["diverge_time_s"]
        diverge_text = "not reached" if diverge == "" else f"{float(diverge):.3f}"
        lines.append(
            f"| {row['sequence']} | {row['role']} | {row['segment_id']} | {row['method_label']} | "
            f"{float(row['gt_path_m']):.3f} | {float(row['pos_end_m']):.3f} | "
            f"{float(row['pos_max_m']):.3f} | {diverge_text} | {float(row['accepted_gt_path_m']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `summary.csv`: machine-readable metrics.",
            "- `summary.json`: same metrics in JSON.",
            f"- `trajectories/*.csv`: common-origin GT/raw/GT-const/ours trajectories for each selected {window_label(args)} window.",
            "- `html/index.html`: HTML report with GT retained in each plot.",
        ]
    )
    path = output_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_metadata(
    output_dir: Path,
    sequence_results: dict[str, Path],
    sequence_roles: dict[str, str],
    args: argparse.Namespace,
) -> None:
    payload = {
        "sequences": args.sequences,
        "dataset_label": args.dataset_label,
        "sequence_results": {seq: str(path) for seq, path in sequence_results.items() if seq in args.sequences},
        "sequence_roles": {seq: sequence_roles.get(seq, "unknown") for seq in args.sequences},
        "window_label": window_label(args),
        "segment_duration_s": args.segment_duration_s,
        "rollout_steps": args.rollout_steps,
        "segments_per_seq": args.segments_per_seq,
        "candidate_stride_s": args.candidate_stride_s,
        "edge_margin_s": args.edge_margin_s,
        "nonoverlap_s": args.nonoverlap_s,
        "diverge_threshold_m": args.diverge_threshold_m,
        "gravity": list(args.gravity),
        "gt_bias_smoothing": {
            "gyro_window": args.gt_bias_sg_window_bg,
            "gyro_order": args.gt_bias_sg_order_bg,
            "accel_window": args.gt_bias_sg_window_ba,
            "accel_order": args.gt_bias_sg_order_ba,
        },
        "methods": args.methods,
    }
    (output_dir / "metadata.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate corldata short-window preintegration drift.")
    parser.add_argument("--dataset-label", default="corldata")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sequences", nargs="+", default=["S00", "S13"])
    parser.add_argument(
        "--sequence-result",
        action="append",
        default=None,
        help="Mapping SEQ=/path/to/results_dir containing Sequnces-results/SEQ/FullTra_result.p. Repeatable.",
    )
    parser.add_argument(
        "--sequence-role",
        action="append",
        default=None,
        help="Mapping SEQ=train|test|... for report labeling. Repeatable.",
    )
    parser.add_argument("--segments-per-seq", type=int, default=4)
    parser.add_argument("--segment-duration-s", type=float, default=1.0)
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=0,
        help="If positive, use exactly this many integration intervals per rollout instead of segment-duration-s.",
    )
    parser.add_argument("--candidate-stride-s", type=float, default=0.5)
    parser.add_argument("--nonoverlap-s", type=float, default=1.0)
    parser.add_argument("--edge-margin-s", type=float, default=5.0)
    parser.add_argument("--min-duration-ratio", type=float, default=0.98)
    parser.add_argument("--diverge-threshold-m", type=float, default=1.0)
    parser.add_argument("--methods", nargs="+", default=["raw", "gt_const_start", "ours"], choices=["raw", "gt_const_start", "ours"])
    parser.add_argument("--gravity", nargs=3, type=float, default=(0.0, 0.0, -9.81), metavar=("GX", "GY", "GZ"))
    parser.add_argument("--gt-bias-sg-window-bg", type=int, default=101)
    parser.add_argument("--gt-bias-sg-order-bg", type=int, default=3)
    parser.add_argument("--gt-bias-sg-window-ba", type=int, default=31)
    parser.add_argument("--gt-bias-sg-order-ba", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.segments_per_seq <= 0:
        raise ValueError("--segments-per-seq must be positive.")
    if args.segment_duration_s <= 0.0:
        raise ValueError("--segment-duration-s must be positive.")
    if args.rollout_steps < 0:
        raise ValueError("--rollout-steps must be non-negative.")
    if args.candidate_stride_s <= 0.0:
        raise ValueError("--candidate-stride-s must be positive.")
    if args.diverge_threshold_m <= 0.0:
        raise ValueError("--diverge-threshold-m must be positive.")

    sequence_results_raw = parse_mapping(args.sequence_result, DEFAULT_SEQUENCE_RESULTS)
    sequence_results = {seq: Path(value) for seq, value in sequence_results_raw.items()}
    sequence_roles_raw = parse_mapping(args.sequence_role, DEFAULT_SEQUENCE_ROLES)
    sequence_roles = {seq: str(value) for seq, value in sequence_roles_raw.items()}

    output_dir = args.output_dir
    trajectories_dir = output_dir / "trajectories"
    html_dir = output_dir / "html"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    segments: list[SegmentResult] = []
    for sequence in args.sequences:
        if sequence not in sequence_results:
            raise ValueError(f"No result dir configured for {sequence}. Use --sequence-result {sequence}=DIR.")
        data = load_sequence(sequence, sequence_results[sequence], args)
        role = sequence_roles.get(sequence, "unknown")
        for window in choose_windows(data, args):
            segment = evaluate_window(data, window, role, args)
            segments.append(segment)
            write_trajectory_csv(segment, trajectories_dir)
            write_segment_html(segment, html_dir, args.diverge_threshold_m)

    rows = make_summary_rows(segments, args)
    write_summary_files(rows, output_dir)
    write_markdown_summary(rows, output_dir, args)
    write_index_html(segments, rows, html_dir, args)
    write_metadata(output_dir, sequence_results, sequence_roles, args)

    print(f"Wrote {len(segments)} segments to {output_dir}")
    print(f"HTML report: {html_dir / 'index.html'}")
    print(f"Summary: {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
