#!/usr/bin/env python3
"""Plot corldata raw OpenVINS against the best PLATO OpenVINS result."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PLATO_ROOT = Path(__file__).resolve().parents[3]
SUMMARY_PATH = PLATO_ROOT / "plato" / "scripts" / "openvins_eval" / "summarize_corldata_openvins_natural.py"


def load_summary_module():
    spec = importlib.util.spec_from_file_location("corldata_openvins_summary", SUMMARY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SUMMARY_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def aligned_run(summary, seq: str, run_root: Path, alignment: str) -> dict[str, np.ndarray]:
    est_path = run_root / seq / "estimate.txt"
    gt_path = summary.DEFAULT_GT_ROOT / seq / "mav0" / "mocap0" / "data.csv"

    t_est, p_est, _ = summary.load_openvins_estimate(est_path)
    t_gt, p_gt, r_gt = summary.load_gt(gt_path)
    valid, p_gt_i, _ = summary.interp_gt(t_est, t_gt, p_gt, r_gt)
    t = t_est[valid]
    p_est = p_est[valid]

    segments = summary.load_segments(summary.DEFAULT_LIGHT_SEGMENTS, seq)
    if alignment == "first_light_on":
        fit_mask = summary.first_light_on_mask(t, segments)
    elif alignment == "all":
        fit_mask = np.ones_like(t, dtype=bool)
    else:
        raise ValueError(f"Unknown alignment: {alignment}")

    r_align, t_align = summary.fit_rigid_no_scale(p_est[fit_mask], p_gt_i[fit_mask])
    p_aligned = (r_align @ p_est.T).T + t_align
    err = np.linalg.norm(p_aligned - p_gt_i, axis=1)
    return {"t": t, "t_rel": t - t[0], "p": p_aligned, "gt_interp": p_gt_i, "err": err}


def load_gt_track(summary, seq: str, t_min: float, t_max: float) -> tuple[np.ndarray, np.ndarray, list]:
    gt_path = summary.DEFAULT_GT_ROOT / seq / "mav0" / "mocap0" / "data.csv"
    t_gt, p_gt, _ = summary.load_gt(gt_path)
    keep = (t_gt >= t_min) & (t_gt <= t_max)
    return t_gt[keep], p_gt[keep], summary.load_segments(summary.DEFAULT_LIGHT_SEGMENTS, seq)


def plot_gt_by_light(
    ax,
    t_gt: np.ndarray,
    p_gt: np.ndarray,
    segments: list,
    *,
    lw: float = 2.0,
    labels: bool = True,
    x_col: int = 0,
    y_col: int = 1,
) -> None:
    handles = {}
    for seg in segments:
        mask = (t_gt >= seg.start) & (t_gt <= seg.end)
        if int(mask.sum()) < 2:
            continue
        if seg.state == "light_on":
            color = "#111111"
            label = "GT light on" if labels else None
        elif seg.state == "light_off":
            color = "#b8b8b8"
            label = "GT light off" if labels else None
        else:
            color = "#dddddd"
            label = "GT unknown" if labels else None
        line = ax.plot(p_gt[mask, x_col], p_gt[mask, y_col], color=color, linewidth=lw, label=label, zorder=3)[0]
        if label is not None:
            handles.setdefault(label, line)


def set_zoom_limits(ax, arrays: list[np.ndarray], *, margin_fraction: float = 0.15, min_margin: float = 1.0) -> None:
    xy = np.vstack([arr[:, :2] for arr in arrays if arr.size])
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    center = 0.5 * (lo + hi)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1], 2.0))
    margin = max(margin_fraction * span, min_margin)
    ax.set_xlim(center[0] - 0.5 * span - margin, center[0] + 0.5 * span + margin)
    ax.set_ylim(center[1] - 0.5 * span - margin, center[1] + 0.5 * span + margin)


def add_light_spans(ax, t0: float, segments: list) -> None:
    for seg in segments:
        if seg.state != "light_off":
            continue
        ax.axvspan(seg.start - t0, seg.end - t0, color="#d8d8d8", alpha=0.55, linewidth=0)


def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err * err)))


def time_window_mask(t: np.ndarray, t0: float, max_time_s: Optional[float]) -> np.ndarray:
    if max_time_s is None:
        return np.ones_like(t, dtype=bool)
    return (t - t0) <= max_time_s


def plot_sequence(
    summary,
    seq: str,
    raw_root: Path,
    ours_root: Path,
    output_dir: Path,
    alignment: str,
    max_time_s: Optional[float],
) -> Path:
    raw = aligned_run(summary, seq, raw_root, alignment)
    ours = aligned_run(summary, seq, ours_root, alignment)

    t0 = min(float(raw["t"][0]), float(ours["t"][0]))
    raw_mask = time_window_mask(raw["t"], t0, max_time_s)
    ours_mask = time_window_mask(ours["t"], t0, max_time_s)
    if not np.any(raw_mask) or not np.any(ours_mask):
        raise ValueError(f"No data inside requested time window for {seq}")

    t_min = t0
    t_max = t0 + max_time_s if max_time_s is not None else max(float(raw["t"][-1]), float(ours["t"][-1]))
    t_gt, p_gt, segments = load_gt_track(summary, seq, t_min, t_max)

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_first{int(max_time_s)}s" if max_time_s is not None else ""
    out_path = output_dir / f"{seq}_raw_vs_ours_error_{alignment}{suffix}.png"

    raw_label = f"raw RMSE {rmse(raw['err'][raw_mask]):.2f} m"
    ours_label = f"ours RMSE {rmse(ours['err'][ours_mask]):.2f} m"

    fig = plt.figure(figsize=(13.8, 9.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
    ax_full = fig.add_subplot(grid[0, 0])
    ax_zoom = fig.add_subplot(grid[0, 1])
    ax_err = fig.add_subplot(grid[1, :])

    for ax, title, labels in [
        (ax_full, f"{seq} full trajectory", True),
        (ax_zoom, f"{seq} zoomed trajectory", False),
    ]:
        full_panel = ax is ax_full
        x_col = 1 if full_panel else 0
        y_col = 0 if full_panel else 1
        plot_gt_by_light(ax, t_gt, p_gt, segments, labels=labels, x_col=x_col, y_col=y_col)
        ax.plot(
            raw["p"][raw_mask, x_col],
            raw["p"][raw_mask, y_col],
            color="#f2c230",
            linewidth=1.6,
            label=raw_label if labels else None,
            zorder=2,
        )
        ax.plot(
            ours["p"][ours_mask, x_col],
            ours["p"][ours_mask, y_col],
            color="#1f77b4",
            linewidth=1.9,
            label=ours_label if labels else None,
            zorder=4,
        )
        ax.set_xlabel("y [m]" if full_panel else "x [m]")
        ax.set_ylabel("x [m]" if full_panel else "y [m]")
        ax.set_title(title)
        ax.grid(True, color="#e6e6e6", linewidth=0.8)

    for ax in [ax_full, ax_zoom]:
        ax.set_aspect("equal", adjustable="datalim")
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(0.52)
    set_zoom_limits(ax_zoom, [p_gt, ours["p"][ours_mask]], margin_fraction=0.05, min_margin=0.20)

    add_light_spans(ax_err, t0, segments)
    ax_err.plot(raw["t"][raw_mask] - t0, raw["err"][raw_mask], color="#f2c230", linewidth=1.5, label=raw_label)
    ax_err.plot(ours["t"][ours_mask] - t0, ours["err"][ours_mask], color="#1f77b4", linewidth=1.6, label=ours_label)
    if max_time_s is not None:
        ax_err.set_xlim(0.0, max_time_s)
    ax_err.set_title("Position error over time; gray spans are light-off intervals")
    ax_err.set_xlabel("time from run start [s]")
    ax_err.set_ylabel("position error [m]")
    ax_err.grid(True, color="#e6e6e6", linewidth=0.8)

    handles, labels = ax_full.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    order = ["GT light on", "GT light off", raw_label, ours_label]
    ax_full.legend([by_label[label] for label in order if label in by_label], [label for label in order if label in by_label])
    ax_err.legend(loc="best", frameon=True)

    fig.suptitle(f"corldata {seq}: raw OpenVINS vs ours", y=0.995)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", action="append", default=["S13", "S15"])
    parser.add_argument("--alignment", choices=["first_light_on", "all"], default="first_light_on")
    parser.add_argument("--max-time-s", type=float, default=40.0, help="Only plot this many seconds from run start.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=PLATO_ROOT / "results" / "openvins_eval" / "corldata" / "natural" / "raw",
    )
    parser.add_argument(
        "--ours-root",
        type=Path,
        default=PLATO_ROOT
        / "results"
        / "openvins_eval"
        / "corldata"
        / "natural"
        / "gated_gyro_ref_duration_reentry_adaptive_gray_runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PLATO_ROOT / "results" / "openvins_eval" / "corldata" / "natural" / "raw_vs_ours_plots",
    )
    args = parser.parse_args()

    summary = load_summary_module()
    for seq in args.seq:
        print(plot_sequence(summary, seq, args.raw_root, args.ours_root, args.output_dir, args.alignment, args.max_time_s))


if __name__ == "__main__":
    main()
