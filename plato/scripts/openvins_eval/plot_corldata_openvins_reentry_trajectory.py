#!/usr/bin/env python3
"""Plot corldata OpenVINS re-entry trajectory comparisons."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

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


def aligned_run(summary, seq: str, run_root: Path, alignment: str):
    est_path = run_root / seq / "estimate.txt"
    gt_path = summary.DEFAULT_GT_ROOT / seq / "mav0" / "mocap0" / "data.csv"
    t_est, p_est, r_est = summary.load_openvins_estimate(est_path)
    t_gt, p_gt, r_gt = summary.load_gt(gt_path)
    valid, p_gt_i, r_gt_i = summary.interp_gt(t_est, t_gt, p_gt, r_gt)
    t = t_est[valid]
    p_est = p_est[valid]
    r_est = r_est[valid]
    segments = summary.load_segments(summary.DEFAULT_LIGHT_SEGMENTS, seq)
    if alignment == "first_light_on":
        fit_mask = summary.first_light_on_mask(t, segments)
    elif alignment == "all":
        fit_mask = np.ones_like(t, dtype=bool)
    else:
        raise ValueError(f"Unknown alignment {alignment}")
    r_align, t_align = summary.fit_rigid_no_scale(p_est[fit_mask], p_gt_i[fit_mask])
    p_aligned = (r_align @ p_est.T).T + t_align
    err = np.linalg.norm(p_aligned - p_gt_i, axis=1)
    return {
        "t": t,
        "t_rel": t - t[0],
        "p": p_aligned,
        "gt": p_gt_i,
        "err": err,
        "segments": segments,
    }


def add_light_spans(ax, t0: float, segments) -> None:
    for seg in segments:
        if seg.state != "light_off":
            continue
        ax.axvspan(seg.start - t0, seg.end - t0, color="#d8d8d8", alpha=0.55, linewidth=0)


def plot_sequence(summary, seq: str, runs: list[tuple[str, Path, str]], alignment: str, output_dir: Path, zoom: bool) -> Path:
    data = [(label, aligned_run(summary, seq, root, alignment), color) for label, root, color in runs]
    ref = data[-1][1]
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_zoom" if zoom else ""
    out_path = output_dir / f"{seq}_{alignment}_reentry_trajectory{suffix}.png"

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 9.0), gridspec_kw={"height_ratios": [1.25, 1.0]})
    ax_xy, ax_err = axes

    ax_xy.plot(ref["gt"][:, 0], ref["gt"][:, 1], color="#111111", linewidth=2.0, label="GT")
    for label, item, color in data:
        ax_xy.plot(item["p"][:, 0], item["p"][:, 1], color=color, linewidth=1.35, label=label)
        ax_err.plot(item["t_rel"], item["err"], color=color, linewidth=1.4, label=label)

    add_light_spans(ax_err, ref["t"][0], ref["segments"])
    ax_xy.set_title(f"{seq} OpenVINS trajectory after dark-to-light re-entry ({alignment} alignment)")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    if zoom:
        final = data[-1][1]
        xy = np.vstack([ref["gt"][:, :2], final["p"][:, :2]])
        lo = xy.min(axis=0)
        hi = xy.max(axis=0)
        center = 0.5 * (lo + hi)
        span = float(max(hi[0] - lo[0], hi[1] - lo[1], 2.0))
        margin = max(0.5 * span, 2.0)
        ax_xy.set_xlim(center[0] - 0.5 * span - margin, center[0] + 0.5 * span + margin)
        ax_xy.set_ylim(center[1] - 0.5 * span - margin, center[1] + 0.5 * span + margin)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True, color="#e5e5e5", linewidth=0.8)
    ax_xy.legend(loc="best", frameon=True)

    ax_err.set_title("Position error over time; gray spans are light-off intervals")
    ax_err.set_xlabel("time from run start [s]")
    ax_err.set_ylabel("position error [m]")
    ax_err.grid(True, color="#e5e5e5", linewidth=0.8)
    ax_err.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", action="append", default=["S13", "S15"])
    parser.add_argument("--alignment", choices=["first_light_on", "all"], default="first_light_on")
    parser.add_argument("--zoom-around-gt", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PLATO_ROOT
        / "results"
        / "openvins_eval"
        / "corldata"
        / "natural"
        / "gated_gyro_ref_duration_reentry_adaptive_gray_runs"
        / "plots",
    )
    args = parser.parse_args()
    summary = load_summary_module()
    runs = [
        ("raw", PLATO_ROOT / "results" / "openvins_eval" / "corldata" / "natural" / "raw", "#8a8a8a"),
        (
            "duration learned gyro",
            PLATO_ROOT / "results" / "openvins_eval" / "corldata" / "natural" / "gated_gyro_ref_duration_runs",
            "#d9822b",
        ),
        (
            "re-entry stabilized",
            PLATO_ROOT
            / "results"
            / "openvins_eval"
            / "corldata"
            / "natural"
            / "gated_gyro_ref_duration_reentry_adaptive_gray_runs",
            "#1f77b4",
        ),
    ]
    for seq in args.seq:
        print(plot_sequence(summary, seq, runs, args.alignment, args.output_dir, args.zoom_around_gt))


if __name__ == "__main__":
    main()
