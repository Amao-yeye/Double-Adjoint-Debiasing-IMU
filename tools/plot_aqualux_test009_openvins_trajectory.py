#!/usr/bin/env python3
"""Plot AquaLux Test_009 raw and compensated OpenVINS trajectories against GT."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Results" / "AquaLux" / "openvins_test" / "Figures"
DEFAULT_DATA_ROOT = Path(
    os.environ.get("AQUALUX_DATA_ROOT", REPO_ROOT / "datasets" / "AquaLux" / "0524_data")
)
DEFAULT_GT = DEFAULT_DATA_ROOT / "0524" / "Test_009" / "ATO" / "gt_imu_gravity_aligned.tum"
DEFAULT_RAW = REPO_ROOT / "Results" / "AquaLux" / "openvins_test" / "raw" / "Test_009" / "estimate_tum.tum"
DEFAULT_COMP = (
    REPO_ROOT
    / "Results"
    / "AquaLux"
    / "openvins_test"
    / "baseline"
    / "segment_delta_acc_covgate"
    / "Test_009"
    / "estimate_tum.tum"
)
DEFAULT_RAW_METRICS = REPO_ROOT / "Results" / "AquaLux" / "openvins_test" / "raw" / "metrics.csv"
DEFAULT_COMP_METRICS = (
    REPO_ROOT / "Results" / "AquaLux" / "openvins_test" / "baseline" / "segment_delta_acc_covgate" / "metrics.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--compensated", type=Path, default=DEFAULT_COMP)
    parser.add_argument("--raw-metrics", type=Path, default=DEFAULT_RAW_METRICS)
    parser.add_argument("--compensated-metrics", type=Path, default=DEFAULT_COMP_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--t-max-diff", type=float, default=0.02)
    parser.add_argument("--sequence", default="Test_009")
    return parser.parse_args()


def read_tum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    stamps: list[float] = []
    positions: list[list[float]] = []
    with path.open(errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 8:
                continue
            stamps.append(float(parts[0]))
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not stamps:
        raise ValueError(f"No TUM trajectory rows in {path}")
    return np.asarray(stamps, dtype=np.float64), np.asarray(positions, dtype=np.float64)


def matched_pairs(
    gt_t: np.ndarray,
    gt_p: np.ndarray,
    est_t: np.ndarray,
    est_p: np.ndarray,
    t_max_diff: float,
) -> tuple[np.ndarray, np.ndarray]:
    gt_idx: list[int] = []
    est_idx: list[int] = []
    j = 0
    for i, t in enumerate(est_t):
        while j + 1 < len(gt_t) and abs(gt_t[j + 1] - t) <= abs(gt_t[j] - t):
            j += 1
        if abs(gt_t[j] - t) <= t_max_diff:
            est_idx.append(i)
            gt_idx.append(j)
    if len(est_idx) < 3:
        raise ValueError("Too few matched poses for SE(3) alignment")
    return est_p[np.asarray(est_idx)], gt_p[np.asarray(gt_idx)]


def align_no_scale(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean
    covariance = src_centered.T @ tgt_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = tgt_mean - rotation @ src_mean
    return rotation, translation


def apply_alignment(positions: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (rotation @ positions.T).T + translation


def downsample(stamps: np.ndarray, positions: np.ndarray, max_points: int = 6000) -> tuple[np.ndarray, np.ndarray]:
    if len(stamps) <= max_points:
        return stamps, positions
    step = int(np.ceil(len(stamps) / max_points))
    return stamps[::step], positions[::step]


def read_metric(path: Path, sequence: str) -> tuple[float, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("sequence") == sequence:
                return float(row["ate_rmse_pos_m"]), float(row["ate_rmse_rot_deg"])
    raise ValueError(f"{sequence} not found in {path}")


def make_plot(args: argparse.Namespace) -> tuple[Path, Path]:
    gt_t, gt_p = read_tum(args.gt)
    raw_t, raw_p = read_tum(args.raw)
    comp_t, comp_p = read_tum(args.compensated)

    raw_match, raw_gt_match = matched_pairs(gt_t, gt_p, raw_t, raw_p, args.t_max_diff)
    comp_match, comp_gt_match = matched_pairs(gt_t, gt_p, comp_t, comp_p, args.t_max_diff)
    raw_r, raw_tvec = align_no_scale(raw_match, raw_gt_match)
    comp_r, comp_tvec = align_no_scale(comp_match, comp_gt_match)
    raw_aligned = apply_alignment(raw_p, raw_r, raw_tvec)
    comp_aligned = apply_alignment(comp_p, comp_r, comp_tvec)

    t_min = max(float(gt_t[0]), min(float(raw_t[0]), float(comp_t[0])))
    t_max = min(float(gt_t[-1]), max(float(raw_t[-1]), float(comp_t[-1])))
    gt_mask = (gt_t >= t_min) & (gt_t <= t_max)
    gt_plot_t, gt_plot_p = downsample(gt_t[gt_mask], gt_p[gt_mask])
    raw_plot_t, raw_plot_p = downsample(raw_t, raw_aligned)
    comp_plot_t, comp_plot_p = downsample(comp_t, comp_aligned)

    raw_pos_rmse, raw_rot_rmse = read_metric(args.raw_metrics, args.sequence)
    comp_pos_rmse, comp_rot_rmse = read_metric(args.compensated_metrics, args.sequence)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(12.5, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax_xy = fig.add_subplot(grid[:, 0])
    ax_z = fig.add_subplot(grid[0, 1])
    ax_err = fig.add_subplot(grid[1, 1])

    ax_xy.plot(gt_plot_p[:, 0], gt_plot_p[:, 1], color="black", linewidth=2.2, label="GT")
    ax_xy.plot(raw_plot_p[:, 0], raw_plot_p[:, 1], color="#d95f02", linewidth=1.5, label="Raw OpenVINS")
    ax_xy.plot(comp_plot_p[:, 0], comp_plot_p[:, 1], color="#1b9e77", linewidth=1.7, label="Compensated")
    ax_xy.set_title("Test_009 Trajectory, XY")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.axis("equal")
    ax_xy.legend(loc="best")

    time0 = min(raw_plot_t[0], comp_plot_t[0], gt_plot_t[0])
    ax_z.plot(gt_plot_t - time0, gt_plot_p[:, 2], color="black", linewidth=2.0, label="GT")
    ax_z.plot(raw_plot_t - time0, raw_plot_p[:, 2], color="#d95f02", linewidth=1.2, label="Raw")
    ax_z.plot(comp_plot_t - time0, comp_plot_p[:, 2], color="#1b9e77", linewidth=1.2, label="Compensated")
    ax_z.set_title("Height Over Time")
    ax_z.set_xlabel("time [s]")
    ax_z.set_ylabel("z [m]")

    raw_err_est, raw_err_gt = matched_pairs(gt_t, gt_p, raw_t, raw_aligned, args.t_max_diff)
    comp_err_est, comp_err_gt = matched_pairs(gt_t, gt_p, comp_t, comp_aligned, args.t_max_diff)
    raw_err = np.linalg.norm(raw_err_est - raw_err_gt, axis=1)
    comp_err = np.linalg.norm(comp_err_est - comp_err_gt, axis=1)
    raw_err_t = raw_t[: len(raw_err)]
    comp_err_t = comp_t[: len(comp_err)]
    raw_err_t, raw_err = downsample(raw_err_t, raw_err[:, None])
    comp_err_t, comp_err = downsample(comp_err_t, comp_err[:, None])
    ax_err.plot(raw_err_t - time0, raw_err[:, 0], color="#d95f02", linewidth=1.2, label="Raw")
    ax_err.plot(comp_err_t - time0, comp_err[:, 0], color="#1b9e77", linewidth=1.2, label="Compensated")
    ax_err.set_title("Translation Error After SE(3) Alignment")
    ax_err.set_xlabel("time [s]")
    ax_err.set_ylabel("error [m]")
    ax_err.legend(loc="best")

    fig.suptitle(
        "AquaLux Test_009 OpenVINS: Raw vs Gated Acc-Delta Compensation\n"
        f"Raw RMSE: {raw_pos_rmse:.3f} m / {raw_rot_rmse:.3f} deg; "
        f"Compensated RMSE: {comp_pos_rmse:.3f} m / {comp_rot_rmse:.3f} deg",
        fontsize=13,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png = args.output_dir / "Test_009_openvins_raw_vs_compensated_gt.png"
    pdf = args.output_dir / "Test_009_openvins_raw_vs_compensated_gt.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main() -> int:
    args = parse_args()
    png, pdf = make_plot(args)
    print(f"wrote {png}")
    print(f"wrote {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
