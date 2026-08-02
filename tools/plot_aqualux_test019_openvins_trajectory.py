#!/usr/bin/env python3
"""Regenerate the AquaLux Test_019 first-frame SE(3) trajectory figure."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def find_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REPO_ROOT = find_repo_root()
DEFAULT_RESULT_DIR = (
    REPO_ROOT / "Results" / "AquaLux" / "openvins_test" / "vo_se3_mse_acc_nogate" / "Test_019"
)
DEFAULT_RAW = REPO_ROOT / "Results" / "AquaLux" / "openvins_test" / "raw" / "Test_019" / "estimate_tum.tum"
DEFAULT_OURS = DEFAULT_RESULT_DIR / "estimate_tum.tum"
DEFAULT_GT = DEFAULT_RESULT_DIR / "gt_imu_gravity_aligned.tum"
DEFAULT_OUTPUT = DEFAULT_RESULT_DIR / "test019_raw_ours_gt_se3_aligned_xy_3d.png"
DEFAULT_T_MIN = 1779620742.494100094
DEFAULT_T_MAX = 1779620812.966639996


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--ours", type=Path, default=DEFAULT_OURS)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pdf", type=Path, default=None)
    parser.add_argument("--t-min", type=float, default=DEFAULT_T_MIN)
    parser.add_argument("--t-max", type=float, default=DEFAULT_T_MAX)
    parser.add_argument("--t-max-diff", type=float, default=0.02)
    parser.add_argument("--max-points", type=int, default=8000)
    parser.add_argument("--legend-font-size", type=float, default=17.0)
    return parser.parse_args()


def read_tum(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stamps: list[float] = []
    positions: list[list[float]] = []
    quats: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 8:
                continue
            stamps.append(float(parts[0]))
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            quats.append([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
    if not stamps:
        raise ValueError(f"No TUM rows found in {path}")
    return (
        np.asarray(stamps, dtype=np.float64),
        np.asarray(positions, dtype=np.float64),
        np.asarray(quats, dtype=np.float64),
    )


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat / np.linalg.norm(quat)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def crop(
    stamps: np.ndarray,
    positions: np.ndarray,
    quats: np.ndarray,
    t_min: float,
    t_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = (stamps >= t_min) & (stamps <= t_max)
    if not np.any(mask):
        raise ValueError(f"No samples in time range [{t_min}, {t_max}]")
    return stamps[mask], positions[mask], quats[mask]


def nearest_index(stamps: np.ndarray, stamp: float, max_diff: float) -> int:
    idx = int(np.argmin(np.abs(stamps - stamp)))
    diff = abs(float(stamps[idx] - stamp))
    if diff > max_diff:
        raise ValueError(f"Nearest timestamp differs by {diff:.6f}s, above {max_diff:.6f}s")
    return idx


def first_frame_positions(
    stamps: np.ndarray,
    positions: np.ndarray,
    quats: np.ndarray,
    gt_stamps: np.ndarray,
    gt_positions: np.ndarray,
    gt_quats: np.ndarray,
    t_max_diff: float,
) -> np.ndarray:
    gt0_idx = nearest_index(gt_stamps, float(stamps[0]), t_max_diff)
    p_gt0 = gt_positions[gt0_idx]
    r_gt0 = quat_xyzw_to_matrix(gt_quats[gt0_idx])
    r_est0 = quat_xyzw_to_matrix(quats[0])
    p_est0 = positions[0]

    r_align = r_gt0 @ r_est0.T
    p_align = p_gt0 - r_align @ p_est0
    aligned_world = (r_align @ positions.T).T + p_align
    return (r_gt0.T @ (aligned_world - p_gt0).T).T


def gt_first_frame_positions(positions: np.ndarray, quats: np.ndarray) -> np.ndarray:
    p0 = positions[0]
    r0 = quat_xyzw_to_matrix(quats[0])
    return (r0.T @ (positions - p0).T).T


def downsample(positions: np.ndarray, max_points: int) -> np.ndarray:
    if len(positions) <= max_points:
        return positions
    step = int(np.ceil(len(positions) / max_points))
    return positions[::step]


def set_equal_3d(ax: plt.Axes, *position_sets: np.ndarray) -> None:
    all_points = np.vstack(position_sets)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = 0.5 * float(np.max(maxs - mins))
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def plot_endpoints(ax: plt.Axes, xy: np.ndarray, color: str, is_3d: bool = False) -> None:
    if is_3d:
        ax.scatter(xy[0, 0], xy[0, 1], xy[0, 2], color=color, marker="o", s=40, depthshade=False)
        ax.scatter(xy[-1, 0], xy[-1, 1], xy[-1, 2], color=color, marker="x", s=48, linewidths=2.2, depthshade=False)
    else:
        ax.scatter(xy[0, 0], xy[0, 1], color=color, marker="o", s=38, zorder=5)
        ax.scatter(xy[-1, 0], xy[-1, 1], color=color, marker="x", s=54, linewidths=2.2, zorder=5)


def make_plot(args: argparse.Namespace) -> Path:
    raw_t, raw_p, raw_q = read_tum(args.raw)
    ours_t, ours_p, ours_q = read_tum(args.ours)
    gt_t, gt_p, gt_q = read_tum(args.gt)

    raw_t, raw_p, raw_q = crop(raw_t, raw_p, raw_q, args.t_min, args.t_max)
    ours_t, ours_p, ours_q = crop(ours_t, ours_p, ours_q, args.t_min, args.t_max)
    gt_t, gt_p, gt_q = crop(gt_t, gt_p, gt_q, args.t_min, args.t_max)

    raw_first = first_frame_positions(raw_t, raw_p, raw_q, gt_t, gt_p, gt_q, args.t_max_diff)
    ours_first = first_frame_positions(ours_t, ours_p, ours_q, gt_t, gt_p, gt_q, args.t_max_diff)
    gt_first = gt_first_frame_positions(gt_p, gt_q)

    raw_plot = downsample(raw_first, args.max_points)
    ours_plot = downsample(ours_first, args.max_points)
    gt_plot = downsample(gt_first, args.max_points)

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 19,
            "axes.labelsize": 16,
            "axes.grid": True,
            "grid.alpha": 0.32,
            "legend.frameon": True,
        }
    )
    fig = plt.figure(figsize=(17.0, 8.125), constrained_layout=True)
    ax_xy = fig.add_subplot(1, 2, 1)
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")

    raw_color = "#7f7f7f"
    ours_color = "#1f77b4"
    gt_color = "#2ca02c"

    ax_xy.plot(raw_plot[:, 0], raw_plot[:, 1], linestyle="--", color=raw_color, linewidth=2.1, label="Raw")
    ax_xy.plot(ours_plot[:, 0], ours_plot[:, 1], color=ours_color, linewidth=2.3, label="VO-NLL")
    ax_xy.plot(gt_plot[:, 0], gt_plot[:, 1], color=gt_color, linewidth=2.3, label="Ground truth")
    for positions, color in ((raw_plot, raw_color), (ours_plot, ours_color), (gt_plot, gt_color)):
        plot_endpoints(ax_xy, positions, color)
    ax_xy.set_title("Test_019 XY top-down")
    ax_xy.set_xlabel("x in first-frame coordinates (m)")
    ax_xy.set_ylabel("y in first-frame coordinates (m)")
    ax_xy.axis("equal")
    ax_xy.legend(loc="upper right", fontsize=args.legend_font_size)

    ax_3d.plot(raw_plot[:, 0], raw_plot[:, 1], raw_plot[:, 2], linestyle="--", color=raw_color, linewidth=2.1, label="Raw")
    ax_3d.plot(ours_plot[:, 0], ours_plot[:, 1], ours_plot[:, 2], color=ours_color, linewidth=2.3, label="VO-NLL")
    ax_3d.plot(gt_plot[:, 0], gt_plot[:, 1], gt_plot[:, 2], color=gt_color, linewidth=2.3, label="Ground truth")
    for positions, color in ((raw_plot, raw_color), (ours_plot, ours_color), (gt_plot, gt_color)):
        plot_endpoints(ax_3d, positions, color, is_3d=True)
    ax_3d.set_title("Test_019 3D")
    ax_3d.set_xlabel("x (m)", labelpad=10)
    ax_3d.set_ylabel("y (m)", labelpad=10)
    ax_3d.set_zlabel("z (m)", labelpad=10)
    ax_3d.view_init(elev=26, azim=-60)
    set_equal_3d(ax_3d, raw_plot, ours_plot, gt_plot)
    ax_3d.legend(loc="upper right", fontsize=args.legend_font_size)

    fig.suptitle("AquaLux Test_019 SE(3) First-Frame Aligned Trajectories", fontsize=19)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=144)
    if args.pdf is not None:
        args.pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.pdf)
    plt.close(fig)
    return args.output


def main() -> int:
    args = parse_args()
    output = make_plot(args)
    print(f"wrote {output}")
    if args.pdf is not None:
        print(f"wrote {args.pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
