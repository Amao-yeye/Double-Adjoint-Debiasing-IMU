#!/usr/bin/env python3
"""Evaluate OpenVINS trajectories after a constant rotation alignment."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PLATO_ROOT = Path(__file__).resolve().parents[3]
METHODS = ["gt_mse", "gt_nll", "vo_mse", "vo_nll"]
LABEL = {
    "gt_mse": "GT SE23-MSE",
    "gt_nll": "GT NLL",
    "vo_mse": "VO SE23-MSE",
    "vo_nll": "VO NLL",
}


@dataclass(frozen=True)
class Trajectory:
    stamp: np.ndarray
    p: np.ndarray
    r: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/home/amao/workspace/formalPLATO/Results/openvins_imu/openvins_results_rigorous_rerun"),
        help="OpenVINS result root containing method/sequence/odomimu.txt.",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=PLATO_ROOT / "external/OpenVINS/src/open_vins/ov_data/euroc_mav",
        help="OpenVINS EuRoC GT TUM files.",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=["MH_02_easy", "MH_04_difficult", "V1_01_easy", "V1_03_difficult", "V2_02_medium"],
    )
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--max-time-diff", type=float, default=0.02)
    parser.add_argument(
        "--output-stem",
        default="rotation_aligned_metric",
        help="Output filename stem inside run-root.",
    )
    return parser.parse_args()


def quat_xyzw_to_rot(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm <= 0:
        raise ValueError("zero quaternion")
    x, y, z, w = q / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def read_tum(path: Path) -> Trajectory:
    stamps: list[float] = []
    positions: list[list[float]] = []
    rotations: list[np.ndarray] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            stamps.append(float(parts[0]))
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            rotations.append(quat_xyzw_to_rot(np.array([float(v) for v in parts[4:8]], dtype=float)))
    if not stamps:
        raise RuntimeError(f"No poses loaded from {path}")
    return Trajectory(np.asarray(stamps), np.asarray(positions), np.asarray(rotations))


def associate(gt: Trajectory, est: Trajectory, max_time_diff: float) -> tuple[np.ndarray, np.ndarray]:
    gt_idx: list[int] = []
    est_idx: list[int] = []
    j = 0
    for i, t in enumerate(est.stamp):
        while j + 1 < len(gt.stamp) and abs(gt.stamp[j + 1] - t) <= abs(gt.stamp[j] - t):
            j += 1
        if abs(gt.stamp[j] - t) <= max_time_diff:
            gt_idx.append(j)
            est_idx.append(i)
    if not gt_idx:
        raise RuntimeError("No timestamp associations found")
    return np.asarray(gt_idx), np.asarray(est_idx)


def project_so3(r: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(r)
    c = u @ vt
    if np.linalg.det(c) < 0:
        u[:, -1] *= -1.0
        c = u @ vt
    return c


def rotation_angle_deg(r: np.ndarray) -> float:
    cos_theta = (float(np.trace(r)) - 1.0) * 0.5
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def constant_rotation_alignment(r_gt: np.ndarray, r_est: np.ndarray) -> np.ndarray:
    accum = np.zeros((3, 3), dtype=float)
    for rg, re in zip(r_gt, r_est):
        accum += rg @ re.T
    return project_so3(accum)


def evaluate_pair(gt_path: Path, est_path: Path, max_time_diff: float) -> dict[str, float | int]:
    gt = read_tum(gt_path)
    est = read_tum(est_path)
    gt_idx, est_idx = associate(gt, est, max_time_diff)
    p_gt = gt.p[gt_idx]
    r_gt = gt.r[gt_idx]
    p_est = est.p[est_idx]
    r_est = est.r[est_idx]

    c = constant_rotation_alignment(r_gt, r_est)
    p_rot = (c @ p_est.T).T
    t = np.mean(p_gt - p_rot, axis=0)
    p_aligned = p_rot + t
    pos_err = np.linalg.norm(p_aligned - p_gt, axis=1)

    rot_err = np.asarray([rotation_angle_deg(rg.T @ c @ re) for rg, re in zip(r_gt, r_est)])
    return {
        "num_matches": int(len(gt_idx)),
        "rot_aligned_pos_rmse_m": float(np.sqrt(np.mean(pos_err * pos_err))),
        "rot_aligned_pos_mean_m": float(np.mean(pos_err)),
        "rot_residual_rmse_deg": float(np.sqrt(np.mean(rot_err * rot_err))),
        "rot_residual_mean_deg": float(np.mean(rot_err)),
        "constant_rot_correction_deg": float(rotation_angle_deg(c)),
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    lines = [
        "# Rotation-Aligned OpenVINS Metric",
        "",
        f"Source: `{args.run_root}`",
        f"GT root: `{args.gt_root}`",
        f"Association threshold: `{args.max_time_diff:.3f} s`",
        "",
        "Definition: solve one constant SO(3) correction from GT/estimate orientations, apply it to positions,",
        "then fit translation only and report position RMSE. Rotation residual is evaluated after the same",
        "constant SO(3) correction.",
        "",
        "## Per-Sequence",
        "",
        "| method | sequence | matches | rot-aligned pos RMSE (m) | rot residual RMSE (deg) | constant rot corr (deg) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method_label']} | {row['sequence']} | {row['num_matches']} | "
            f"{row['rot_aligned_pos_rmse_m']} | {row['rot_residual_rmse_deg']} | "
            f"{row['constant_rot_correction_deg']} |"
        )

    lines.extend(
        [
            "",
            "## Method Average",
            "",
            "| method | sequences | mean rot-aligned pos RMSE (m) | mean rot residual RMSE (deg) |",
            "|---|---:|---:|---:|",
        ]
    )
    for method in args.methods:
        vals = [row for row in rows if row["method"] == method]
        lines.append(
            f"| {LABEL.get(method, method)} | {len(vals)} | "
            f"{statistics.fmean(float(row['rot_aligned_pos_rmse_m']) for row in vals):.6f} | "
            f"{statistics.fmean(float(row['rot_residual_rmse_deg']) for row in vals):.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []
    for method in args.methods:
        for sequence in args.sequences:
            gt_path = args.gt_root / f"{sequence}.txt"
            est_path = args.run_root / method / sequence / "odomimu.txt"
            metrics = evaluate_pair(gt_path, est_path, args.max_time_diff)
            rows.append(
                {
                    "method": method,
                    "method_label": LABEL.get(method, method),
                    "sequence": sequence,
                    "num_matches": str(metrics["num_matches"]),
                    "rot_aligned_pos_rmse_m": f"{metrics['rot_aligned_pos_rmse_m']:.6f}",
                    "rot_aligned_pos_mean_m": f"{metrics['rot_aligned_pos_mean_m']:.6f}",
                    "rot_residual_rmse_deg": f"{metrics['rot_residual_rmse_deg']:.6f}",
                    "rot_residual_mean_deg": f"{metrics['rot_residual_mean_deg']:.6f}",
                    "constant_rot_correction_deg": f"{metrics['constant_rot_correction_deg']:.6f}",
                }
            )

    args.run_root.mkdir(parents=True, exist_ok=True)
    out_csv = args.run_root / f"{args.output_stem}.csv"
    out_md = args.run_root / f"{args.output_stem}.md"
    write_csv(out_csv, rows)
    write_markdown(out_md, rows, args)
    print(out_md)
    print(out_csv)


if __name__ == "__main__":
    main()
