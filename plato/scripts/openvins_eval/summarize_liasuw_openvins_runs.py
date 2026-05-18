#!/usr/bin/env python3
"""Summarize LIASuw OpenVINS estimate files against IMU/body GT."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


PLATO_ROOT = Path(__file__).resolve().parents[3]


def read_openvins_estimate(path: Path) -> dict[str, np.ndarray]:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append([float(x) for x in stripped.split()])
    if not rows:
        raise ValueError(f"No estimate rows found in {path}")
    data = np.asarray(rows, dtype=np.float64)
    if data.shape[1] < 17:
        raise ValueError(f"{path} has too few columns: {data.shape[1]}")
    return {
        "t_s": data[:, 0],
        "q_xyzw": data[:, 1:5],
        "p": data[:, 5:8],
        "v": data[:, 8:11],
        "bg": data[:, 11:14],
        "ba": data[:, 14:17],
    }


def read_gt_asl(path: Path) -> dict[str, np.ndarray]:
    t_ns: list[int] = []
    values: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            t_ns.append(int(row[0].strip()))
            values.append([float(x) for x in row[1:17]])
    if not t_ns:
        raise ValueError(f"No GT rows found in {path}")
    data = np.asarray(values, dtype=np.float64)
    q_wxyz = data[:, 3:7]
    return {
        "t_s": np.asarray(t_ns, dtype=np.float64) * 1e-9,
        "p": data[:, 0:3],
        "q_xyzw": q_wxyz[:, [1, 2, 3, 0]],
        "v": data[:, 7:10],
    }


def interp_columns(x_src: np.ndarray, values: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(x_dst, x_src, values[:, idx]) for idx in range(values.shape[1])])


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else math.nan


def summarize_run(method: str, estimate_path: Path, gt: dict[str, np.ndarray]) -> dict[str, object]:
    est = read_openvins_estimate(estimate_path)
    valid = (est["t_s"] >= gt["t_s"][0]) & (est["t_s"] <= gt["t_s"][-1])
    if not np.any(valid):
        raise ValueError(f"No estimate rows overlap GT for {method}")
    t = est["t_s"][valid]
    est_p = est["p"][valid]
    est_v = est["v"][valid]
    est_q = est["q_xyzw"][valid]

    gt_p = interp_columns(gt["t_s"], gt["p"], t)
    gt_v = interp_columns(gt["t_s"], gt["v"], t)
    gt_q = Slerp(gt["t_s"], Rotation.from_quat(gt["q_xyzw"]))(t).as_quat()

    pos_err = np.linalg.norm(est_p - gt_p, axis=1)
    vel_err = np.linalg.norm(est_v - gt_v, axis=1)
    rot_err_deg = (Rotation.from_quat(est_q) * Rotation.from_quat(gt_q).inv()).magnitude() * 180.0 / math.pi

    start_s = float(t[0])
    end_s = float(t[-1])
    return {
        "method": method,
        "estimate_path": str(estimate_path),
        "rows_in_file": int(read_openvins_estimate(estimate_path)["t_s"].shape[0]),
        "rows_used": int(t.shape[0]),
        "start_s": start_s,
        "end_s": end_s,
        "duration_s": float(end_s - start_s),
        "position_rmse_m": rmse(pos_err),
        "position_mean_m": float(np.mean(pos_err)),
        "position_median_m": float(np.median(pos_err)),
        "position_max_m": float(np.max(pos_err)),
        "rotation_rmse_deg": rmse(rot_err_deg),
        "rotation_mean_deg": float(np.mean(rot_err_deg)),
        "rotation_median_deg": float(np.median(rot_err_deg)),
        "rotation_max_deg": float(np.max(rot_err_deg)),
        "velocity_rmse_mps": rmse(vel_err),
        "velocity_mean_mps": float(np.mean(vel_err)),
        "velocity_max_mps": float(np.max(vel_err)),
        "final_position_error_m": float(pos_err[-1]),
        "final_rotation_error_deg": float(rot_err_deg[-1]),
        "final_velocity_error_mps": float(vel_err[-1]),
        "final_est_position_m": est_p[-1].tolist(),
        "final_gt_position_m": gt_p[-1].tolist(),
        "final_est_bg_rad_s": est["bg"][valid][-1].tolist(),
        "final_est_ba_m_s2": est["ba"][valid][-1].tolist(),
    }


def write_markdown(path: Path, summaries: list[dict[str, object]]) -> None:
    headers = [
        "method",
        "duration_s",
        "rows",
        "pos_rmse_m",
        "pos_final_m",
        "rot_rmse_deg",
        "rot_final_deg",
        "vel_rmse_mps",
        "vel_final_mps",
    ]
    lines = [
        "# LIASuw OpenVINS GT-init Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for item in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["method"]),
                    f"{float(item['duration_s']):.3f}",
                    str(item["rows_used"]),
                    f"{float(item['position_rmse_m']):.4f}",
                    f"{float(item['final_position_error_m']):.4f}",
                    f"{float(item['rotation_rmse_deg']):.3f}",
                    f"{float(item['final_rotation_error_deg']):.3f}",
                    f"{float(item['velocity_rmse_mps']):.4f}",
                    f"{float(item['final_velocity_error_mps']):.4f}",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "`raw`, `constant`, and `network` use zero-bias GT-init. "
        "`raw_constbias` uses the same pose/velocity GT-init but initializes raw-IMU "
        "bias to the constant-bias estimate, which should be equivalent to the "
        "constant-corrected IMU stream. Raw zero-bias ended early because its "
        "estimate diverged."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        default=PLATO_ROOT
        / "results"
        / "openvins_eval"
        / "LIASuw"
        / "phys_delta_vo_framefixed_clean_imu"
        / "0205s14_80",
    )
    parser.add_argument("--methods", nargs="+", default=["raw", "constant", "network"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt = read_gt_asl(args.sequence_dir / "groundtruth" / "body_gt_openvins_asl.csv")
    summaries = []
    for method in args.methods:
        estimate_path = args.sequence_dir / "runs" / f"gtinit_full_{method}" / "estimate.txt"
        summaries.append(summarize_run(method, estimate_path, gt))

    out_json = args.sequence_dir / "runs" / "gtinit_full_summary.json"
    out_md = args.sequence_dir / "runs" / "gtinit_full_summary.md"
    out_json.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    write_markdown(out_md, summaries)
    print(out_md)
    for item in summaries:
        print(
            f"{item['method']}: duration={item['duration_s']:.3f}s "
            f"pos_rmse={item['position_rmse_m']:.4f}m "
            f"rot_rmse={item['rotation_rmse_deg']:.3f}deg "
            f"vel_rmse={item['velocity_rmse_mps']:.4f}m/s"
        )


if __name__ == "__main__":
    main()
