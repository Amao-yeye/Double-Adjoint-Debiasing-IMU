#!/usr/bin/env python3
"""Summarize OpenVINS odometry recorder outputs for LIASuw visual dropout tests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from summarize_liasuw_openvins_runs import read_gt_asl, rmse


PLATO_ROOT = Path(__file__).resolve().parents[3]


def read_pose_to_file(path: Path) -> dict[str, np.ndarray]:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append([float(x) for x in stripped.split()])
    if not rows:
        raise ValueError(f"No odometry rows found in {path}")
    data = np.asarray(rows, dtype=np.float64)
    if data.shape[1] < 8:
        raise ValueError(f"{path} has too few columns: {data.shape[1]}")
    return {
        "t_s": data[:, 0],
        "p": data[:, 1:4],
        "q_xyzw": data[:, 4:8],
    }


def interp_columns(x_src: np.ndarray, values: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(x_dst, x_src, values[:, idx]) for idx in range(values.shape[1])])


def compute_errors(odom: dict[str, np.ndarray], gt: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = (odom["t_s"] >= gt["t_s"][0]) & (odom["t_s"] <= gt["t_s"][-1])
    t = odom["t_s"][valid]
    p = odom["p"][valid]
    q = odom["q_xyzw"][valid]
    gt_p = interp_columns(gt["t_s"], gt["p"], t)
    gt_q = Slerp(gt["t_s"], Rotation.from_quat(gt["q_xyzw"]))(t).as_quat()
    pos_err = np.linalg.norm(p - gt_p, axis=1)
    rot_err_deg = (Rotation.from_quat(q) * Rotation.from_quat(gt_q).inv()).magnitude() * 180.0 / math.pi
    return t, pos_err, rot_err_deg


def sample_error_at(t: np.ndarray, err: np.ndarray, target_t: float) -> float:
    idx = int(np.argmin(np.abs(t - target_t)))
    return float(err[idx])


def summarize_method(
    method: str,
    odom_path: Path,
    gt: dict[str, np.ndarray],
    dropout_start_abs: float,
    dropout_end_abs: float,
) -> dict[str, object]:
    odom = read_pose_to_file(odom_path)
    t, pos_err, rot_err = compute_errors(odom, gt)
    dropout_mask = (t >= dropout_start_abs) & (t <= dropout_end_abs)
    if not np.any(dropout_mask):
        raise ValueError(f"No odometry samples in dropout window for {method}")
    t_drop = t[dropout_mask]
    pos_drop = pos_err[dropout_mask]
    rot_drop = rot_err[dropout_mask]
    checkpoints = {}
    for seconds in [0, 10, 30, 50, 70]:
        target = dropout_start_abs + seconds
        if target <= t_drop[-1]:
            checkpoints[f"dropout_plus_{seconds:02d}s_pos_m"] = sample_error_at(t, pos_err, target)
            checkpoints[f"dropout_plus_{seconds:02d}s_rot_deg"] = sample_error_at(t, rot_err, target)
    return {
        "method": method,
        "odom_path": str(odom_path),
        "rows": int(t.shape[0]),
        "start_s": float(t[0]),
        "end_s": float(t[-1]),
        "duration_s": float(t[-1] - t[0]),
        "dropout_start_s": float(dropout_start_abs),
        "dropout_end_s": float(dropout_end_abs),
        "dropout_rows": int(t_drop.shape[0]),
        "dropout_position_rmse_m": rmse(pos_drop),
        "dropout_position_final_m": float(pos_drop[-1]),
        "dropout_position_max_m": float(np.max(pos_drop)),
        "dropout_rotation_rmse_deg": rmse(rot_drop),
        "dropout_rotation_final_deg": float(rot_drop[-1]),
        "dropout_rotation_max_deg": float(np.max(rot_drop)),
        "overall_position_rmse_m": rmse(pos_err),
        "overall_position_final_m": float(pos_err[-1]),
        "overall_position_max_m": float(np.max(pos_err)),
        "overall_rotation_rmse_deg": rmse(rot_err),
        "overall_rotation_final_deg": float(rot_err[-1]),
        "overall_rotation_max_deg": float(np.max(rot_err)),
        **checkpoints,
    }


def write_markdown(path: Path, summaries: list[dict[str, object]]) -> None:
    lines = [
        "# LIASuw OpenVINS Visual Dropout Summary",
        "",
        "| method | dropout rows | dropout pos RMSE m | dropout final pos m | dropout rot RMSE deg | dropout final rot deg | overall final pos m | overall final rot deg | +10s pos m | +30s pos m | +50s pos m |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["method"]),
                    str(item["dropout_rows"]),
                    f"{float(item['dropout_position_rmse_m']):.3f}",
                    f"{float(item['dropout_position_final_m']):.3f}",
                    f"{float(item['dropout_rotation_rmse_deg']):.3f}",
                    f"{float(item['dropout_rotation_final_deg']):.3f}",
                    f"{float(item['overall_position_final_m']):.3f}",
                    f"{float(item['overall_rotation_final_deg']):.3f}",
                    f"{float(item.get('dropout_plus_10s_pos_m', math.nan)):.3f}",
                    f"{float(item.get('dropout_plus_30s_pos_m', math.nan)):.3f}",
                    f"{float(item.get('dropout_plus_50s_pos_m', math.nan)):.3f}",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("Dropout interval removes stereo images but keeps IMU. Errors use odomimu propagation output.")
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
    parser.add_argument("--methods", nargs="+", default=["constant", "network"])
    parser.add_argument("--dropout-start-s", type=float, default=5.0)
    parser.add_argument("--dropout-end-s", type=float, default=75.0)
    parser.add_argument("--run-suffix", default="", help="Optional suffix added to each run directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt = read_gt_asl(args.sequence_dir / "groundtruth" / "body_gt_openvins_asl.csv")
    validation = json.loads((args.sequence_dir / "bags" / "bag_validation.json").read_text(encoding="utf-8"))
    bag_start_s = float(validation[0]["start_ns"]) * 1e-9
    dropout_start_abs = bag_start_s + args.dropout_start_s
    dropout_end_abs = bag_start_s + args.dropout_end_s

    summaries = []
    for method in args.methods:
        summaries.append(
            summarize_method(
                method,
                args.sequence_dir
                / "runs"
                / f"dropout_{args.dropout_start_s:g}_{args.dropout_end_s:g}_{method}{args.run_suffix}"
                / "odomimu.txt",
                gt,
                dropout_start_abs,
                dropout_end_abs,
            )
        )
    out_stem = f"dropout_{args.dropout_start_s:g}_{args.dropout_end_s:g}{args.run_suffix}_summary"
    out_json = args.sequence_dir / "runs" / f"{out_stem}.json"
    out_md = args.sequence_dir / "runs" / f"{out_stem}.md"
    out_json.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    write_markdown(out_md, summaries)
    print(out_md)
    for item in summaries:
        print(
            f"{item['method']}: dropout_pos_rmse={item['dropout_position_rmse_m']:.3f}m "
            f"dropout_rot_rmse={item['dropout_rotation_rmse_deg']:.3f}deg "
            f"final_pos={item['dropout_position_final_m']:.3f}m"
        )


if __name__ == "__main__":
    main()
