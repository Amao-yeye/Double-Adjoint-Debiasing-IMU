#!/usr/bin/env python3
"""Formal EuRoC raw-vs-ours short rollout evaluation.

This follows the Altawaitan-style window protocol used in the current task:

* use the EuRoC test split;
* GT is already interpolated to IMU timestamps in Debias_IMU FullTra_result.p;
* sample non-overlapping windows of L IMU integration intervals at 200 Hz;
* initialize every rollout from the GT state at the window start;
* propagate raw IMU or the learned bias-corrected IMU stream;
* average rotation, velocity, and position errors over all sampled test windows
  and all propagated state timestamps inside each window.

The initialized first state is written as an explicit diagnostic, but is not
counted in the primary reported mean.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PLATO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT_DIR = (
    PLATO_ROOT
    / "results"
    / "debias_imu"
    / "EUROC_conservative_residual_train_global_y0_fullrollout_v201_w256_mse_e31"
)
DEFAULT_OUTPUT_DIR = PLATO_ROOT / "new_results" / "Euroc" / "formal_windows"
DEFAULT_TEST_SEQS = ["MH_02_easy", "MH_04_difficult", "V1_01_easy", "V1_03_difficult", "V2_02_medium"]
DEFAULT_LENGTHS = [50, 100, 150, 200]
METHOD_LABELS = {"raw": "raw no bias", "ours": "ours learned bias"}


@dataclass(frozen=True)
class SequenceData:
    sequence: str
    t: np.ndarray
    imu_raw: np.ndarray
    learned_bias: np.ndarray
    R_gt: np.ndarray
    v_gt: np.ndarray
    p_gt: np.ndarray


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


def load_sequence(result_dir: Path, sequence: str) -> SequenceData:
    path = result_dir / "Sequnces-results" / sequence / "FullTra_result.p"
    if not path.exists():
        raise FileNotFoundError(path)
    data = load_pickle(path)
    sol_pred = as_numpy(data["sol_pred"]).astype(np.float64)
    X_gt = as_numpy(data["X_gt"]).astype(np.float64)
    t = as_numpy(data["t_gt"]).astype(np.float64)
    imu_raw = as_numpy(data["imu_raw"]).astype(np.float64)
    learned_bias = np.column_stack([sol_pred[:, 4:7], sol_pred[:, 13:16]])
    if not (t.shape[0] == imu_raw.shape[0] == learned_bias.shape[0] == X_gt.shape[0]):
        raise ValueError(
            f"{sequence} length mismatch: t={t.shape[0]}, imu={imu_raw.shape[0]}, "
            f"bias={learned_bias.shape[0]}, X_gt={X_gt.shape[0]}"
        )
    return SequenceData(sequence, t, imu_raw, learned_bias, X_gt[:, :3, :3], X_gt[:, :3, 3], X_gt[:, :3, 4])


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
        R[i + 1] = R[i] @ so3_exp(omega * dt)
        v[i + 1] = v[i] + acc_world * dt
        p[i + 1] = p[i] + v[i] * dt
    return R, v, p


def state_errors(
    R_gt: np.ndarray,
    v_gt: np.ndarray,
    p_gt: np.ndarray,
    R_est: np.ndarray,
    v_est: np.ndarray,
    p_est: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rot = np.array([np.linalg.norm(so3_log(R_gt[i].T @ R_est[i])) for i in range(R_gt.shape[0])], dtype=np.float64)
    vel = np.linalg.norm(v_est - v_gt, axis=1)
    pos = np.linalg.norm(p_est - p_gt, axis=1)
    return rot, vel, pos


def window_starts(num_states: int, length_steps: int) -> range:
    if length_steps <= 0:
        raise ValueError("Window length must be positive.")
    # L integration intervals require L + 1 states. Starts step by L intervals.
    return range(0, num_states - length_steps, length_steps)


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "rmse": float(np.sqrt(np.mean(arr * arr))),
        "max": float(np.max(arr)),
    }


def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    gravity = np.asarray(args.gravity, dtype=np.float64)
    sequences = [load_sequence(args.result_dir, seq) for seq in args.sequences]
    summary_rows: list[dict[str, Any]] = []
    per_sequence_rows: list[dict[str, Any]] = []
    per_window_rows: list[dict[str, Any]] = []

    for length in args.lengths:
        aggregate: dict[str, dict[str, list[float]]] = {
            method: {"rot": [], "vel": [], "pos": [], "rot_excl": [], "vel_excl": [], "pos_excl": []}
            for method in args.methods
        }
        aggregate_windows = {method: 0 for method in args.methods}
        aggregate_states = {method: 0 for method in args.methods}
        aggregate_states_with_initial = {method: 0 for method in args.methods}

        for data in sequences:
            seq_values: dict[str, dict[str, list[float]]] = {
                method: {"rot": [], "vel": [], "pos": [], "rot_excl": [], "vel_excl": [], "pos_excl": []}
                for method in args.methods
            }
            seq_windows = {method: 0 for method in args.methods}
            seq_states = {method: 0 for method in args.methods}
            seq_states_with_initial = {method: 0 for method in args.methods}
            starts = list(window_starts(data.t.shape[0], length))
            for window_index, start in enumerate(starts):
                stop = start + length + 1
                t = data.t[start:stop]
                for method in args.methods:
                    if method == "raw":
                        bias = np.zeros((length + 1, 6), dtype=np.float64)
                    elif method == "ours":
                        bias = data.learned_bias[start:stop]
                    else:
                        raise ValueError(f"Unknown method: {method}")
                    R_est, v_est, p_est = propagate(
                        t,
                        data.imu_raw[start:stop],
                        bias,
                        data.R_gt[start],
                        data.v_gt[start],
                        data.p_gt[start],
                        gravity,
                    )
                    rot, vel, pos = state_errors(
                        data.R_gt[start:stop],
                        data.v_gt[start:stop],
                        data.p_gt[start:stop],
                        R_est,
                        v_est,
                        p_est,
                    )
                    for target in (aggregate[method], seq_values[method]):
                        target["rot"].extend(rot.tolist())
                        target["vel"].extend(vel.tolist())
                        target["pos"].extend(pos.tolist())
                        target["rot_excl"].extend(rot[1:].tolist())
                        target["vel_excl"].extend(vel[1:].tolist())
                        target["pos_excl"].extend(pos[1:].tolist())
                    aggregate_windows[method] += 1
                    aggregate_states[method] += int(rot[1:].shape[0])
                    aggregate_states_with_initial[method] += int(rot.shape[0])
                    seq_windows[method] += 1
                    seq_states[method] += int(rot[1:].shape[0])
                    seq_states_with_initial[method] += int(rot.shape[0])
                    per_window_rows.append(
                        {
                            "L_steps": length,
                            "states_per_window": length,
                            "states_per_window_including_initial": length + 1,
                            "sequence": data.sequence,
                            "window_index": window_index,
                            "start_index": start,
                            "start_time_s": float(t[0]),
                            "duration_s": float(t[-1] - t[0]),
                            "method": method,
                            "mean_rot_rad": float(np.mean(rot[1:])),
                            "mean_rot_deg": float(np.mean(rot[1:]) * 180.0 / math.pi),
                            "mean_vel_mps": float(np.mean(vel[1:])),
                            "mean_pos_m": float(np.mean(pos[1:])),
                            "mean_rot_rad_including_initial": float(np.mean(rot)),
                            "mean_rot_deg_including_initial": float(np.mean(rot) * 180.0 / math.pi),
                            "mean_vel_mps_including_initial": float(np.mean(vel)),
                            "mean_pos_m_including_initial": float(np.mean(pos)),
                            "endpoint_rot_rad": float(rot[-1]),
                            "endpoint_rot_deg": float(rot[-1] * 180.0 / math.pi),
                            "endpoint_vel_mps": float(vel[-1]),
                            "endpoint_pos_m": float(pos[-1]),
                        }
                    )

            for method in args.methods:
                vals = seq_values[method]
                rot_s = summarize(vals["rot"])
                vel_s = summarize(vals["vel"])
                pos_s = summarize(vals["pos"])
                rot_excl_s = summarize(vals["rot_excl"])
                vel_excl_s = summarize(vals["vel_excl"])
                pos_excl_s = summarize(vals["pos_excl"])
                per_sequence_rows.append(
                    {
                        "L_steps": length,
                        "states_per_window": length,
                        "states_per_window_including_initial": length + 1,
                        "sequence": data.sequence,
                        "method": method,
                        "windows": seq_windows[method],
                        "states": seq_states[method],
                        "states_including_initial": seq_states_with_initial[method],
                        "mean_rot_rad": rot_excl_s["mean"],
                        "mean_rot_deg": rot_excl_s["mean"] * 180.0 / math.pi,
                        "mean_vel_mps": vel_excl_s["mean"],
                        "mean_pos_m": pos_excl_s["mean"],
                        "rmse_rot_rad": rot_s["rmse"],
                        "rmse_rot_deg": rot_s["rmse"] * 180.0 / math.pi,
                        "rmse_vel_mps": vel_s["rmse"],
                        "rmse_pos_m": pos_s["rmse"],
                        "max_rot_rad": rot_s["max"],
                        "max_rot_deg": rot_s["max"] * 180.0 / math.pi,
                        "max_vel_mps": vel_s["max"],
                        "max_pos_m": pos_s["max"],
                        "mean_rot_rad_including_initial": rot_s["mean"],
                        "mean_rot_deg_including_initial": rot_s["mean"] * 180.0 / math.pi,
                        "mean_vel_mps_including_initial": vel_s["mean"],
                        "mean_pos_m_including_initial": pos_s["mean"],
                    }
                )

        for method in args.methods:
            vals = aggregate[method]
            rot_s = summarize(vals["rot"])
            vel_s = summarize(vals["vel"])
            pos_s = summarize(vals["pos"])
            rot_excl_s = summarize(vals["rot_excl"])
            vel_excl_s = summarize(vals["vel_excl"])
            pos_excl_s = summarize(vals["pos_excl"])
            summary_rows.append(
                {
                    "L_steps": length,
                    "states_per_window": length,
                    "states_per_window_including_initial": length + 1,
                    "method": method,
                    "windows": aggregate_windows[method],
                    "states": aggregate_states[method],
                    "states_including_initial": aggregate_states_with_initial[method],
                    "mean_rot_rad": rot_excl_s["mean"],
                    "mean_rot_deg": rot_excl_s["mean"] * 180.0 / math.pi,
                    "mean_vel_mps": vel_excl_s["mean"],
                    "mean_pos_m": pos_excl_s["mean"],
                    "rmse_rot_rad": rot_s["rmse"],
                    "rmse_rot_deg": rot_s["rmse"] * 180.0 / math.pi,
                    "rmse_vel_mps": vel_s["rmse"],
                    "rmse_pos_m": pos_s["rmse"],
                    "max_rot_rad": rot_s["max"],
                    "max_rot_deg": rot_s["max"] * 180.0 / math.pi,
                    "max_vel_mps": vel_s["max"],
                    "max_pos_m": pos_s["max"],
                    "mean_rot_rad_including_initial": rot_s["mean"],
                    "mean_rot_deg_including_initial": rot_s["mean"] * 180.0 / math.pi,
                    "mean_vel_mps_including_initial": vel_s["mean"],
                    "mean_pos_m_including_initial": pos_s["mean"],
                }
            )

    return summary_rows, per_sequence_rows, per_window_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    args: argparse.Namespace,
    summary_rows: list[dict[str, Any]],
    per_sequence_rows: list[dict[str, Any]],
    per_window_rows: list[dict[str, Any]],
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "per_sequence_summary.csv", per_sequence_rows)
    write_csv(args.output_dir / "per_window_summary.csv", per_window_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "dataset": "Euroc",
        "result_dir": str(args.result_dir),
        "sequences": args.sequences,
        "lengths": args.lengths,
        "methods": args.methods,
        "gravity": list(args.gravity),
        "window_definition": "L integration intervals at 200 Hz; each window has L+1 state timestamps.",
        "sampling": "non-overlapping starts at 0, L, 2L, ...; trailing incomplete windows are dropped.",
        "reported_mean": "mean over all sampled test windows and propagated state timestamps; the initialized first state is excluded from primary mean columns.",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Euroc Formal Rollout Window Evaluation",
        "",
        "Errors are averaged over all non-overlapping test windows and propagated state timestamps inside each window.",
        "Rotation error is `||Log(R_gt^T R_hat)||`. The initialized first state is excluded from the primary mean columns; including-initial diagnostics are saved in CSV.",
        "",
        f"Ours source: `{args.result_dir}`",
        "",
        "| L steps | method | windows | states | mean rot rad | mean rot deg | mean vel m/s | mean pos m |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['L_steps']} | {row['method']} | {row['windows']} | {row['states']} | "
            f"{row['mean_rot_rad']:.6f} | {row['mean_rot_deg']:.4f} | "
            f"{row['mean_vel_mps']:.4f} | {row['mean_pos_m']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Files:",
            "",
            "- `summary.csv`: aggregate over all test sequences.",
            "- `per_sequence_summary.csv`: aggregate per sequence.",
            "- `per_window_summary.csv`: one row per sampled window and method.",
            "- `metadata.json`: protocol details.",
        ]
    )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate formal EuRoC non-overlapping IMU rollout windows.")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_TEST_SEQS)
    parser.add_argument("--lengths", nargs="+", type=int, default=DEFAULT_LENGTHS)
    parser.add_argument("--methods", nargs="+", choices=["raw", "ours"], default=["raw", "ours"])
    parser.add_argument("--gravity", nargs=3, type=float, default=(0.0, 0.0, -9.81), metavar=("GX", "GY", "GZ"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(length <= 0 for length in args.lengths):
        raise ValueError("--lengths must be positive.")
    summary_rows, per_sequence_rows, per_window_rows = evaluate(args)
    write_outputs(args, summary_rows, per_sequence_rows, per_window_rows)
    print(f"Wrote formal EuRoC window evaluation to {args.output_dir}")
    print(args.output_dir / "summary.md")


if __name__ == "__main__":
    main()
